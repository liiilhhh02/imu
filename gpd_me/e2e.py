"""End-to-end estimator, v2 — physics-parameterised output, longer horizon, stable losses.

What changed from v1, and why (this is the design review the user asked for)

1. **The algebra is computed inside the system, not learned.**  `w_alg` is the closed-form
   lever-arm inversion evaluated from *measurable* quantities only (`s`, `gyro`, the saturation
   mask and the identified `k`).  It is not privileged information — any real system can compute it
   — and handing it over means the network does not have to relearn a square root.  The network then
   predicts the **residual beyond the physics**, which is where the actual difficulty lives
   (transients, multi-axis saturation, identification error).
2. **Physics-parameterised output.**  The unsaturated gyro axes are exact measurements, so the
   output is
       w_hat = w_alg + sat * delta      (only the *saturated* axes may be corrected)
   which removes ~all wasted capacity and makes the in-range anchor trivially satisfiable.
3. **The accelerometer DC is removed.**  The feature is the residual `s = a_m - (T/M) e_z` (plus
   its norm), not the raw accelerometer whose 9.8 m/s^2 thrust offset swamps the lever-arm term.
4. **A longer horizon.**  The closed-loop main experiment showed the *accumulated* attitude error
   is what kills the vehicle, so the INS rollout is evaluated over 48 frames (240 ms) instead of 24.
5. **Stable losses.**  No heteroscedastic NLL (it was ill-posed for this problem); Huber on the
   saturated frames, Huber on the INS tilt error, plus the unlabelled physics / torque / anchor /
   spectral terms.  The slow (parameter) head is **detached in stage 1** so its random early output
   cannot corrupt the physics and torque losses, and supervised in stage 2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

WINDOW = 48                      # 240 ms at 200 Hz (the rate is 95 % below 7.1 Hz; drift accumulates)
COARSE = 17                      # causal 2 s summary: 5 aggregates + 3 per-axis sat fracs + 9 priors
N_PRIOR = 9                      # r(3), g_T, G, T, range, tau, k
N_CORR = 6                       # dr(3), dg_T, dG, dT
FINE_FEATURES = 3 + 1 + 3 + 3 + 4 + 1 + N_PRIOR + 1 + 3      # 28
# feature layout: s(0:3) | |s|(3) | gyro(4:7) | sat(7:10) | u(10:14) | T/M(14) | priors(15:24)
#                 | |gyro|(24) | w_alg(25:28)
# `s` and `T/M` are built from the *command* (identified g_T + identified actuator lag), never from
# the simulator's true thrust; `k = |r_perp|` (prior[8]) is the lever scale the algebra needs.
GYRO_SLICE = slice(4, 7)        # <- the only correct way to read the measured rate back out


# --------------------------------------------------------------------------- differentiable helpers
def skew(w: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([z, -w[..., 2], w[..., 1]], -1),
        torch.stack([w[..., 2], z, -w[..., 0]], -1),
        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def rodrigues(w: torch.Tensor, dt: float) -> torch.Tensor:
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K = skew(w / th)
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + torch.sin(th * dt)[..., None] * K + (1.0 - torch.cos(th * dt))[..., None] * (K @ K)


def rodrigues_dt(w: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """exp([w]_x * dt) with a per-sample dt (the shards are multi-rate: 100-400 Hz)."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    K = skew(w / th)
    a = th * dt[:, None]
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + torch.sin(a)[..., None] * K + (1.0 - torch.cos(a))[..., None] * (K @ K)


def ins_rollout_dt(omega: torch.Tensor, R0: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """(B,H,3) rates with a per-sample (B,) timestep -> (B,H,3,3) attitudes."""
    Rs = [R0]
    for k in range(omega.shape[1]):
        Rs.append(Rs[-1] @ rodrigues_dt(omega[:, k], dt))
    return torch.stack(Rs[1:], dim=1)


def ins_rollout(omega: torch.Tensor, R0: torch.Tensor, dt: float) -> torch.Tensor:
    Rs = [R0]
    for k in range(omega.shape[1]):
        Rs.append(Rs[-1] @ rodrigues(omega[:, k], dt))
    return torch.stack(Rs[1:], dim=1)


def att_errors_deg(R_hat: torch.Tensor, R_true: torch.Tensor, eps: float = 1e-6):
    # arccos has an infinite derivative at +-1, so clamp strictly inside: clamp(-1, 1) gives inf grads
    lo, hi = -1.0 + eps, 1.0 - eps
    tilt = torch.rad2deg(torch.arccos((R_hat[..., 2] * R_true[..., 2]).sum(-1).clamp(lo, hi)))
    tr = (R_hat.transpose(-1, -2) @ R_true).diagonal(dim1=-2, dim2=-1).sum(-1)
    return tilt, torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(lo, hi)))


def make_dft(h: int, dt: float) -> torch.Tensor:
    n = torch.arange(h).float()
    k = torch.arange(h // 2 + 1).float()[:, None]
    return torch.cos(2 * np.pi * k * n / h), torch.sin(2 * np.pi * k * n / h)


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    a = x.abs()
    return torch.where(a < delta, 0.5 * a * a, delta * (a - 0.5 * delta))


# ------------------------------------------------------------------- the algebraic (physics) front end
def algebraic_estimate(gyro: np.ndarray, s: np.ndarray, k, sat: np.ndarray,
                       lim: float) -> np.ndarray:
    """Closed-form lever-arm inversion from measurable quantities: `|w|^2 = |s|/k`.

    Unsaturated axes keep the gyro value (they are exact measurements); saturated axes are recovered
    from the invariant, the measured axes and the clipped sign (ratio preservation when several axes
    are pinned).  `k` is a scalar: the scale the identification minimised (`prior[8]`, itself a
    `|r_perp|` for the dominant spin direction -- *not* `‖r‖`).

    A per-sample `|r_perp|` computed from the identified vector was tried and **removed again**: it
    inherits the 20-50 % error of the identified `r` and measurably loses to the self-calibrated
    scalar (80 held-out flights, per-flight lag: scalar 4.77, per-sample 5.27, hybrid 5.04 rad/s;
    the per-sample version only wins above 1500 dps, where the clip is already nearly exact).

    A sample whose scale is unknown, or whose invariant says the saturated magnitude is *below* what
    the unsaturated axes already imply, keeps the clipped gyro: never worse than the plain clip.
    Returns (N,3).
    """
    w = np.array(gyro, float, copy=True)
    k_arr = np.asarray(k, float).ravel()
    ks = (np.full(len(w), float(k_arr[0])) if k_arr.size == 1 else k_arr)
    n2 = np.linalg.norm(s, axis=1) / np.where(np.abs(ks) > 1e-12, ks, np.nan)
    n2 = np.where(np.isfinite(n2), np.maximum(n2, 0.0), np.nan)
    for i in np.where(sat.any(axis=1))[0]:
        if not np.isfinite(n2[i]):
            continue                                   # unknown scale -> keep the clip
        m = sat[i]
        known = float(np.sum(gyro[i, ~m] ** 2))
        need = float(n2[i] - known)
        if need <= 0.0:
            continue                                   # clip already >= the invariant -> keep it
        if m.sum() == 1:
            w[i, m] = np.sign(gyro[i, m]) * np.sqrt(need)
        else:
            base = float(np.linalg.norm(gyro[i, m]))
            if base < 1e-9:
                w[i, m] = np.sign(gyro[i, m]) * np.sqrt(need / m.sum())
            else:
                w[i, m] = gyro[i, m] * (np.sqrt(need) / base)
    return w


def fine_features(accel, gyro, sat, u_cmd, tom_hat, prior, w_alg):
    """(H, FINE_FEATURES): residual + norm + raw gyro + mask + command + priors + algebra + norms.

    `tom_hat` must be the *command-derived* specific thrust (identified g_T through the identified
    actuator lag).  Feeding the simulator's true thrust here would hand the network the very nuisance
    term the estimator is supposed to reconstruct.
    """
    s = accel - np.array([0.0, 0.0, 1.0])[None, :] * tom_hat[:, None]
    H = len(accel)
    return np.concatenate([
        s, np.linalg.norm(s, axis=1)[:, None], gyro, sat.astype(float), u_cmd, tom_hat[:, None],
        np.tile(np.asarray(prior, float)[None, :], (H, 1)),
        np.linalg.norm(gyro, axis=1)[:, None], w_alg,
    ], axis=1).astype(np.float32)


def coarse_summary(gyro, sat, u_cmd, tom_hat, prior):
    """Causal 2 s summary, exactly COARSE dims: 5 aggregates + 3 per-axis saturation fracs + 9 priors."""
    agg = np.array([tom_hat.mean(), tom_hat.std(), np.abs(gyro).mean(), np.abs(gyro).std(),
                    np.abs(u_cmd).mean()], float)
    return np.concatenate([agg, sat.mean(axis=0), np.asarray(prior, float)]).astype(np.float32)


# ----------------------------------------------------------------------------------------- the model
class E2ENet(nn.Module):
    def __init__(self, fine_features: int = FINE_FEATURES, coarse: int = COARSE,
                 hidden: int = 128, n_prior: int = N_PRIOR, n_corr: int = N_CORR):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(fine_features, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fast = nn.Linear(hidden, 3)                    # residual on the SATURATED axes only
        nn.init.zeros_(self.fast.weight); nn.init.zeros_(self.fast.bias)   # start exactly at physics
        self.slow = nn.Sequential(nn.Linear(hidden + coarse + n_prior, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_corr))
        nn.init.zeros_(self.slow[-1].weight); nn.init.zeros_(self.slow[-1].bias)
        self.n_prior, self.n_corr = n_prior, n_corr

    def forward(self, fine, coarse, prior_norm, w_alg, sat):
        """-> w_hat (B,H,3), corr (B,C6).  w_hat = w_alg + sat * delta, so it starts at the physics."""
        h, _ = self.gru(self.encoder(fine))
        delta = self.fast(h)
        w_hat = w_alg + sat.to(delta.dtype) * delta
        corr = self.slow(torch.cat([h[:, -1], coarse, prior_norm], dim=-1))
        return w_hat, corr