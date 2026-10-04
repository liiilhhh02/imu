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

from .priors import tom_from_command

WINDOW = 96                      # 480 ms at 200 Hz.  Was 48 (240 ms); the temporal model is the
#                                 strong estimator here (it beats the true-k oracle), and under
#                                 multi-axis saturation the spin *direction* is fixed in the body frame
#                                 while only the magnitude is observable -- so a longer window is the
#                                 principled way to average the direction out.  The GRU weights do not
#                                 depend on the window length, so this is a layout-compatible change.
COARSE = 17                      # causal 2 s summary: 5 aggregates + 3 per-axis sat fracs + 9 priors
N_PRIOR = 9                      # r(3), g_T, G, T, range, tau, k
N_CORR = 6                       # dr(3), dg_T, dG, dT
FINE_FEATURES = 3 + 1 + 3 + 3 + 4 + 1 + N_PRIOR + 1 + 3 + 1      # 29
# feature layout: s(0:3) | |s|(3) | gyro(4:7) | sat(7:10) | u(10:14) | T/M(14) | priors(15:24)
#                 | |gyro|(24) | w_alg(25:28) | w_z_model(28)
# The last column is the *yaw* channel's own physics: the identified ARX `T*dwz/dt + wz = G*sum(+-u)`
# predicts w_z from the command alone, i.e. an independent estimate of the axis that saturates most
# often -- a one-step-ahead model prediction, available on a real vehicle, appended last so every
# existing slice (GYRO_SLICE, the prior block) keeps its index.
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
    # yaw-model feature: the identified ARX's *equilibrium* yaw rate for the commanded differential,
    # `w_z_ss = G * sum(+-u)`.  Deliberately dt-free (a one-step prediction would need the per-window dt
    # threaded into this function) and deliberately an equilibrium: the identified time constant is
    # ~2.6 s, so this is a slow, independent estimate of the axis that saturates most often -- the
    # network decides how far to trust it.  Falls back to the clipped gyro when G was not identified.
    pri = np.asarray(prior, float)
    G_p = float(pri[4])
    if np.isfinite(G_p) and G_p > 0:
        yaw_cmd = (u_cmd * np.array([1.0, -1.0, 1.0, -1.0])[None, :]).sum(axis=1)
        wz_m = G_p * yaw_cmd
    else:
        wz_m = np.asarray(gyro, float)[:, 2]
    return np.concatenate([
        s, np.linalg.norm(s, axis=1)[:, None], gyro, sat.astype(float), u_cmd, tom_hat[:, None],
        np.tile(np.asarray(prior, float)[None, :], (H, 1)),
        np.linalg.norm(gyro, axis=1)[:, None], w_alg, wz_m[:, None],
    ], axis=1).astype(np.float32)


def coarse_summary(gyro, sat, u_cmd, tom_hat, prior):
    """Causal 2 s summary, exactly COARSE dims: 5 aggregates + 3 per-axis saturation fracs + 9 priors."""
    agg = np.array([tom_hat.mean(), tom_hat.std(), np.abs(gyro).mean(), np.abs(gyro).std(),
                    np.abs(u_cmd).mean()], float)
    return np.concatenate([agg, sat.mean(axis=0), np.asarray(prior, float)]).astype(np.float32)


# ----------------------------------------------------------------------------------------- the model
DT = 1.0 / 200.0        # control period of the deployment loop (both sims)
_WARNED: set = set()    # checkpoints whose incompatibility has already been announced


class NetRate:
    """Causal wrapper around the trained estimator: keeps a rolling window and returns w_hat."""

    def __init__(self, ckpt, dev, prior, dps):
        self.dev = dev
        self.prior = np.asarray(prior, float)
        # prior = [r(3), g_T, G, T, range_rad, tau, k]; the algebraic front end needs k = |r_perp|
        self.g_T = float(self.prior[3]); self.tau = float(self.prior[7]); self.k = float(self.prior[8])
        self.lim = np.deg2rad(dps)
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        # the checkpoint records the architecture (train_e2e saves `hidden`/`layers`); rebuild the same
        # shape or `load_state_dict` fails on a v7/v8 checkpoint.  Old checkpoints carry no such keys.
        ck_hidden, ck_layers = int(ck.get("hidden", 128)), int(ck.get("layers", 1))
        self.net = E2ENet(hidden=ck_hidden, layers=ck_layers).to(dev).eval()
        try:
            self.net.load_state_dict(ck["state"])
            self.compat = True
            # the checkpoint stores numpy float64 normalisation constants -> cast, or the first
            # Linear sees float64 and torch aborts with "mat1 and mat2 must have the same dtype"
            self.xf_m = torch.tensor(ck["xf_m"], device=dev, dtype=torch.float32)
            self.xf_s = torch.tensor(ck["xf_s"], device=dev, dtype=torch.float32)
            self.xp_m = torch.tensor(ck["xp_m"], device=dev, dtype=torch.float32)
            self.xp_s = torch.tensor(ck["xp_s"], device=dev, dtype=torch.float32)
        except RuntimeError as exc:
            # The checkpoint predates the current feature/prior layout (its fine vector is 26 wide
            # with 7 priors; the current one is 28/9).  It cannot be loaded, so the residual heads
            # stay at their zero init and w_hat reduces to the algebraic physics front end -- the
            # network output is only valid again after retraining.  Kept loud, not silent.
            self.compat = False
            msg = next((ln.strip() for ln in str(exc).splitlines() if "size mismatch" in ln),
                       str(exc).splitlines()[0].strip())
            if ckpt not in _WARNED:
                _WARNED.add(ckpt)
                print(f"[e4] WARNING: {ckpt} was trained on the OLD 26-feature / 7-prior layout and "
                      f"cannot be loaded ({msg}).  The 'net' row falls back to the algebraic "
                      f"estimate; retrain to make it valid.")
            F = fine_features(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), bool),
                              np.zeros((1, 4)), np.zeros(1), self.prior, np.zeros((1, 3))).shape[1]
            self.xf_m = torch.zeros(F, device=dev); self.xf_s = torch.ones(F, device=dev)
            self.xp_m = torch.zeros(len(self.prior), device=dev)
            self.xp_s = torch.ones(len(self.prior), device=dev)
        self.buf = {k: [] for k in ("a", "g", "u", "m")}

    def reset(self):
        """Empty the rolling window.  A new flight starts with no history; without this the
        first WINDOW steps of every episode replay the *previous* episode's tail (and the
        coarse summary silently becomes cross-episode), which is also an unbounded memory leak
        over a multi-hour fine-tune."""
        self.buf = {k: [] for k in ("a", "g", "u", "m")}

    def preseed(self, gyro, accel, u_cmd, mask):
        """Fill the rolling window with the real pre-fault samples.

        Without this the window starts by repeating the first *post-fault* sample, i.e. the
        estimator spends its first ~WINDOW steps looking at a fake constant history exactly while
        the vehicle is spinning at 40+ rad/s and the controller has to break that spin.  A real
        vehicle has the pre-fault history (unsaturated gyro, accelerometer and commands -- all
        measured), so using it is both legitimate and strictly more informative.
        """
        for key, val in (("g", gyro), ("a", accel), ("u", u_cmd), ("m", mask)):
            arr = np.asarray(val, float)
            self.buf[key] = [arr[i] for i in range(len(arr))]


    def step(self, accel, gyro, u_cmd, mask):
        """One causal estimator step: returns w_hat for the current instant.

        `u_cmd` is the *commanded* per-rotor thrust (N) this controller sent and `mask` the
        per-sample alive-rotor mask; both feed the identified actuator model, so the specific thrust
        the estimator uses is `tom_from_command(...)`, never the simulator's true thrust.
        """
        for key, val in (("a", accel), ("g", gyro), ("u", u_cmd), ("m", mask)):
            self.buf[key].append(np.asarray(val, float))
        H = min(WINDOW, len(self.buf["a"]))
        pad = WINDOW - H
        rep = lambda x: np.concatenate([np.repeat(x[:1], pad, 0), x], 0) if pad else x
        a_ = rep(np.stack(self.buf["a"][-H:])); g_ = rep(np.stack(self.buf["g"][-H:]))
        u_ = rep(np.stack(self.buf["u"][-H:])); m_ = rep(np.stack(self.buf["m"][-H:]))
        sat_ = np.abs(g_) >= self.lim - 1e-9
        tom_ = tom_from_command(u_, m_, DT, self.g_T, self.tau)
        w_alg = algebraic_estimate(g_, a_ - np.array([0.0, 0.0, 1.0]) * tom_[:, None],
                                   self.k, sat_, self.lim)
        # coarse summary over the trailing 2 s (padded for the first samples)
        Na = len(self.buf["a"]); Hc = min(400, Na); repc = lambda x: np.concatenate(
            [np.repeat(x[:1], 400 - Hc, 0), x], 0) if 400 - Hc else x
        cg = repc(np.stack(self.buf["g"][-Hc:])); cm = repc(np.stack(self.buf["m"][-Hc:]))
        cu = repc(np.stack(self.buf["u"][-Hc:]))
        ctom_ = tom_from_command(cu, cm, DT, self.g_T, self.tau)
        cs = coarse_summary(cg, np.abs(cg) >= self.lim - 1e-9, cu, ctom_, self.prior)
        T_ = lambda x: torch.tensor(np.asarray(x)[None], dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            ft = ((T_(fine_features(a_, g_, sat_, u_, tom_, self.prior, w_alg)) - self.xf_m)
                  / self.xf_s).clamp(-50, 50)
            pt = (T_(self.prior) - self.xp_m) / self.xp_s
            w_hat, _ = self.net(ft, T_(cs), pt, T_(w_alg), torch.tensor(sat_[None], device=self.dev))
        return w_hat[0, -1].cpu().numpy().astype(float)


def deploy_obs(rel, rate, ta, tom, last_action, mask):
    """The observation the senior's policy actually receives at deployment.

    Written once and shared by the fine-tuning rollout and the acceptance harness so the two cannot
    drift apart field by field -- a silent mismatch here would make fine-tuning non-transferable and
    would look like "the policy cannot learn".  Mirrors the inline construction in
    ``scripts/e4_closed_loop.py`` (asserted bit-identical by ``scripts/test_deploy_obs.py``):

        [ rel_x, rel_y | rate/10, rate/10, rate/50 | (ta-9.8)/3, (tom-9.8)/3
          | last_action * mask | mask*2-1 ]

    ``rel`` is the target body-z direction expressed in the *estimated* attitude, i.e.
    ``R_hat.T @ z_body``; ``rate`` is whatever the estimator produced; ``tom`` is the measured
    thrust-over-mass (the accelerometer's x sample), never a commanded or simulated quantity.
    """
    # Dims 7:11 MUST reproduce the senior's own assembly, bug-for-bug: MetaShutDown7._computeObs
    # (gym_pybullet_drones/envs/MetaShutDown7.py:174) writes `self.last_action[0] * self.shut_down`,
    # i.e. the FIRST element of the action vector broadcast through the mask -- not the action vector
    # masked element-wise.  The published checkpoints were trained on that distribution, so feeding
    # the element-wise form (which this function and e4_closed_loop.py did until an external audit's
    # criticism of a weak test exposed it: scripts/test_obs_convention.py) puts 2 of 15 inputs out of
    # distribution.  Broadcast the scalar exactly as the environment does.
    la = np.asarray(last_action, float).ravel()
    return np.array([rel[0], rel[1],
                     rate[0] / 10.0, rate[1] / 10.0, rate[2] / 50.0,
                     (ta - 9.8) / 3.0, (tom - 9.8) / 3.0,
                     *(float(la[0]) * np.asarray(mask, float)),
                     *(np.asarray(mask, float) * 2.0 - 1.0)], dtype=np.float32)


class E2ENet(nn.Module):
    def __init__(self, fine_features: int = FINE_FEATURES, coarse: int = COARSE,
                 hidden: int = 128, n_prior: int = N_PRIOR, n_corr: int = N_CORR,
                 layers: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(fine_features, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        # `layers` lets the same recipe scale the temporal model: 140 k parameters (1 layer, 128) is
        # already deployable (137 KB int8), so a 2-layer/192-hidden variant (~600 k, ~600 KB int8) is
        # still inside an onboard budget and is the natural next step when the metric is capacity-bound.
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
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