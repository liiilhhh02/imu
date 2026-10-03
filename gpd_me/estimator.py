"""Learned estimator for the out-of-range body rate — implementation.

Architecture (why this shape)
-----------------------------
The algebraic observer already reconstructs the rate to 0.04 rad/s wherever its two ingredients are
good (identified lever-arm scale + quasi-steady spin).  What it cannot do is the *transient*
(`wdot x r` is dropped), the multi-axis-saturated case (the saturated block is distributed by a
heuristic ratio) and low-rate noise.  So the network is a **residual model on top of the algebraic
estimate**:

    w_hat = w_alg(s, gyro, k_hat) + delta_w(network)

which guarantees it cannot be worse than the physics in the well-conditioned regime.

Inputs per frame (F features, all *measurable* plus the identified priors):
    s (3)            accelerometer residual  a_m - (T/M) e_z
    gyro (3)         raw gyro (clipped on the saturated axes)
    sat (3)          saturation mask (which axes are pinned -> not real measurements)
    u (4)            masked thrust command
    T/M (1)
    w_alg (3)        the algebraic lever-arm estimate
    k_hat (1)        identified |r_perp|
    sat_axis ratio (1)  how far inside the range the unsaturated axes are (a confidence proxy)
  -> F = 19

Temporal model: GRU over an H-frame window (H = 12 -> 60 ms at 200 Hz), 64 hidden units, MLP head.

Losses
------
  L_rate   |w_hat - w_true|^2 on saturated steps              (supervision, simulation only)
  L_ins    tilt error of a *differentiable INS rollout* driven by w_hat over the window
           (the main experiment showed the attitude channel is what decides the flight)
  L_phys   |a_m - (T/M)e_z - wdot_hat x r_hat - w_hat x (w_hat x r_hat)|^2  (no labels needed)
  L_torque |J_z wdot_z - KM (T0-T1+T2-T3)|^2                               (no labels needed)
  L_anchor |w_hat - gyro|^2 on *in-range* steps             (self-supervised, also deployable)

L_anchor is what makes the in-range segment the "reference" the user asked for: there the gyro is
exact, so the network is forced to reproduce it while consuming the same features it will use when
the gyro is pinned.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_FEATURES = 19
WINDOW = 12


def skew(w: torch.Tensor) -> torch.Tensor:
    """Batch of skew matrices from a (...,3) vector."""
    z = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([z, -w[..., 2], w[..., 1]], -1),
        torch.stack([w[..., 2], z, -w[..., 0]], -1),
        torch.stack([-w[..., 1], w[..., 0], z], -1),
    ], -2)


def rodrigues(w: torch.Tensor, dt: float) -> torch.Tensor:
    """Differentiable exp([w]_x dt) for a batch of (...,3) rotation vectors."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    K = skew(k)
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    s = torch.sin(th * dt)[..., None]
    c = (1.0 - torch.cos(th * dt))[..., None]
    return eye + s * K + c * (K @ K)


def ins_rollout(omega_hat: torch.Tensor, R0: torch.Tensor, dt: float) -> torch.Tensor:
    """Propagate R0 through a (B, H, 3) rate sequence; returns (B, H, 3, 3) attitudes."""
    Rs = [R0]
    for k in range(omega_hat.shape[1]):
        Rs.append(Rs[-1] @ rodrigues(omega_hat[:, k], dt))
    return torch.stack(Rs[1:], dim=1)


def tilt_error_deg(R_hat: torch.Tensor, R_true: torch.Tensor) -> torch.Tensor:
    """Angle between the two thrust axes (body z), in degrees."""
    ze = R_hat[..., 2]
    zt = R_true[..., 2]
    c = (ze * zt).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(c))


class EstimatorNet(nn.Module):
    """GRU + MLP head predicting a residual on the algebraic lever-arm estimate."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64, window: int = WINDOW):
        super().__init__()
        self.window = window
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor, w_alg: torch.Tensor) -> torch.Tensor:
        """x: (B, H, F) features; w_alg: (B, H, 3) algebraic estimate -> (B, H, 3) estimate."""
        h, _ = self.gru(x)
        return w_alg + self.head(h)

    @staticmethod
    def build_features(s, gyro, sat, u, tom, w_alg, k_hat, lim):
        """Assemble the (H, F) feature matrix for one window.

        All arguments are (H, ...) arrays; `lim` is the gyro range in rad/s.
        """
        conf = (np.abs(gyro).max(axis=1, keepdims=True) / max(lim, 1e-9)).clip(0.0, 1.5)
        return np.concatenate([
            s, gyro, sat.astype(float), u, tom[:, None], w_alg, np.full_like(tom[:, None], k_hat),
            conf,
        ], axis=1).astype(np.float32)


import numpy as np  # noqa: E402  (kept last: build_features uses it, module stays import-light)