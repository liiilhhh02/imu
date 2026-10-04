"""Bound the INS's drift with the *thrust axis* measured from inertial acceleration.

`AttitudeINS` integrates a body rate whose error is, in the saturated regime, a bias-like few
rad/s; integrating a bias gives an attitude error that grows without limit, and the closed loop
dies of attitude divergence -- that is the failure mechanism this repository measured
(docs/STATUS.md).  The estimator-side attempts to shrink that rate error hit an information wall
(five versions, 0.393 -> 0.204 on the acceptance metric, closed loop unmoved).

This module attacks the other end of the same chain.  The controller consumes the *reduced*
attitude -- the thrust axis `R[:, 2]` -- and that axis is directly observable from the inertial
acceleration, which the outer loop's position/velocity channel already provides:

    a_i = d(vel)/dt ,   a_i = R (T/M) e_z - g e_z^world
    =>  R[:, 2] = (a_i + g e_z^world) / |a_i + g e_z^world|          (T/M > 0)

The body-frame accelerometer cannot do this (gravity cancels in specific force and the lever-arm
terms contaminate it -- that is exactly why the estimator exists), but the *inertial* frame can,
because gravity is a known constant there.  No new sensor: velocity is already assumed by the
outer PID (`RLControl.RLShutDownControl` consumes position and velocity).

What it does NOT observe: rotation about the thrust axis (heading).  The filter therefore corrects
only the z axis and leaves the yaw exactly as the gyro integrated it, so it cannot fabricate a
heading that was never measured.

Caveats to state whenever this is used (all simulation-side here):
  * it assumes the rotor thrust is the only inertial force -- no aerodynamic drag in this plant;
  * `d(vel)/dt` is a noisy numerical derivative; a real vehicle needs a filtered velocity estimate,
    so the effective `tau` is limited by that filter's bandwidth, not by this code;
  * near free fall (`a_i ~ -g`, thrust ~ 0 while the rotors are dead) the measurement is undefined
    and is rejected -- the INS then runs open-loop, as it must.
"""
from __future__ import annotations

import numpy as np


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


class TiltObserver:
    """Causal complementary blend of the INS's z axis with the acceleration-derived thrust axis.

    `correct(R_ins, vel, dt)` returns a corrected rotation.  `tau` is the blend time constant: the
    measurement is trusted over ~`tau` seconds, so its noise is attenuated by roughly sqrt(dt/tau)
    in exchange for that much lag.  `vclip` rejects absurd accelerations (a divergent run can
    produce enormous numerical derivatives that would otherwise whip the attitude).
    """

    def __init__(self, g: float = 9.8, tau: float = 0.5, vclip: float = 200.0,
                 min_thrust: float = 0.5):
        self.g = float(g)
        self.tau = float(tau)
        self.vclip = float(vclip)
        self.min_thrust = float(min_thrust)
        self.reset()

    def reset(self):
        self.v_prev = None
        self.n_used = 0
        self.n_rejected = 0
        self.tilt_err_used = []      # |angle between the measurement and the INS's z|, radians

    def correct(self, R_ins: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        v = np.asarray(vel, float).ravel()
        if self.v_prev is None:
            self.v_prev = v.copy()
            return R_ins
        a_i = (v - self.v_prev) / dt
        self.v_prev = v.copy()
        if not np.all(np.isfinite(a_i)) or float(np.linalg.norm(a_i)) > self.vclip:
            self.n_rejected += 1
            return R_ins
        t = a_i + np.array([0.0, 0.0, self.g])
        n = float(np.linalg.norm(t))
        if n < self.min_thrust:          # thrust ~ 0 (free fall): no tilt information exists
            self.n_rejected += 1
            return R_ins
        z_meas = t / n
        R_ins = np.asarray(R_ins, float)
        z_ins = R_ins[:, 2]
        self.tilt_err_used.append(float(np.arccos(np.clip(float(z_ins @ z_meas), -1.0, 1.0))))
        self.n_used += 1
        alpha = dt / (self.tau + dt)
        z_f = (1.0 - alpha) * z_ins + alpha * z_meas
        nz = float(np.linalg.norm(z_f))
        if nz < 1e-9:
            return R_ins
        z_f = z_f / nz
        ax = np.cross(z_ins, z_f)
        s = float(np.linalg.norm(ax))
        c = float(np.clip(z_ins @ z_f, -1.0, 1.0))
        if s < 1e-12:
            return R_ins
        K = _skew(ax / s)
        th = float(np.arctan2(s, c))
        R_world = np.eye(3) + np.sin(th) * K + (1.0 - c) * (K @ K)   # rotation about ax, world frame
        return R_world @ R_ins                                       # yaw untouched by construction