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

What it does NOT observe: rotation about the thrust axis (heading).  Each step's update is a
rotation about an axis perpendicular to z, so it injects no rotation *about* the thrust axis in that
step.  Do NOT read that as "the heading is preserved": (a) Euler yaw still moves whenever roll is
nonzero; (b) the update is parallel transport on the sphere, which is not holonomy-free -- carrying
the frame around a closed loop rotates it by the enclosed solid angle; (c) the controller is not
yaw-insensitive anyway, because a heading error mis-splits the commanded tilt between roll and pitch
in body coordinates, and with two rotors dead the attainable torque set is body-fixed and strongly
anisotropic.  The heading error must therefore be measured (`scripts/tilt_check.py` reports it), not
assumed harmless.

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

    def __init__(self, g: float = 9.8, tau: float = 0.05, vclip: float = 200.0,
                 min_thrust: float = 0.05, thrust_c: float = 1.0, align_s: float | None = None):
        self.g = float(g)
        self.tau = float(tau)
        self.vclip = float(vclip)
        self.min_thrust = float(min_thrust)   # hard floor only; the weight below does the rest
        self.thrust_c = float(thrust_c)       # |t| scale over which the sample is trusted
        # Total timestamp alignment applied to the measurement (seconds).  None = dt/2, the natural
        # timestamp of a backward difference over (t-dt, t].  A controlled offset sweep
        # (scripts/tilt_check.py --offsets) shows this plant's optimum is ~2 control steps = 10 ms,
        # not 2.5 ms: the command path adds its own delay.  It is a pure timing constant, identifiable
        # from logged flight data by the same sweep -- legitimate to calibrate, and NOT a free knob to
        # tune against an acceptance metric.
        self.align_s = None if align_s is None else float(align_s)
        self.reset()

    def reset(self):
        self.v_prev = None
        self.n_used = 0
        self.n_rejected = 0
        self.n_rejected_vclip = 0
        self.n_rejected_thrust = 0
        self.tilt_err_used = []      # |angle between the measurement and the INS's z|, radians
        self.z_meas = None           # last accepted measurement (world-frame thrust axis)

    def correct(self, R_ins: np.ndarray, vel: np.ndarray, dt: float,
                omega: np.ndarray | None = None) -> np.ndarray:
        v = np.asarray(vel, float).ravel()
        if self.v_prev is None:
            self.v_prev = v.copy()
            return R_ins
        a_i = (v - self.v_prev) / dt
        self.v_prev = v.copy()
        if not np.all(np.isfinite(a_i)) or float(np.linalg.norm(a_i)) > self.vclip:
            self.n_rejected += 1     # either a divergent run or ground contact -- both unusable
            self.n_rejected_vclip += 1
            return R_ins
        t = a_i + np.array([0.0, 0.0, self.g])
        n = float(np.linalg.norm(t))
        if n < self.min_thrust:          # thrust ~ 0 (free fall): no tilt information exists
            self.n_rejected += 1
            self.n_rejected_thrust += 1
            return R_ins
        z_meas = t / n
        if omega is not None:
            # The backward difference a_i covers the interval (t-dt, t] while the attitude is at t,
            # so the raw measurement is timestamped ~half a step early -- at 43 rad/s that is ~6 deg,
            # a large share of the 19-32 deg measured error.  Rotate it forward by the half step using
            # the available rate estimate.  Over 2.5 ms even a 6.7 rad/s-wrong rate contributes <1 deg,
            # so this does not couple the observer to the drift it is correcting.
            w = np.asarray(omega, float).ravel()
            wn = float(np.linalg.norm(w))
            if wn > 1e-9:
                align = (0.5 * dt) if self.align_s is None else float(self.align_s)
                th = wn * align
                K = _skew(w / wn)
                z_meas = (np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)) @ z_meas
        self.z_meas = z_meas
        R_ins = np.asarray(R_ins, float)
        z_ins = R_ins[:, 2]
        ax = np.cross(z_ins, z_meas)
        sj = float(np.linalg.norm(ax))
        cj = float(np.clip(z_ins @ z_meas, -1.0, 1.0))
        theta = float(np.arctan2(sj, cj))
        self.tilt_err_used.append(theta)
        self.n_used += 1
        # Rotate by alpha*theta in ANGLE space, not by normalising a vector blend.  A normalised
        # linear blend moves the axis by atan2(a sin th, (1-a)+a cos th) ~ a*sin(th), so its gain
        # collapses exactly where this module has to work: 0.64 at 90 deg, 0.19 at 150, 0.06 at 170,
        # i.e. tau_eff degrades to ~8.5 s for an inverted INS (found by the opus review).
        w = n / (n + self.thrust_c)          # down-weight weak/thrust-free samples instead of gating
        step = (dt / (self.tau + dt)) * w * theta
        if sj < 1e-9:
            if cj >= 0.0:
                return R_ins                       # already aligned
            # antipodal: any perpendicular axis is a valid great circle.  The old code returned the
            # attitude uncorrected here, so a fully inverted INS was never recovered -- the state this
            # module exists to fix.
            trial = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            axis = trial[int(np.argmin(np.abs(z_ins)))]
            ax = np.cross(z_ins, axis)
            sj = float(np.linalg.norm(ax))
            if sj < 1e-12:
                return R_ins
        ax = ax / sj
        K = _skew(ax)
        R_world = np.eye(3) + np.sin(step) * K + (1.0 - np.cos(step)) * (K @ K)
        return R_world @ R_ins