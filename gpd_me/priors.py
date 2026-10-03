"""Identification of every prior the estimator needs — from flight data only, no bench calibration.

The physics/torque losses need exactly four identified quantities plus the instrument range:

    g_T = c_T / M     thrust per unit command, per unit mass   (hover equilibrium: sum(u_alive) = M g / c_T)
    G   = K_M c_T / c yaw command -> yaw rate DC gain           (in-range ARX on the yaw command)
    T   = J_z / c     yaw channel time constant                (same ARX, dominant pole)
    r   (3)           lever-arm vector                         (in-range accelerometer LS, else scan)

`G` and `T` are the *only* combinations that enter the yaw torque relation
`T*d(w_z)/dt + w_z = G * sum(+-u)`, so no absolute `K_M` or `J_z` is required — which is what makes
the whole prior set obtainable from a normal flight plus the post-failure transient.
"""
from __future__ import annotations

import numpy as np

from gpd_me.observer import LeverArmObserver

YAW_SIGN = np.array([+1.0, -1.0, +1.0, -1.0])       # MetaBaseAviary4: tau_z = KM*(T0-T1+T2-T3)


def _yaw_command(u_cmd: np.ndarray, mask=None) -> np.ndarray:
    """Torque-proportional command `sum(+-u)` over the alive rotors (mask may be per-sample)."""
    x = YAW_SIGN[None, :] * u_cmd
    if mask is not None:
        mk = np.asarray(mask, float)
        x = x * (mk[None, :] if mk.ndim == 1 else mk[:len(x)])
    return x.sum(axis=1)


def id_thrust_gain(u_cmd, mask, vel=None, acc=None, grav=9.81, dt=None, skip_s=0.3):
    """g_T = c_T/M from the quasi-hover segment (no net acceleration).

    `mask` may be a single 4-vector or a per-sample (N,4) array (the mask changes mid-episode when
    a healthy identification phase is followed by the fault injection).
    """
    mk = np.asarray(mask, float)
    if mk.ndim == 1:
        su = (u_cmd * mk[None, :]).sum(axis=1)
    else:
        su = (u_cmd * mk[:len(u_cmd)]).sum(axis=1)
    settled = np.ones(len(su), bool)
    if dt is not None:                                  # ignore the actuator-lag transient
        settled[:int(skip_s / dt)] = False
    if vel is not None and len(vel) > 8:
        dv = np.linalg.norm(np.gradient(vel, axis=0), axis=1)
        hover = (dv < np.percentile(dv, 30)) & settled
    elif acc is not None:
        hover = np.abs(np.linalg.norm(acc, axis=1) - grav) < 1.0
    else:
        hover = np.ones(len(su), bool)
    sel = hover & (su > 1.0)
    if sel.sum() < 10:
        sel = settled & (su > np.percentile(su, 60))
    if sel.sum() < 5:
        return np.nan
    return float(grav / np.mean(su[sel]))


def id_yaw_channel(u_cmd, omega_z, in_range, dt, order=2, mask=None):
    """ARX(order,order) on (sum(+-u) -> w_z) restricted to in-range samples.

    Returns (G, T) where `T*dwz/dt + wz = G*sum(+-u)` is the identified first-order yaw model.
    """
    x = _yaw_command(u_cmd, mask)
    ok = np.asarray(in_range, bool)
    ok[:-1] &= ok[1:]
    if order == 2:
        ok[:-1] &= ok[1:]
        ok = ok & np.concatenate([ok[1:], [False]]) & np.concatenate([ok[2:], [False, False]])
        k = np.where(ok)[0]
        k = k[(k >= 2) & (k < len(x))]
        if len(k) < 50:
            return np.nan, np.nan
        A = np.stack([omega_z[k - 1], omega_z[k - 2], x[k - 1], x[k - 2], np.ones_like(k, float)], 1)
        b = omega_z[k]
    else:
        k = np.where(ok[:-1])[0]
        if len(k) < 50:
            return np.nan, np.nan
        A = np.stack([omega_z[k], x[k], np.ones_like(k, float)], 1)
        b = omega_z[k + 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    if order == 2:
        a1, a2, b1, b2, c = sol
        den = 1.0 - a1 - a2
        if abs(den) < 1e-9:
            return np.nan, np.nan
        G = float((b1 + b2) / den)
        poles = np.roots([1.0, -a1, -a2])
        mag = np.abs(poles)
        p = poles[int(np.argmax(mag))]
        T = float(-dt / np.log(max(abs(p), 1e-9))) if abs(p) < 0.9999 else np.inf
    else:
        a, bb, c = sol
        G = float(bb / max(1 - a, 1e-9))
        T = float(-dt / np.log(max(abs(a), 1e-9))) if abs(a) < 0.9999 else np.inf
    return G, T


def id_lever_arm(gyro, accel, tom, dt, lim, calib=400):
    """r from in-range samples (linear LS) else the 1-D scan; returns (r, k, how)."""
    ob = LeverArmObserver(gyro_limit_dps=np.rad2deg(lim))
    thr = np.ones(len(gyro)) * tom                     # residual only needs (T/M) e_z
    inr = ~(np.abs(gyro) >= lim - 1e-9).any(axis=1)
    n_info = int((inr & (np.linalg.norm(gyro, axis=1) > 2.0)).sum())
    if n_info >= 8:
        try:
            r = ob.identify_lever_arm(gyro[inr], accel[inr], thr[inr], dt, use_tangential=False)
            return r, float(np.linalg.norm(r)), f"inrange(n={n_info})"
        except Exception:
            pass
    try:
        r, k = ob.calibrate_joint(gyro[:calib], accel[:calib], thr[:calib])
        return np.asarray(r, float), float(k), "scan-joint"
    except Exception:
        try:
            k = ob.calibrate_scan(gyro[:calib], accel[:calib], thr[:calib])
            return np.asarray(ob.lever_arm_est, float), float(k), "scan"
        except Exception:
            return np.zeros(3), np.nan, "fail"


def identify_priors(gyro, accel, u_cmd, omega_true, mask, dps, dt, vel=None, calib=400,
                    tom=None):
    """Full prior vector: [r(3), g_T, G, T, range] plus diagnostics.

    `mask` may be a 4-vector or a per-sample (N,4) array.  Each identifier selects its own samples:
    `r` wants in-range ones, `g_T` wants the trimmed (low-acceleration) part of the healthy phase,
    `G`/`T` want in-range pairs of the yaw channel.
    """
    mk = np.asarray(mask, float)
    n = min([len(np.asarray(a)) for a in (gyro, accel, u_cmd, omega_true)]
            + ([len(np.asarray(vel))] if vel is not None else [])
            + ([len(mk)] if mk.ndim == 2 else []))
    gyro = np.asarray(gyro)[:n]; accel = np.asarray(accel)[:n]
    u_cmd = np.asarray(u_cmd)[:n]; omega_true = np.asarray(omega_true)[:n]
    vel = None if vel is None else np.asarray(vel)[:n]
    mask_step = mk[:n] if mk.ndim == 2 else mk
    lim = np.deg2rad(dps)
    inr = ~((np.abs(gyro) >= lim - 1e-9).any(axis=1))
    thr = np.asarray(tom, float)[:n] if tom is not None else np.zeros(n)   # T/M: the accelerometer's
    r, k, how = id_lever_arm(gyro, accel, thr, dt, lim, min(calib, n))     # DC along body z must go
    g_T = id_thrust_gain(u_cmd, mask_step, vel=vel, dt=dt)
    G, T = id_yaw_channel(u_cmd, omega_true[:, 2], inr, dt, mask=mask_step)
    prior = np.array([r[0], r[1], r[2], g_T, G, T, lim], float)
    diag = dict(k=k, how=how, inr_frac=float(inr.mean()), g_T=g_T, G=G, T=T)
    return np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0), diag
