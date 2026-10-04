"""Identification of every prior the estimator needs — from flight data only, no bench calibration.

The estimator needs nine numbers, and every one of them is identifiable from a normal flight plus
the post-failure transient:

    r   (3)           lever-arm vector          in-range accelerometer LS (tangential term on),
                                                else the 1-D scale scan when nothing is in range
    g_T = c_T / M     thrust per unit command   hover equilibrium: sum(u_alive) = M g / c_T
    G   = K_M c_T / c yaw command -> yaw rate    in-range ARX on the yaw command
    T   = J_z / c     yaw channel time constant same ARX, dominant pole
    range             gyro full scale           instrument setting
    tau               actuator lag              first-order rotor-speed fit to the axial accel
    k   = |r_perp|    lever scale               from the identified r and the current spin axis

`G` and `T` are the *only* combinations that enter the yaw torque relation
`T*d(w_z)/dt + w_z = G * sum(+-u)`, so no absolute `K_M` or `J_z` is required — which is what makes
the prior set obtainable without a bench.

Two consequences of the review that are now enforced here:

* **The residual is never built from simulated truth.**  ``tom`` (the true specific thrust) is not an
  input of any identifier; the thrust used for the accelerometer residual comes from the *command*
  through the identified `g_T` and the identified actuator lag (``tom_from_command``).
* **`k` is `|r_perp|`, not `|r|`.**  The component of `r` along the spin axis is unobservable
  (`s = (ww^T - |w|^2 I) r` annihilates it) and including it inflates the scale, which then floors
  the reconstructed saturated axis at zero -- worse than the clip it is supposed to beat.
"""
from __future__ import annotations

import numpy as np

from gpd_me.observer import LeverArmObserver

YAW_SIGN = np.array([+1.0, -1.0, +1.0, -1.0])       # MetaBaseAviary4: tau_z = KM*(T0-T1+T2-T3)
TAU_DEFAULT = 0.026                                 # URDF nominal; only used if tau is unidentifiable
MIN_RATE = 0.8                                      # rad/s: below this the lever signal is in the noise
#                                                     (was 2.0, which rejected every sample at the
#                                      0.3*range=0.5-1.6 rad/s identification manoeuvre of 100-300 dps)


def _yaw_command(u_cmd: np.ndarray, mask=None) -> np.ndarray:
    """Torque-proportional command `sum(+-u)` over the alive rotors (mask may be per-sample)."""
    x = YAW_SIGN[None, :] * u_cmd
    if mask is not None:
        mk = np.asarray(mask, float)
        x = x * (mk[None, :] if mk.ndim == 1 else mk[:len(x)])
    return x.sum(axis=1)


def _first_mask_change(mask_t) -> int | None:
    """Sample index of the fault injection, or None when the alive set never changes."""
    if mask_t is None:
        return None
    mk = np.asarray(mask_t, float)
    if mk.ndim == 1:
        return None
    changed = np.abs(mk - mk[0][None, :]).sum(axis=1) > 1e-9
    if not changed.any():
        return None
    return int(np.argmax(changed))


# --------------------------------------------------------------- actuator model (identified, never truth)
def rotor_speed_lag(u_cmd: np.ndarray, mask_t: np.ndarray, dt: float, tau: float,
                    n0: np.ndarray | None = None, input_delay: int = 1) -> np.ndarray:
    """Rotor speed from the command through the plant's own discrete actuator model.

    The aircraft's dynamics update the rotor *speed* ``n`` with a first-order lag and take
    ``T = n^2``::

        n_new = n_target + (n_old - n_target) * exp(-dt / tau)

    which is exactly what the estimator is allowed to assume (with `tau` identified from data).
    Starts from rest (matching a reset motor state) unless `n0` is given.  Returns ``(N,4)`` speeds.

    ``input_delay`` is the number of control samples between *issuing* a command and the plant using
    it.  It is one here and that is not cosmetic: the environment latches the action inside `step()`
    and applies it to the following physics step, so modelling the same sample gives an error of
    0.1-1.75 m/s^2 on the specific thrust (measured), while modelling the one-sample-delayed command
    reproduces the plant's true thrust to 0.001-0.006 m/s^2.  A known, deterministic timing
    convention -- the estimator knows the commands it sent -- but it must be modelled.
    """
    u = np.asarray(u_cmd, float)
    if input_delay and len(u) > input_delay:
        u = np.concatenate([np.repeat(u[:1], input_delay, axis=0), u[:-input_delay]], axis=0)
    mk = np.asarray(mask_t, float)
    if mk.ndim == 1:
        mk = np.tile(mk[None, :], (len(u), 1))
    n_t = np.sqrt(np.maximum(u * mk[:len(u)], 0.0))          # target speed (units of sqrt(N))
    a = float(np.exp(-dt / max(float(tau), 1e-6)))
    out = np.empty_like(n_t)
    n = np.zeros(u.shape[1]) if n0 is None else np.asarray(n0, float).copy()
    for i in range(len(n_t)):
        n = n_t[i] + (n - n_t[i]) * a
        out[i] = n
    return out


def tom_from_command(u_cmd: np.ndarray, mask_t: np.ndarray, dt: float, g_T: float,
                     tau: float) -> np.ndarray:
    """``T/M`` as the estimator may compute it: command -> rotor speed -> squared -> per unit mass."""
    if not np.isfinite(g_T) or g_T <= 0:
        return np.full(len(np.asarray(u_cmd)), np.nan)
    n = rotor_speed_lag(u_cmd, mask_t, dt, tau)
    return float(g_T) * (n ** 2).sum(axis=1)


def id_thrust_lag(accel: np.ndarray, u_cmd: np.ndarray, mask_t: np.ndarray, dt: float,
                  g_T: float, skip_s: float = 0.15, tau_lo: float = 2e-3,
                  tau_hi: float = 0.15, n_grid: int = 48) -> float:
    """`tau` from the *nominal* segment, where the accelerometer's axial channel is clean.

    For a spin about body z the lever-arm residual ``s = w(w.r) - |w|^2 r`` has **identically zero**
    z-component (the ``|w|^2 r_z`` terms cancel), so during the hover + yaw-identification phase the
    axial accelerometer measures ``T/M`` directly.  Fitting the command response there therefore
    identifies the actuator lag without any truth.

    For every candidate `tau` the *optimal amplitude* is solved in closed form, i.e. only the shape
    of the response is matched.  That matters: with a fixed amplitude a `g_T` error is compensated by
    the optimiser by shrinking `tau` (measured: a 15 % low `g_T` pulled `tau_hat` to -89 % of the
    truth), because a faster lag means a larger high-frequency gain.
    """
    accel = np.asarray(accel, float)
    u_cmd = np.asarray(u_cmd, float)
    if not np.isfinite(g_T) or g_T <= 0:
        return float("nan")
    t_f = _first_mask_change(mask_t)
    end = len(u_cmd) if t_f is None else t_f
    i0 = int(skip_s / max(dt, 1e-6))
    if end - i0 < 20:
        return float("nan")
    a_z = accel[i0:end, 2]
    a_c = a_z - a_z.mean()

    def cost(tau):
        # simulate from t=0 so the model carries the plant's own history (starting the integrator at
        # i0 manufactures a spurious ramp-up whose length is ~tau, which biases the optimum towards
        # the smallest tau on the grid), then score only the settled window
        p = tom_from_command(u_cmd[:end], np.asarray(mask_t)[:end], dt, 1.0, tau)
        p = p[i0:end]
        if not np.isfinite(p).all() or np.ptp(p) < 1e-9:
            return 1e9
        p = p - p.mean()
        sc = float((p @ a_c) / (p @ p))                     # closed-form amplitude (absorbs g_T)
        return float(np.sqrt(np.mean((sc * p - a_c) ** 2)))

    grid = np.geomspace(tau_lo, tau_hi, n_grid)
    costs = np.array([cost(t) for t in grid])
    i = int(np.argmin(costs))
    lo = grid[max(0, i - 1)]
    hi = grid[min(len(grid) - 1, i + 1)]
    for _ in range(12):                                   # golden-section refinement
        m1 = lo + 0.382 * (hi - lo)
        m2 = lo + 0.618 * (hi - lo)
        if cost(m1) < cost(m2):
            hi = m2
        else:
            lo = m1
    return float(0.5 * (lo + hi))


# ------------------------------------------------------------------------------------ hover trim
def id_thrust_gain(u_cmd, mask, vel=None, acc=None, grav=9.81, dt=None, skip_s=0.3, diag=None):
    """g_T = c_T/M from the quasi-hover segment (no net acceleration).

    `mask` may be a single 4-vector or a per-sample (N,4) array.  When the alive set changes the
    identifier restricts itself to the **nominal segment before the fault**: that phase is a hover by
    construction (a differential yaw command cancels in the sum), so the estimate needs no velocity
    channel and carries no selection bias -- which is what makes it reproducible offline from the log
    alone.  The velocity/acceleration heuristics below are only a fallback for logs without a fault.
    """
    mk = np.asarray(mask, float)
    if mk.ndim == 1:
        su = (u_cmd * mk[None, :]).sum(axis=1)
    else:
        su = (u_cmd * mk[:len(u_cmd)]).sum(axis=1)
    settled = np.ones(len(su), bool)
    branch = "fallback"
    if dt is not None:                                  # ignore the actuator-lag transient
        settled[:int(skip_s / dt)] = False
    t_f = _first_mask_change(mask)
    if t_f is not None:                                 # nominal (healthy) hover segment
        nom = np.zeros(len(su), bool)
        nom[:t_f] = True
        sel = nom & settled & (su > 1.0)
        branch = "nominal"
        if acc is not None:
            # hover trim means *zero net acceleration*, and in the nominal phase the axial channel
            # measures it: a_z = T/M, so |a_z - g| < band is literally the trim condition.  Without
            # this the recovery transient after the initial free-fall (commanded thrust > M g while
            # the drone climbs back to the setpoint) biased the mean command high and g_T low.
            az = np.asarray(acc, float)[:len(su), 2]
            for band in (0.3, 1.0):
                sel2 = sel & (np.abs(az - grav) < band)
                if sel2.sum() >= 10:
                    sel, branch = sel2, f"nominal|az-g|<{band}"
                    break
            else:
                # no trim samples at all: use the *definition* per sample instead -- the axial
                # accelerometer measures T/M, so once the actuator lag has settled g_T = a_z / sum(u).
                # (Aimed at episodes where the fault lands before the drone recovers from the initial
                # free-fall; widening the trim band instead admitted un-trimmed samples and came out
                # 15-30 % low on ~3 % of episodes.)
                ok = settled[:len(su)] & (su > 1.0)
                if ok.sum() >= 10:
                    ratio = float(np.median(az[ok] / su[ok]))
                    if diag is not None:
                        diag["g_T_branch"], diag["n_gT"] = "nominal|ratio", int(ok.sum())
                    return ratio
    elif vel is not None and len(vel) > 8:
        dv = np.linalg.norm(np.gradient(vel, axis=0), axis=1)
        hover = (dv < np.percentile(dv, 30)) & settled
        sel = hover & (su > 1.0)
        branch = "vel"
    elif acc is not None:
        hover = np.abs(np.linalg.norm(acc, axis=1) - grav) < 1.0
        sel = hover & settled & (su > 1.0)
        branch = "acc"
    else:
        sel = settled & (su > 1.0)
        branch = "all"
    if sel.sum() < 10:
        sel = settled & (su > np.percentile(su, 60))
        branch += "+p60"                                # biased high: last resort only
    if sel.sum() < 5:
        return np.nan
    if diag is not None:
        diag["g_T_branch"] = branch
        diag["n_gT"] = int(sel.sum())
    return float(grav / np.mean(su[sel]))


def id_yaw_channel(u_cmd, omega_z, in_range, dt, order=2, mask=None):
    """ARX(order,order) on (sum(+-u) -> w_z) restricted to in-range samples.

    Returns (G, T) where `T*dwz/dt + wz = G*sum(+-u)` is the identified first-order yaw model.
    Both are NaN when the excitation is too poor to identify them; the caller must then *skip* the
    torque term rather than treat a zeroed prior as a constraint.
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


# ------------------------------------------------------------------------------------ lever arm
def _dominant_dir_clip(gyro):
    """Mean spin direction over a window from the *clipped* gyro (approximate: a pinned axis is
    under-weighted, so this is only used when the reconstruction cannot be trusted)."""
    g = np.asarray(gyro, float)
    if not len(g):
        return None
    v = g.mean(axis=0)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else None


def _dominant_dir(ob: LeverArmObserver, gyro, accel, tom, k: float):
    """Mean spin direction (body frame) over a saturated window, from the reconstruction itself."""
    try:
        ob.k = float(k)
        w = ob.reconstruct(gyro, accel, tom)
        wm = np.mean(w, axis=0)
        nrm = float(np.linalg.norm(wm))
        if nrm < 1e-9:
            return None
        return wm / nrm
    except Exception:
        return None


def id_lever_arm(gyro, accel, tom, dt, lim, mask_t=None, post_win=400, pre_win=25):
    """`r` from in-range samples (tangential term ON) else the 1-D scan; returns (r, k, how).

    The scan runs on a window *around the fault onset* (``mask_t`` tells us where it is) instead of
    the first ``calib`` samples -- the fault time is uniform in [0.12, 0.6] of the episode, so
    scanning the head of the log saw a saturated sample only ~17 % of the time and returned
    ``k = NaN -> 0`` for the rest.

    Returns ``k = |r_perp|`` w.r.t. the dominant spin direction -- the quantity the invariant
    ``|s| = |w|^2 k`` actually needs -- and NaN when nothing could be identified (the caller then
    treats the scale as *unknown* rather than zero).
    """
    gyro = np.asarray(gyro, float)
    ob = LeverArmObserver(gyro_limit_dps=np.rad2deg(lim))
    # same relative-margin rule as the observer itself, so the mask the scan sees and the mask the
    # scan's internal `_solve` sees cannot disagree (they did: the scan kept reporting "no saturated
    # sample" on windows the data flags as saturated, because the measured rail value carries noise)
    sat = np.abs(gyro) >= lim * (1.0 - ob.sat_frac) - ob.tol
    inr = ~sat.any(axis=1)
    t_f = _first_mask_change(mask_t)
    how = []
    n_strong = int((inr & (np.linalg.norm(gyro, axis=1) > max(2.0, 0.25 * lim))).sum())
    n_any = int(inr.sum())
    # route choice is by *information content*, not by availability.  The scan uses the steady
    # high-rate saturated samples (excellent SNR); the in-range LS uses the spin-up transient and is
    # only trustworthy when the in-range samples themselves carry a real rate.  At 100-300 dps they
    # never do (the whole range is below 2 rad/s), so a weak LS used to win over a good scan and the
    # identified arm came out noise-dominated (measured: 23 mm error vs 2.5 mm at high range).
    # The scan window starts *after* the spin-up: its model is the quasi-steady invariant
    # `s = (ww^T - |w|^2 I) r` and has no tangential term, so the first ~0.3 s after the fault (where
    # wdot x r dominates) biases it.
    t0 = (t_f if t_f is not None else 0) + (int(0.3 / dt) if t_f is not None else 0)
    seg = slice(max(0, t0 - int(pre_win)), min(len(gyro), t0 + int(post_win)))

    # ---- 1) in-range LS **first**: it also identifies the accelerometer bias, which the scan's model
    #         lacks and which biases `k` most where the lever signal is weakest (low rates)
    r_ls = None
    b_acc = np.zeros(3)
    if n_any >= 8:
        try:
            r_ls = ob.identify_lever_arm(gyro, accel, tom, dt, use_tangential=True, smooth=5,
                                         fit_bias=False)   # see observer: bias fit tried, rejected
            b_acc = np.asarray(getattr(ob, "accel_bias", np.zeros(3)), float)
            how.append(f"inrange(n={n_any},strong={n_strong},|b|={np.linalg.norm(b_acc):.2f})")
        except Exception:
            r_ls = None
    accel_c = accel - b_acc                 # bias-corrected for everything downstream

    # ---- 2) the 1-D scale scan on the post-fault window, using the bias-corrected residual
    k_scan = float("nan")
    r_scan = None
    if sat[seg].any():
        try:
            k_scan = float(ob.calibrate_scan(gyro[seg], accel_c[seg], tom[seg]))
            if not getattr(ob, "scale_identified", False):
                # collapsed scale (cost flat under a 2x change): the fitted r absorbed the scale, so
                # neither k nor r is trustworthy -> declare the scale unknown instead of shipping a
                # value that would inflate the reconstructed saturated axis by the same factor
                how.append(f"scan_unidentified(flat={getattr(ob,'k_flat',float('nan')):.2f})")
                k_scan = float("nan")
            else:
                r_scan = np.asarray(ob.lever_arm_est, float)
                how.append(f"scan(n_sat={int(sat[seg].any(axis=1).sum())},flat={ob.k_flat:.1f})")
        except Exception:
            k_scan = float("nan")

    # ---- 3) independent cross-check of the scale: the in-range samples give k = |s|/|w|^2 directly,
    #         but only when they carry a real rate (at 100-300 dps the whole range is below the floor)
    if np.isfinite(k_scan) and n_strong >= 20 and n_any >= 8:
        try:
            k_in = float(ob.calibrate_inrange(gyro, accel_c, tom))
            if np.isfinite(k_in) and k_in > 0 and not (1 / 3.0 <= k_scan / k_in <= 3.0):
                how.append(f"k_inrange_override({k_scan*1e2:.3f}->{k_in*1e2:.3f}cm)")
                k_scan = k_in
        except Exception:
            pass

    # ---- 4) route choice by *information content*, not availability: the scan uses the steady
    #         high-rate saturated samples (excellent SNR), the in-range LS the spin-up transient and is
    #         only trustworthy when those samples carry a real rate.  At 100-300 dps they never do.
    if r_ls is not None and (n_strong >= 20 or r_scan is None):
        r = r_ls
    elif r_scan is not None:
        r = r_scan
    elif r_ls is not None:
        r = r_ls
    else:
        return np.zeros(3), float("nan"), "fail"
    r = np.asarray(r, float)

    # ---- 5) the scale the front end will use
    if np.isfinite(k_scan):
        d = _dominant_dir(ob, gyro[seg], accel_c[seg], tom[seg], k_scan)
        k = float(np.linalg.norm(r - (r @ d) * d)) if d is not None else k_scan
    elif sat[seg].any():
        # The scan is *scale-degenerate* whenever the spin-axis direction is pinned by the measured
        # axes alone (one saturated axis, or several distributed by their clipped ratio): then
        # `|s_i| = |w_i|^2 |r_perp|` with `|w_i|^2 = |s_i|/k` is a tautology and any k fits equally
        # well -- measured cost(2k)/cost(k) = 1.00-1.33 for a collapse versus 2.0-2.1 when the
        # tangential term really pins it.  In that case the scale has to come from the in-range LS
        # (weak at 100-300 dps, but the *direction* it carries is the whole point); with the clipped
        # gyro as the direction estimate, which is only approximate under multi-axis saturation.
        d = _dominant_dir_clip(gyro[seg])
        k = float(np.linalg.norm(r - (r @ d) * d)) if d is not None else float("nan")
    else:
        # nothing saturated in the identification window: `k` is irrelevant (the clip is exact there)
        # and is recorded as unknown rather than fabricated
        return r, float("nan"), "+".join(how) or "nominal-only"
    if not np.isfinite(k) or k <= 0:
        k = k_scan if np.isfinite(k_scan) else float("nan")
    return r, k, "+".join(how)


def identify_priors(gyro, accel, u_cmd, omega_true, mask, dps, dt, vel=None, tom=None,
                    post_win=400):
    """Full prior vector ``[r(3), g_T, G, T, range, tau, k]`` plus diagnostics.

    `mask` may be a 4-vector or a per-sample (N,4) array; it is needed per-sample because the alive
    set changes at the fault.  `tom` (true specific thrust) is accepted for **diagnostics only** --
    it never enters an identification, so the same code runs on a real aircraft.
    """
    mk = np.asarray(mask, float)
    mask_t = mk if mk.ndim == 2 else np.tile(mk[None, :], (len(gyro), 1))
    n = min([len(np.asarray(a)) for a in (gyro, accel, u_cmd, omega_true)]
            + [len(mask_t)]
            + ([len(np.asarray(vel))] if vel is not None else []))
    gyro = np.asarray(gyro)[:n]; accel = np.asarray(accel)[:n]
    u_cmd = np.asarray(u_cmd)[:n]; omega_true = np.asarray(omega_true)[:n]
    mask_t = mask_t[:n]
    vel = None if vel is None else np.asarray(vel)[:n]
    lim = np.deg2rad(dps)
    inr = ~((np.abs(gyro) >= lim * 0.99 - 1e-9).any(axis=1))      # same margin as the observer

    diag = {}
    g_T = id_thrust_gain(u_cmd, mask_t, vel=vel, acc=accel, dt=dt, diag=diag)
    tau = id_thrust_lag(accel, u_cmd, mask_t, dt, g_T)
    if not np.isfinite(tau):
        tau = TAU_DEFAULT
    thr = tom_from_command(u_cmd, mask_t, dt, g_T, tau)        # <- command + identified lag, not truth
    r, k, how = id_lever_arm(gyro, accel, thr, dt, lim, mask_t=mask_t, post_win=post_win)
    # Fit the yaw channel on the MEASURED gyro restricted to in-range samples, not on omega_true:
    # restricting to in-range samples already removes the saturation error, using the truth on top
    # of that would silently delete the gyro noise and make 'all priors come from measurable'
    # false for G and T (found by the user's code review).
    G, T = id_yaw_channel(u_cmd, np.asarray(gyro)[:, 2], inr, dt, mask=mask_t)
    torque_ok = bool(np.isfinite(G) and np.isfinite(T) and G > 0 and T > 0 and
                     abs(T) < 1e3 and G < 1e4)
    diag.update(k=k, how=how, inr_frac=float(inr.mean()), g_T=g_T, G=G, T=T, tau=tau,
                torque_ok=torque_ok, n_tom=len(thr))
    if tom is not None:                    # how good is the command-based thrust? (diagnostic only)
        tm = np.asarray(tom, float)[:n]
        den = float(np.mean(np.abs(tm))) + 1e-9
        diag["tom_err_abs"] = float(np.mean(np.abs(np.asarray(thr) - tm)))
        diag["tom_err_rel"] = float(diag["tom_err_abs"] / den)
    prior = np.array([r[0], r[1], r[2], g_T,
                      G if np.isfinite(G) else 0.0, T if np.isfinite(T) else 0.0,
                      lim, tau,
                      k if np.isfinite(k) and k > 0 else 0.0], float)
    return np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0), diag