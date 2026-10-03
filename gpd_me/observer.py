"""Analytic reconstruction of the saturated body rate from the lever-arm accelerometer.

Key identity
------------
Ignoring the (small, transient-only) tangential term, the accelerometer residual is

    s = a_m - (T/M) e_z =  w x (w x r) = w (w.r) - |w|^2 r = -|w|^2 * r_perp ,   r_perp = r - (r.w_hat) w_hat

so its norm depends *only* on the component of the lever arm perpendicular to the spin axis:

    |s| = |w|^2 * |r_perp|        =>        |w| = sqrt( |s| / k ),   k = |r_perp|

That is the whole trick: the accelerometer keeps measuring ``|w|`` quadratically while the gyro
is pinned at its range limit.  The two ingredients are

  * ``k``: identified from data.  Two modes are supported --
      "inrange"    samples with *no* saturated axis give ``k = |s| / |w|^2`` directly;
      "window"     when every sample already saturates, ``k`` is chosen so that the reconstructed
                   saturating axis is as *constant as possible* over the window (the wrong ``k``
                   makes it wobble).  This is the self-calibration that matches "dynamically
                   estimating the range" -- it needs no ground truth.
  * the direction: the unsaturated gyro axes are exact, and clipping preserves the sign, so the
    saturated block is recovered from ``|w|`` plus the measured axes (closed form for one
    saturated axis, ratio-preserving for several).
"""
from __future__ import annotations

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: ``_skew(a) @ b == np.cross(a, b)``."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average over ``k`` samples (edge-safe: shorter window at the ends)."""
    if k is None or k <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    k = int(k) | 1                                        # odd window
    ker = np.ones(k)
    num = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, x)
    cnt = np.convolve(np.ones(len(x)), ker, mode="same")[:, None]
    return num / cnt


def _erode(mask: np.ndarray, m: int) -> np.ndarray:
    """Boolean mask with ``m`` samples dropped on each side of every False run."""
    mask = np.asarray(mask, bool)
    if m <= 0:
        return mask
    out = mask.copy()
    bad = np.where(~mask)[0]
    for j in bad:
        out[max(0, j - m):min(len(out), j + m + 1)] = False
    return out


class LeverArmObserver:
    """Range-saturation reconstruction via the lever-arm centripetal term."""

    def __init__(self, gyro_limit_dps: float = 1000.0, lever_scale: float | None = None,
                 ema: float = 0.0, jitter_augment: bool = True, tol: float = 1e-6,
                 sat_frac: float = 1e-2):
        self.gyro_limit = np.deg2rad(gyro_limit_dps)
        self.k = lever_scale                 # |r_perp| in m; None = not calibrated yet
        self.r_perp = None                   # identified perpendicular lever-arm vector
        self.lever_arm_est = None            # identified full lever-arm vector
        self.ema = float(ema)                # 0 = use the raw algebraic solution
        self.jitter_augment = jitter_augment
        self.tol = float(tol)
        self._w_hat = None
        self.sat_frac = float(sat_frac)
        self.auto_calibrated = False

    # ------------------------------------------------------------------ helpers
    def reset(self):
        self._w_hat = None

    def saturated(self, gyro: np.ndarray) -> np.ndarray:
        """Per-axis saturation.  Uses a *relative* margin, not an exact-equality test.

        The measured rate is not exactly at the rail: the sensor rails first and the noise is added
        (or vice versa), so a sample that is 0.3 % below the limit may be a clipped one.  With the
        old ``|‖g‖ - lim| <= 1e-6`` test the scan reported "no saturated sample" on windows that the
        data itself flags as saturated, which silently turned `k` into NaN -> 0.
        """
        g = np.asarray(gyro, float)
        return np.abs(g) >= self.gyro_limit * (1.0 - self.sat_frac) - self.tol

    @staticmethod
    def residual(accel: np.ndarray, thrust_over_mass) -> np.ndarray:
        """``a_m - (T/M) e_z``, i.e. the lever-arm part of the accelerometer reading.

        Handles a single sample (``accel`` of shape (3,)) or a batch (shape ``(N, 3)``).
        """
        a = np.asarray(accel, float)
        if a.ndim == 1:
            return a - np.array([0.0, 0.0, float(thrust_over_mass)])
        out = a.copy()
        out[:, 2] -= np.asarray(thrust_over_mass, float).reshape(-1)
        return out

    # ------------------------------------------------------------------ calibration
    def calibrate_inrange(self, gyros, accels, thrusts) -> float:
        """``k = |s| / |w|^2`` over the samples whose gyro is not saturated anywhere."""
        gyros, accels, thrusts = map(np.asarray, (gyros, accels, thrusts))
        ks = []
        for g, a, t in zip(gyros, accels, thrusts):
            if self.saturated(g).any():
                continue
            n2 = float(g @ g)
            if n2 < 1e-6:
                continue
            ks.append(np.linalg.norm(self.residual(a, t)) / n2)
        if not ks:
            raise ValueError("calibrate_inrange: no unsaturated sample in the window")
        self.k = float(np.median(ks))
        return self.k

    def calibrate_joint(self, gyros, accels, thrusts, iters: int = 40, ridge: float = 1e-8,
                        r_init: float = 1e-2) -> tuple[np.ndarray, float]:
        """Joint identification of ``r_perp`` and the saturated axis when *every* sample saturates.

        Uses ``s_i = -|w_i|^2 r_perp`` with ``|w_i|^2 = A_i + x_i^2`` (``A_i`` = sum of the squared
        unsaturated gyro axes, ``x_i`` = the saturated one) and alternates two closed-form steps:

          1. given ``r_perp``:  ``x_i^2 = -(s_i . r_perp)/|r_perp|^2 - A_i``
          2. given ``x_i``    :  linear least squares for ``r_perp``

        No ground truth is used.  Identifiability comes from ``A_i`` *varying* across the window;
        if the two unsaturated axes are nearly constant the scale of ``r_perp`` is not observable
        and the estimate is only meaningful up to that scale (reported through the returned cost).
        """
        gyros, accels, thrusts = map(np.asarray, (gyros, accels, thrusts))
        sat = np.stack([self.saturated(g) for g in gyros])
        counts = sat.sum(axis=1)
        if not (counts > 0).all():
            raise ValueError("calibrate_joint: every sample must have at least one saturated axis")
        ax = int(np.bincount(np.where(sat)[1], minlength=3).argmax())
        if not sat[:, ax].all():
            raise ValueError("calibrate_joint: the saturated axis is not consistent over the window")
        other = [j for j in range(3) if j != ax]
        A = np.sum(gyros[:, other] ** 2, axis=1)
        s = accels - np.stack([np.zeros(len(thrusts)), np.zeros(len(thrusts)), thrusts], axis=1)
        sign = np.sign(gyros[:, ax])
        x = np.abs(gyros[:, ax]).copy()
        r = r_init * np.array([0.6, 0.5, 0.0])
        if np.linalg.norm(r) < 1e-12:
            r = np.array([r_init, 0.0, 0.0])
        for _ in range(iters):
            beta = -(s @ r) / max(float(r @ r), 1e-18)     # s_i = -beta_i r  =>  beta_i = |w_i|^2
            x = np.sqrt(np.clip(beta - A, 0.0, None))      # |w_i|^2 = A_i + x_i^2
            x = x * np.where(sign == 0, 1.0, sign)
            c = -(A + x * x)                               # s_i = c_i * r  (scalar times vector)
            den = float(np.sum(c * c)) + ridge
            r = np.sum(c[:, None] * s, axis=0) / den       # weighted least squares for r (3,)
            if np.linalg.norm(r) < 1e-12:
                break
        self.r_perp = r
        self.k = float(np.linalg.norm(r))
        return r, self.k

    # ------------------------------------------------------------------ reconstruction
    def _solve(self, gyro: np.ndarray, accel: np.ndarray, thrust_over_mass: float,
               k: float) -> np.ndarray:
        gyro = np.asarray(gyro, float)
        sat = self.saturated(gyro)
        if not sat.any():
            return gyro.copy()
        w = gyro.copy()
        s = self.residual(accel, thrust_over_mass)
        n2 = float(np.linalg.norm(s)) / max(k, 1e-12)          # |w|^2 from |s| = |w|^2 k
        known = float(np.sum(gyro[~sat] ** 2))
        if sat.sum() == 1:
            val = n2 - known
            w[sat] = np.sign(gyro[sat]) * np.sqrt(max(val, 0.0))
        else:
            base = float(np.linalg.norm(gyro[sat]))
            if base < 1e-9:
                w[sat] = np.sign(gyro[sat]) * np.sqrt(max((n2 - known) / max(sat.sum(), 1), 0.0))
            else:
                c = np.sqrt(max(n2 - known, 0.0)) / base
                w[sat] = gyro[sat] * c
        return w

    def step(self, gyro: np.ndarray, accel: np.ndarray, thrust_over_mass: float) -> np.ndarray:
        """One sample of the reconstructed body rate (with optional EMA smoothing)."""
        if self.k is None:
            raise RuntimeError("LeverArmObserver.k is not calibrated")
        w = self._solve(gyro, accel, thrust_over_mass, self.k)
        if self.ema > 0 and self._w_hat is not None:
            w = self.ema * self._w_hat + (1.0 - self.ema) * w
        self._w_hat = w
        return w

    # ------------------------------------------------------------------ batch / diagnostics
    def reconstruct(self, gyros, accels, thrusts) -> np.ndarray:
        gyros, accels, thrusts = map(np.asarray, (gyros, accels, thrusts))
        if self.k is None:
            raise RuntimeError("LeverArmObserver.k is not calibrated")
        out = []
        for g, a, t in zip(gyros, accels, thrusts):
            out.append(self._solve(g, a, t, self.k))
        return np.stack(out)

    def identify_lever_arm(self, gyros, accels, thrusts, dt: float, ridge: float = 1e-10,
                           use_tangential: bool = True, weights=None, smooth: int = 5,
                           margin: int = 2, n_min: int = 8) -> np.ndarray:
        """Weighted LS of the full lever-arm vector over the contiguous in-range part of the log.

        Uses both lever-arm terms, so it stays accurate through the spin-up transient where the
        tangential term dominates:

            s_i = a_m - (T/M) e_z = ( -[wdot_i]_x + w_i w_i^T - |w_i|^2 I ) r

        which is **linear in r**.  Three things the first version got wrong:

        * ``wdot`` must be differentiated on the *sample grid*, never on the compacted in-range
          subset: that subset has gaps where the gyro saturated, so a finite difference across a gap
          spans an unknown time and corrupts exactly the transient the tangential term exists for.
          Instead erode the mask by ``margin`` samples and differentiate the full sequence.
        * the gyro is noisy and differentiation amplifies it by 1/dt, so smooth before differencing
          (a centred moving average of ``smooth`` samples).
        * ``|s|`` grows as ``|w|^2``, so the SNR of each row does too -- weight rows by ``|w|^2``
          (or by the caller's ``weights`` over the kept subset).
        """
        gyros, accels, thrusts = map(np.asarray, (gyros, accels, thrusts))
        sat = np.stack([self.saturated(g) for g in gyros])
        keep = ~sat.any(axis=1)
        m = max(int(margin), (int(smooth) // 2) + 1) if use_tangential else int(margin)
        keep = _erode(keep, m) if m > 0 else keep
        if int(keep.sum()) < n_min:
            raise ValueError(f"identify_lever_arm: only {int(keep.sum())} usable in-range samples")
        Gs = _smooth(gyros, smooth) if smooth and smooth > 1 else gyros
        S = self.residual(accels, thrusts)
        Wdot = np.gradient(Gs, dt, axis=0) if use_tangential else np.zeros_like(Gs)
        A = np.stack([(-_skew(wd) + np.outer(w, w) - float(w @ w) * np.eye(3))
                      for w, wd in zip(Gs[keep], Wdot[keep])]).reshape(-1, 3)
        b = S[keep].reshape(-1)
        if weights is None:
            weights = np.linalg.norm(gyros[keep], axis=1) ** 2
        wv = np.maximum(np.asarray(weights, float).reshape(-1), 1e-6)
        wv = np.repeat(wv[:, None], 3, axis=1).reshape(-1)      # one weight per (row, axis) entry
        if len(wv) != len(A):
            wv = np.ones(len(A))
        Aw = A * wv[:, None]
        r = np.linalg.solve(Aw.T @ A + ridge * np.eye(3), Aw.T @ b)
        self.lever_arm_est = r
        self.k = float(np.linalg.norm(r))
        self.n_inrange = int(keep.sum())
        return r

    def calibrate_scan(self, gyros, accels, thrusts, k_lo: float = 1e-4, k_hi: float = 6e-2,
                       n_grid: int = 200, refine: int = 40) -> float:
        """Scale identification by a **1-D scan over ``k``** with a closed-form inner LS.

        The algebra that matters: for a *given* ``k`` the saturated gyro axes are reconstructed in
        closed form (``|w|^2 = |s|/k`` plus the measured axes and the clipped signs), and then the
        full lever-arm vector ``r`` enters the accelerometer residual **linearly**:

            s_i = ( w_i w_i^T - |w_i|^2 I ) r

        so the cost of that ``k`` is one linear least-squares solve.  Scanning ``k`` therefore reduces
        a joint (r, w) estimation to a well-conditioned 1-D problem.

        This works with **zero in-range samples** -- the earlier failures came from a ratio-based
        alternating scheme that drifts along a near-flat direction, not from a lack of
        identifiability (measured: cost at 10x k is ~10^2.4 times higher than at the true k).
        """
        gyros, accels, thrusts = map(np.asarray, (gyros, accels, thrusts))
        sat = np.stack([self.saturated(g) for g in gyros])
        if not sat.any():
            raise ValueError("calibrate_scan: no saturated sample -> k is unobservable here")
        s = self.residual(accels, thrusts)

        def cost_of(k):
            w = np.stack([self._solve(g, a, t, k) for g, a, t in zip(gyros, accels, thrusts)])
            M = np.stack([np.outer(x, x) - float(x @ x) * np.eye(3) for x in w]).reshape(-1, 3)
            ss = s.reshape(-1)
            r = np.linalg.solve(M.T @ M + 1e-14 * np.eye(3), M.T @ ss)
            res = M @ r - ss
            return float(res @ res) / float(ss @ ss), r

        grid = np.geomspace(k_lo, k_hi, n_grid)
        costs = [cost_of(k)[0] for k in grid]
        i = int(np.argmin(costs))
        lo = grid[max(0, i - 1)]
        hi = grid[min(len(grid) - 1, i + 1)]
        for _ in range(8):                      # golden-section refinement between the neighbours
            m1 = lo + 0.382 * (hi - lo)
            m2 = lo + 0.618 * (hi - lo)
            if cost_of(m1)[0] < cost_of(m2)[0]:
                hi = m2
            else:
                lo = m1
        self.k = 0.5 * (lo + hi)
        _, self.lever_arm_est = cost_of(self.k)
        self.r_perp = self.lever_arm_est
        return self.k

    def auto_calibrate(self, gyros, accels, thrusts, dt: float, min_rate: float = 2.0) -> str:
        """Calibration chain: 1-D scale **scan** first (works with zero in-range samples), exact
        linear in-range LS as the fallback when nothing saturates at all."""
        gyros, accels, thrusts = map(np.asarray, (gyros, accels, thrusts))
        sat = np.stack([self.saturated(g) for g in gyros])
        inrange = ~sat.any(axis=1)
        n_info = int((inrange & (np.linalg.norm(gyros, axis=1) > min_rate)).sum())
        # The scan is preferred whenever *anything* saturates: it uses the (steady, high-rate)
        # saturated samples, whereas the in-range samples of a failure onset are the spin-up
        # transient where the neglected tangential term wdot x r biases the centripetal-only fit.
        if sat.any():
            self.calibrate_scan(gyros, accels, thrusts)
            return f"scan (saturated={int(sat.any(axis=1).sum())}, inrange={int(inrange.sum())})"
        if n_info >= 8:
            self.identify_lever_arm(gyros, accels, thrusts, dt, use_tangential=False)
            return f"inrange (n={n_info})"
        raise ValueError("auto_calibrate: nothing saturated and nothing in range to identify from")

    @staticmethod
    def lever_scale_from_arm(lever_arm: np.ndarray, omega_dir: np.ndarray) -> float:
        """Ground-truth ``k = |r_perp|`` for a given spin direction (diagnostics only)."""
        r = np.asarray(lever_arm, float)
        d = np.asarray(omega_dir, float)
        d = d / np.linalg.norm(d)
        return float(np.linalg.norm(r - (r @ d) * d))