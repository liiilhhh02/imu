"""How much in-range information does each failure scenario actually give, and how to use it.

The lever-arm scale `k = |r_perp|` is what the accelerometer reconstruction needs.  It can only be
identified from samples where the gyro is *inside* its range (or from an absolute reference such as
the commanded yaw torque).  This script measures, for the 1- and 3-rotor failure cases:

  * how many samples with an in-range gyro exist at all (two onset models:
    ``spin0=False`` = the senior's hard-coded initial spin, ``spin0=True`` = physical onset at
    hover, the body spins up from rest),
  * how well several estimator variants recover ``k`` from those samples,
  * whether a torque-anchored estimate works when there is no in-range sample at all.

Run:  PYTHONPATH=<repo>:<me> python scripts/study_calibration.py
"""
import os
import sys
import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpd_me.observer import LeverArmObserver  # noqa: E402

sys.path.insert(0, os.path.join(ME, "scripts"))
from verify_observer import CKPT, LEVER_ARM, make_env  # noqa: E402
from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402
from verify_imu import load_policy  # noqa: E402

DPS = 1000.0


def collect(flag, spin0, steps=800, dps=DPS, target_pos=(0.0, 0.0, 1.0)):
    """Flies the case, logs (gyro, accel, thrust, omega_true, tau_z_proxy) at every control step."""
    policy = load_policy(CKPT[flag])
    env = make_env(flag, dps, "truth", spin0=spin0)
    ctrl = RLControl(DroneModel.CF2X)
    ctrl.reset()
    tp = np.array(target_pos, float)
    out = {k: [] for k in ("g", "a", "T", "w", "t")}
    for _ in range(steps):
        raw = env._getDroneStateVector(0)
        ta, tz = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                        cur_quat=raw[3:7], cur_vel=raw[10:13], target_pos=tp)
        r = np.hypot(tz[0], tz[1])
        if r > 0.26:
            s = 0.26 / r
            tz = np.array([tz[0] * s, tz[1] * s, np.sqrt(1 - (tz[0] * s) ** 2 - (tz[1] * s) ** 2)])
        env.target_a, env.target_z_body = float(ta), tz
        obs = env._computeObs()
        action = policy.select_action(obs, deterministic=True)
        env.step(action)
        out["g"].append(env.gyro_meas.copy())
        out["a"].append(env.accel_meas.copy())
        out["T"].append(np.asarray(env.thrust[0], float).copy())
        out["w"].append(env.omega_true.copy())
        out["t"].append(float(np.atleast_1d(env.last_acc)[0]))
    km, jzz, dt = float(env.KM), 0.007, env.TIMESTEP * env.AGGR_PHY_STEPS
    env.close()
    return {k: np.array(v) for k, v in out.items()}, km, jzz, dt


def in_range_stats(g, lim):
    """Range availability: all-axes-in-range and per-axis counts."""
    sat = np.abs(g) >= lim - 1e-9
    return int((~sat.any(axis=1)).sum()), int((~sat.all(axis=1)).sum()), sat.sum(axis=0)


def weighted_identify(k_o, g, a, t, dt, idx, weights, use_tangential=False):
    """Least-squares r from the selected samples, with per-sample row weights."""
    if len(idx) < 3:
        return None
    G, S = g[idx], k_o.residual(a[idx], t[idx])
    Wd = np.gradient(g[idx], dt, axis=0) if use_tangential else np.zeros_like(G)
    A = []
    b = []
    for i in range(len(idx)):
        w = float(weights(i))
        M = -np.array([[0, -Wd[i, 2], Wd[i, 1]], [Wd[i, 2], 0, -Wd[i, 0]], [-Wd[i, 1], Wd[i, 0], 0]]) \
            if use_tangential else np.zeros((3, 3))
        M = M + np.outer(G[i], G[i]) - float(G[i] @ G[i]) * np.eye(3)
        A.append(w * M)
        b.append(w * S[i])
    A = np.vstack(A)
    b = np.concatenate(b)
    r = np.linalg.solve(A.T @ A + 1e-12 * np.eye(3), A.T @ b)
    return r


def torque_anchored(k_o, g, a, t, T, dt, km, jzz, sat_axis=2, n_anchor=0):
    """Scale from the commanded yaw torque: integrate ``J_zz * d(wz)/dt = tau_z`` to predict the
    saturated axis, then ``k = |s| / |w|^2``."""
    sat = k_o.saturated(g)
    tau_z = km * (T[:, 0] - T[:, 1] + T[:, 2] - T[:, 3])
    x = np.zeros(len(g))
    if n_anchor > 0 and not sat[n_anchor - 1, sat_axis]:
        x[n_anchor - 1] = g[n_anchor - 1, sat_axis]
    for i in range(n_anchor, len(g)):
        x[i] = x[i - 1] + tau_z[i] / jzz * dt
    other = [j for j in range(3) if j != sat_axis]
    n2 = np.sum(g[:, other] ** 2, axis=1) + x ** 2
    kk = np.linalg.norm(k_o.residual(a, t), axis=1) / np.maximum(n2, 1e-9)
    m = sat[:, sat_axis] & (n2 > 25.0)          # only saturated samples with a meaningful signal
    return float(np.median(kk[m])) if m.any() else None, int(m.sum())


def scan_identify(k_o, g, a, t, dt, k_lo=1e-4, k_hi=6e-2, n=240, sat_axis=2, min_n2=25.0):
    """1-D scan over the scale k; for each k each sample's saturated axis is reconstructed from
    ``|w|^2 = |s|/k`` and the full lever-arm vector is obtained by linear least squares.
    Returns (k_best, cost_curve, r_best, k_true_cost)."""
    sat = k_o.saturated(g)
    s = k_o.residual(a, t)
    other = [j for j in range(3) if j != sat_axis]
    A = np.sum(g[:, other] ** 2, axis=1)
    sign = np.sign(g[:, sat_axis])
    use = sat[:, sat_axis]                     # samples where exactly the chosen axis is the unknown
    ks = np.geomspace(k_lo, k_hi, n)
    costs, best = [], (np.inf, None, None)
    for k in ks:
        n2 = np.linalg.norm(s, axis=1) / k
        x = sign * np.sqrt(np.clip(n2 - A, 0.0, None))
        w = g.copy(); w[:, sat_axis] = x
        M = np.stack([np.outer(w[i], w[i]) - float(w[i] @ w[i]) * np.eye(3) for i in range(len(w))])
        Mm, ss = M[use].reshape(-1, 3), s[use].reshape(-1)
        r = np.linalg.solve(Mm.T @ Mm + 1e-14 * np.eye(3), Mm.T @ ss)
        res = Mm @ r - ss
        cost = float(res @ res) / float(ss @ ss)
        costs.append(cost)
        if cost < best[0]:
            best = (cost, k, r)
    costs = np.array(costs)
    return best[1], costs, best[2], ks


def main():
    lim = np.deg2rad(DPS)
    print(f"=== calibration study, gyro {DPS:.0f} dps (limit {lim:.2f} rad/s) ===")
    print(f"{'scenario':<26} {'all-in-range':>12} {'per-axis-in':>11} {'|w| of those':>14} "
          f"{'k uniform':>10} {'k |w|^2':>9} {'k thr>5':>9} {'k thr>10':>10} {'torque-anchored':>15}")
    for flag, spin0 in [(0, False), (0, True), (2, False), (2, True)]:
        d, km, jzz, dt = collect(flag, spin0)
        g, a, t, w = d["g"], d["a"], d["t"], d["w"]
        k_true = LeverArmObserver.lever_scale_from_arm(np.array(LEVER_ARM), w[-1] + 1e-9)
        ka, kp, sat_axes = in_range_stats(g, lim)
        idx_all = np.where(~np.abs(g).max(axis=1).__ge__(lim - 1e-9))[0]
        mag = np.linalg.norm(w[idx_all], axis=1) if len(idx_all) else np.array([np.nan])
        k_o = LeverArmObserver(gyro_limit_dps=DPS)
        res = {}
        res["uniform"] = weighted_identify(k_o, g, a, t, dt, idx_all, lambda i: 1.0)
        res["w2"] = weighted_identify(k_o, g, a, t, dt, idx_all,
                                      lambda i: max(np.linalg.norm(g[idx_all][i]), 1e-3) ** 2)
        sel = idx_all[np.linalg.norm(w[idx_all], axis=1) > 5.0]
        res["thr5"] = weighted_identify(k_o, g, a, t, dt, sel, lambda i: 1.0)
        sel10 = idx_all[np.linalg.norm(w[idx_all], axis=1) > 10.0]
        res["thr10"] = weighted_identify(k_o, g, a, t, dt, sel10, lambda i: 1.0)
        kt, nt = torque_anchored(k_o, g, a, t, d["T"], dt, km, jzz)
        fmt = lambda r: ("  %6.1f%%" % (100 * abs(np.linalg.norm(r) - k_true) / k_true)) if r is not None else "     n/a"
        print(f"flag={flag} spin0={str(spin0):<5}          {ka:>6}      {kp:>6}      "
              f"{np.nanmean(mag):>6.1f} rad/s   {fmt(res['uniform'])} {fmt(res['w2'])} "
              f"{fmt(res['thr5'])} {fmt(res['thr10'])}  "
              f"{('  %6.1f%% (n=%d)' % (100*abs(kt-k_true)/k_true, nt)) if kt else '      n/a'}")
        k_true = LeverArmObserver.lever_scale_from_arm(np.array(LEVER_ARM), w[-1] + 1e-9)
        ksc, costs, rsc, ks = scan_identify(k_o := LeverArmObserver(gyro_limit_dps=DPS), g, a, t, dt)
        i_true = int(np.argmin(np.abs(ks - k_true)))
        print(f"    scan: k_best={ksc*100:.4f} cm ({100*abs(ksc-k_true)/k_true:5.1f}% err)  "
              f"cost(k_best)={costs.min():.3e}  cost(k_true)={costs[i_true]:.3e}  "
              f"cost(k=10*k_true)={costs[int(np.argmin(np.abs(ks-10*k_true)))]:.3e}  "
              f"sharpness=log10(cost@10k/cost@k*)={np.log10(max(costs[int(np.argmin(np.abs(ks-10*k_true)))],1e-30)/max(costs.min(),1e-30)):.2f}")


if __name__ == "__main__":
    main()