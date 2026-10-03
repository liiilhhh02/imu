"""Is there exploitable regularity in the *in-range* segment? (frequency-domain analysis)

The user's hypothesis: while the gyro is inside its range the data carry structure (temporal /
spectral) that can be identified once and then extrapolated into the saturated segment.  This script
tests it on ground truth, using the scenario where in-range data actually exist (1 rotor failed,
where the body spins up from rest through the whole range):

  1. how the rate is distributed in frequency (is it concentrated at low frequency?),
  2. whether the yaw channel is a first-order system from the *known* control input
     `tau_z = KM*(T0-T1+T2-T3)` to the body rate, identified on in-range samples only,
  3. the decisive test: **identify the model on in-range samples, then free-run it through the
     saturated segment** and compare against ground truth, the clipped gyro and the algebraic
     lever-arm estimate.

Run:  PYTHONPATH=<repo>:<me> python scripts/analyze_regularity.py
"""
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, os.path.join(ME, "scripts"))
from verify_ins_attitude import make_env, CKPT  # noqa: E402
from gpd_me.policy import PositionPID, load_policy, quat_to_matrix  # noqa: E402
from gpd_me.observer import LeverArmObserver  # noqa: E402
from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel  # noqa: E402

DPS = 1000.0
DT = 1.0 / 200.0


def collect(flag=0, steps=4000, seed=0):
    np.random.seed(seed)
    env = make_env(flag, DPS, freq=200, rate_source="truth")
    policy = load_policy(CKPT[flag])
    ctrl = RLControl(DroneModel.CF2X)
    ctrl.reset()
    tp = np.array([0.0, 0.0, 1.0])
    log = {k: [] for k in ("g", "a", "T", "w", "q")}
    for _ in range(steps):
        raw = env._getDroneStateVector(0)
        ta, tz = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                        cur_quat=raw[3:7], cur_vel=raw[10:13], target_pos=tp)
        r = np.hypot(tz[0], tz[1])
        if r > 0.26:
            s = 0.26 / r
            tz = np.array([tz[0] * s, tz[1] * s, np.sqrt(1 - (tz[0] * s) ** 2 - (tz[1] * s) ** 2)])
        env.target_a, env.target_z_body = float(ta), tz
        env._computeObs()
        log["g"].append(env.gyro_meas.copy())
        log["a"].append(env.accel_meas.copy())
        log["T"].append(np.asarray(env.thrust[0], float).copy())
        log["w"].append(env.omega_true.copy())
        log["q"].append(env.quat[0].copy())
        env.step(policy.select_action(env._computeObs(), deterministic=True))
    km, m, delay = float(env.KM), float(env.M), float(env.delay)
    env.close()
    return {k: np.array(v) for k, v in log.items()}, km, m, delay


def main():
    d, km, m, delay = collect()
    w, g, T = d["w"], d["g"], d["T"]
    lim = np.deg2rad(DPS)
    sat = (np.abs(g) >= lim - 1e-9).any(axis=1)
    inr = ~sat
    tau_z = km * (T[:, 0] - T[:, 1] + T[:, 2] - T[:, 3])
    print(f"=== regularity analysis, 1-rotor failure, {DPS:.0f} dps, {len(w)} steps "
          f"({100*inr.mean():.0f}% in range) ===")

    # ---- 1) spectral content of the rate (in-range steps only) ----
    wz = w[inr, 2]
    if wz.size > 64:
        f, p = np.fft.rfft(wz - wz.mean()), None
        p = np.abs(f) ** 2
        freq = np.fft.rfftfreq(wz.size, DT)
        tot = p.sum()
        c95 = freq[np.searchsorted(np.cumsum(p) / tot, 0.95)]
        print(f"  1) body rate spectrum (in range): 95% of the power below {c95:.2f} Hz "
              f"(Nyquist {0.5/DT:.0f} Hz), mean |wz| = {np.abs(wz).mean():.1f} rad/s, "
              f"std = {wz.std():.2f} rad/s")

    # ---- 2) identify the yaw dynamics on in-range samples: wz[k+1] = a*wz[k] + b*tau_z[k] ----
    k = np.where(inr[:-1] & inr[1:])[0]
    A = np.stack([w[k, 2], tau_z[k], np.ones_like(k, float)], axis=1)
    bvec = w[k + 1, 2]
    sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    a, b, c = sol
    decay = -DT / np.log(max(a, 1e-9))
    dc_gain = b / max(1 - a, 1e-9)
    print(f"  2) yaw ARX(1,1) from the *known* torque input (fit on {len(k)} in-range pairs):")
    print(f"     a={a:.6f}  b={b:+.6e}  offset={c:+.4f}")
    print(f"     -> equivalent first-order time constant {decay:.3f} s "
          f"({1/(2*np.pi*decay):.3f} Hz corner), DC gain {dc_gain:.2f} rad/s per N*m")
    print(f"     -> implies damping/coefficient ratio: tau_z/|wz| at steady state = "
          f"{1/max(dc_gain,1e-9):.5f} N*m per rad/s")

    # ---- 3) decisive test: free-run the identified model through the saturated segment ----
    first_sat = int(np.argmax(sat)) if sat.any() else len(w) - 1
    k0 = first_sat - 1
    w_pred = np.zeros(len(w))
    w_pred[k0] = w[k0, 2]
    for i in range(k0, len(w) - 1):
        w_pred[i + 1] = a * w_pred[i] + b * tau_z[i] + c
    seg = slice(first_sat, len(w))
    kz = 2
    err_model = np.abs(w_pred[seg] - w[seg, kz]).mean()
    err_clip = np.abs(g[seg, kz] - w[seg, kz]).mean()
    print(f"  3) free-run prediction through the saturated segment ({len(w)-first_sat} steps):")
    print(f"     |wz| true   = {np.abs(w[seg, kz]).mean():7.2f} rad/s")
    print(f"     model error = {err_model:7.2f} rad/s")
    print(f"     clipped-gyro error = {err_clip:7.2f} rad/s")
    print(f"     -> the in-range-identified model {'BEATS' if err_model < err_clip else 'DOES NOT BEAT'}"
          f" the clipped gyro by {(1-err_model/err_clip)*100:+.0f}%")

    # ---- 4) what does the accelerometer actually carry? ----
    s = d["a"] - np.array([0.0, 0.0, 1.0])[None, :] * (T.sum(axis=1) / m)[:, None]
    n2 = np.sum(w ** 2, axis=1)
    s_norm = np.linalg.norm(s, axis=1)
    kw = np.median((s_norm[inr & (n2 > 25)] / n2[inr & (n2 > 25)])) if (inr & (n2 > 25)).any() else np.nan
    print(f"  4) lever-arm invariant |s| = |w|^2 * k on in-range samples: k = {kw*100:.4f} cm "
          f"(true |r_perp| = {np.linalg.norm(np.array([-0.012, -0.0055, 0.0]))*100:.4f} cm)")
    print(f"     |w|^2 carries a DC term plus {np.std(n2)/np.mean(n2)*100:.1f}% ripple (AC/DC)")
    # how well does the algebraic reconstruction do when driven by the in-range k?
    ob = LeverArmObserver(gyro_limit_dps=DPS, lever_scale=kw)
    rec = np.stack([ob.step(g[i], d["a"][i], T.sum(axis=1)[i] / m) for i in range(len(w))])
    err_alg = np.abs(rec[seg, kz] - w[seg, kz]).mean()
    print(f"     algebraic reconstruction error on saturated steps: {err_alg:7.2f} rad/s "
          f"(clipped: {err_clip:.2f})")

    # ---- 5) FUSION: identified process model + lever-arm measurement -> scalar Kalman filter ----
    # This is the classical (non-learned) version of the architecture the network must beat:
    #   process : wz[k+1] = a wz[k] + b tau_z[k]        (identified on in-range data)
    #   measurement (only when saturated): |wz| = sqrt(|s|/k - wx^2 - wy^2), sign from the nozzle
    sig_a = 0.02                    # accelerometer noise -> measurement noise on |w|^2
    P = 1e-3                        # initial covariance
    Q = 1e-6                        # process noise (model mismatch)
    wz_hat = w[k0, 2]
    err_kf, err_kf_n = 0.0, 0
    for i in range(k0, len(w) - 1):
        # predict
        wz_hat = a * wz_hat + b * tau_z[i] + c
        P = a * a * P + Q
        # update with the lever-arm invariant where the gyro is saturated
        if sat[i + 1]:
            wx, wy = g[i + 1, 0], g[i + 1, 1]
            n2_meas = s_norm[i + 1] / max(kw, 1e-12)
            val = n2_meas - wx * wx - wy * wy
            if val > 1.0:
                h = np.sqrt(val) * (np.sign(g[i + 1, 2]) or -1.0)
                # sigma on |w| from sigma on |s|: |w| = sqrt(|s|/k) -> dw = ds/(2 sqrt(k |s|))
                R = (sig_a / (2.0 * np.sqrt(max(kw * n2_meas, 1e-12)))) ** 2 + 1e-4
                K = P / (P + R)
                wz_hat = wz_hat + K * (h - wz_hat)
                P = (1 - K) * P
                err_kf += abs(wz_hat - w[i + 1, kz]); err_kf_n += 1
    err_fusion = err_kf / max(err_kf_n, 1)
    print(f"  5) FUSION (identified model + lever-arm measurement, scalar KF) on saturated steps:")
    print(f"     clipped gyro     : {err_clip:6.2f} rad/s")
    print(f"     open-loop model  : {err_model:6.2f} rad/s")
    print(f"     algebraic only   : {err_alg:6.2f} rad/s")
    print(f"     model+meas fusion: {err_fusion:6.2f} rad/s  <-- what the network must beat")
    print(f"     k identified on in-range data only: {kw*100:.4f} cm vs true "
          f"{np.linalg.norm(np.array([-0.012, -0.0055, 0.0]))*100:.4f} cm")


if __name__ == "__main__":
    main()