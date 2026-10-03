"""Verifies the lever-arm observer: self-calibration, beyond-range reconstruction, closed loop.

Protocol (default: 3-rotor failure, 1000 dps gyro -- the regime where saturation really loses the
vehicle, see scripts/verify_imu.py):

  phase 1 (calib_steps)  the controller is fed whatever ``--calib-rate`` says; we only harvest IMU
                         samples.  ``measured`` mimics "0.5 s of blind data before recovery".
  phase 2 (rest)         ``env.rate_override`` is written with the observer's reconstructed rate
                         (route A: swap the rate channel of the policy observation).

Reported: identified ``k`` vs the true ``|r_perp|``, per-sample rate-reconstruction error vs the
naive clipped gyro, and the closed-loop altitude/XY error vs the ``measured`` (broken) and
``truth`` (unreachable upper bound) baselines.

Run:  PYTHONPATH=<repo>:<me> python scripts/verify_observer.py [flag] [gyro_dps] [calib_steps]
"""
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.observer import LeverArmObserver  # noqa: E402

sys.path.insert(0, os.path.join(ME, "scripts"))
from verify_imu import load_policy  # noqa: E402

LEVER_ARM = (-0.012, -0.0055, 0.0)
CKPT = {0: "shutdown_real_7", 2: "shutdown_real_7_4"}


def make_env(flag, dps, rate_source, seed=0, spin0=False):
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200,
        gui=False, record=False, obstacles=False, rate_source=rate_source,
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=LEVER_ARM,
                          gyro_noise_std=0.05, accel_noise_std=0.02))
    env.eval = False
    np.random.seed(seed)
    env.shut_down_rotors(flag)
    if spin0:
        # realistic failure onset at hover: the single rotor spins the body up from rest, which
        # takes the gyro *through* the in-range regime and makes k identifiable
        import pybullet as p
        p.resetBaseVelocity(objectUniqueId=env.DRONE_IDS[0], linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0], physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
    return env


def run(flag=2, dps=1000.0, calib_steps=150, calib_rate="measured", steps=1200, seed=0,
        ema=0.0, verbose=True):
    policy = load_policy(CKPT[flag])
    env = make_env(flag, dps, calib_rate, seed)
    ctrl = RLControl(DroneModel.CF2X)
    ctrl.reset()
    obs_est = LeverArmObserver(gyro_limit_dps=dps, ema=ema)
    tp = np.array([0.0, 0.0, 1.0])

    log = {k: [] for k in ("z", "xy", "w_true", "w_hat", "w_clip", "gyro")}
    calib = {k: [] for k in ("g", "a", "t")}
    for i in range(steps):
        raw = env._getDroneStateVector(0)
        ta, tz = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                        cur_quat=raw[3:7], cur_vel=raw[10:13], target_pos=tp)
        r_xy = np.hypot(tz[0], tz[1])
        if r_xy > 0.26:
            s = 0.26 / r_xy
            tz[0] *= s; tz[1] *= s; tz[2] = np.sqrt(1 - tz[0]**2 - tz[1]**2)
        env.target_a, env.target_z_body = float(ta), tz

        # --- sample the IMU once, then hand the reconstructed rate to the policy -------------
        env.RATE_SOURCE = "measured"
        env._computeObs()
        g, a, t = env.gyro_meas.copy(), env.accel_meas.copy(), env.thrust_over_mass
        if i < calib_steps:
            calib["g"].append(g); calib["a"].append(a); calib["t"].append(t)
        if i == calib_steps and i > 0:
            joint_lever = None
            how = obs_est.auto_calibrate(calib["g"], calib["a"], calib["t"],
                                         env.TIMESTEP * env.AGGR_PHY_STEPS)
            k = obs_est.k
            k_true = LeverArmObserver.lever_scale_from_arm(np.array(LEVER_ARM),
                                                          env.omega_true + 1e-9)
            if verbose:
                print(f"  [calib @step {i}] k_est = {k*100:.4f} cm ({how}), "
                      f"k_true ~ {k_true*100:.4f} cm, rel.err = {abs(k-k_true)/k_true*100:.2f}%")
        if obs_est.k is not None:
            env.rate_override = obs_est.step(g, a, t)
            env.RATE_SOURCE = "override"
        obs = env._computeObs()
        action = policy.select_action(obs, deterministic=True)
        env.step(action)

        log["z"].append(env.pos[0][2]); log["xy"].append(np.linalg.norm(env.pos[0][:2]))
        log["w_true"].append(env.omega_true.copy())
        log["w_hat"].append(np.asarray(env.rate_override).copy() if env.rate_override is not None
                            else g.copy())
        log["w_clip"].append(g.copy())
        log["gyro"].append(g.copy())
    env.close()
    out = {k: np.array(v) for k, v in log.items()}
    out["k_est"] = obs_est.k
    return out


def summarise(tag, out, flag, dps):
    z, xy = out["z"], out["xy"]
    w_t, w_h, w_c = out["w_true"], out["w_hat"], out["w_clip"]
    lim = np.deg2rad(dps)
    sat = (np.abs(w_t) > lim).any(axis=1)
    e_hat = np.linalg.norm(w_h - w_t, axis=1)
    e_clip = np.linalg.norm(w_c - w_t, axis=1)
    zl, xl = z[-200:].mean(), xy[-200:].mean()
    print(f"  {tag:<22} z_last={zl:7.2f} m (std {z[-200:].std():5.2f})  xy_last={xl:6.3f} m"
          f"   rate err: obs {e_hat[sat].mean():6.2f} vs clipped {e_clip[sat].mean():6.2f} rad/s"
          f"  [{'saved' if zl > 0.3 else 'LOST'}]")
    return dict(z=zl, xy=xl, err_obs=float(e_hat[sat].mean()), err_clip=float(e_clip[sat].mean()))


def main():
    flag = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dps = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
    calib_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    print(f"=== lever-arm observer | flag={flag} ({4 - sum(1 for x in [0,0,0,1] if x == 0)} rotors "
          f"failed), gyro {dps:.0f} dps, calibration over {calib_steps} steps ===")

    res = {}
    print(" phase 1 = 'truth' (estimator calibrated before the failure is felt):")
    res["obs(truth-calib)"] = summarise("observer (truth-calib)", run(
        flag, dps, calib_steps, calib_rate="truth"), flag, dps)
    print(" phase 1 = 'measured' (0.5 s of blind data, then recovery):")
    res["obs(blind-calib)"] = summarise("observer (blind-calib)", run(
        flag, dps, calib_steps, calib_rate="measured"), flag, dps)
    print(" baselines:")
    res["truth"] = summarise("upper bound (truth rate)", _baseline(flag, dps, "truth"), flag, dps)
    res["measured"] = summarise("broken (saturated gyro)", _baseline(flag, dps, "measured"), flag, dps)
    print("OK")


def _baseline(flag, dps, rate_source):
    policy = load_policy(CKPT[flag])
    env = make_env(flag, dps, rate_source)
    ctrl = RLControl(DroneModel.CF2X)
    ctrl.reset()
    tp = np.array([0.0, 0.0, 1.0])
    log = {k: [] for k in ("z", "xy", "w_true", "w_hat", "w_clip")}
    for _ in range(1200):
        raw = env._getDroneStateVector(0)
        ta, tz = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                        cur_quat=raw[3:7], cur_vel=raw[10:13], target_pos=tp)
        r_xy = np.hypot(tz[0], tz[1])
        if r_xy > 0.26:
            s = 0.26 / r_xy
            tz[0] *= s; tz[1] *= s; tz[2] = np.sqrt(1 - tz[0]**2 - tz[1]**2)
        env.target_a, env.target_z_body = float(ta), tz
        obs = env._computeObs()
        action = policy.select_action(obs, deterministic=True)
        env.step(action)
        log["z"].append(env.pos[0][2]); log["xy"].append(np.linalg.norm(env.pos[0][:2]))
        log["w_true"].append(env.omega_true.copy()); log["w_hat"].append(env.gyro_meas.copy())
        log["w_clip"].append(env.gyro_meas.copy())
    env.close()
    return {k: np.array(v) for k, v in log.items()}


if __name__ == "__main__":
    main()