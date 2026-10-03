"""Complete IMU model: the *attitude* is now also propagated from the range-limited gyro.

Before this script, only the explicit rate input of the controller was replaced; the attitude came
straight from the simulator, so saturation could not corrupt it.  Here ``--att ins`` integrates the
attitude with the same clipped gyro the controller sees (`gpd_me.ins.AttitudeINS`), i.e. **every
consumer of a saturated axis gets the range value**, exactly as a real IMU would.

Stacks:
  rl     the senior's stack: outer position PID -> 15-D obs -> ACRL policy -> thrusts
  indi   the senior's traditional baseline: allocation (failed rotor excluded) + INDI + reduced attitude

Sources:
  truth      ground-truth rate and attitude      (unreachable upper bound)
  measured   saturated, noisy gyro               (the failure case)
  override   lever-arm observer reconstruction   (route A)

A faithfulness check runs first: with truth attitude/rate and zero noise, the re-implemented 15-D
observation must equal the environment's own `_computeObs()` to 1e-6 (it is the senior's obs, only
with the attitude/rate replaced).

Run:  PYTHONPATH=<repo>:<me> python scripts/verify_ins_attitude.py
"""
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.spatial.transform import Rotation  # noqa: E402

from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.indi import (FAILED_MOTOR, INDIController, PositionController as IndiPosCtl,  # noqa: E402
                         PrimaryAxisAttitudeController, VALID_MOTORS)
from gpd_me.ins import AttitudeINS  # noqa: E402
from gpd_me.observer import LeverArmObserver  # noqa: E402
from gpd_me.policy import PositionPID, ActuatorLag, load_policy, quat_to_matrix, mixer_body_wrench  # noqa: E402

LEVER_ARM = (-0.012, -0.0055, 0.0)
CKPT = {0: "shutdown_real_7", 2: "shutdown_real_7_4", 3: "shutdown_real_7_4"}
MASK = {0: [0, 1, 1, 1], 1: [0, 1, 0, 1], 2: [0, 0, 0, 1], 3: [0, 0, 1, 1]}
DT = 1.0 / 200.0


def make_env(flag, dps, freq=200, noise=True, rate_source="measured"):
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=freq, gui=False, record=False,
        obstacles=False, rate_source=rate_source,
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=LEVER_ARM,
                          gyro_noise_std=0.05 if noise else 0.0,
                          accel_noise_std=0.02 if noise else 0.0))
    env.eval = False
    env.reset()
    env.shut_down_rotors(flag)
    return env


def build_obs(R, rate, target_a, tom, last_action, mask):
    rel = R.T @ np.asarray(target_a[1], float)          # target body-z direction in body frame
    des_rad = rel[:2]
    return np.array([des_rad[0], des_rad[1],
                     rate[0] / 10.0, rate[1] / 10.0, rate[2] / 50.0,
                     (target_a[0] - 9.8) / 3.0, (tom - 9.8) / 3.0,
                     *last_action, *(mask * 2 - 1)], dtype=np.float32)


def faithfulness_check():
    """My 15-D obs must equal the env's own with truth attitude/rate and no noise."""
    env = make_env(0, 1e9, noise=False, rate_source="truth")
    env.gyro_noise_std, env.imu_noise_std = 0.0, 0.0
    pid = PositionPID()
    rng = np.random.default_rng(0)
    mask = np.array(MASK[0], float)
    last_action = -np.ones(4)
    worst = 0.0
    for i in range(60):
        ta = pid.step(DT, env.pos[0], env.quat[0], env.vel[0], np.array([0.0, 0.0, 1.0]))
        env.target_a, env.target_z_body = float(ta[0]), np.asarray(ta[1], float)
        obs_env = env._computeObs()
        R = quat_to_matrix(env.quat[0])
        ta = (float(ta[0]), np.asarray(ta[1], float))
        obs_mine = build_obs(R, env.omega_true, ta, float(np.atleast_1d(env.last_acc)[0]),
                             last_action * mask, mask)
        worst = max(worst, float(np.abs(obs_mine - obs_env).max()))
        action = rng.uniform(-1, 1, 4)
        last_action = action.copy()
        env.step(action)
    env.close()
    return worst


def run(stack="rl", flag=0, dps=1000.0, att="ins", src="measured", steps=1600,
        calib_steps=250, target=(0.0, 0.0, 1.0), seed=0, ins_rate="gyro", preset_k=None):
    np.random.seed(seed)
    freq = 250 if stack == "indi" else 200
    global DT
    DT = 1.0 / freq
    env = make_env(flag, dps, freq=freq, rate_source="measured")
    mask = np.array(env.shut_down, float)
    w_true = np.array(MASK[flag], float)
    assert np.array_equal(mask, w_true), (mask, w_true)
    policy = load_policy(CKPT[flag]) if stack == "rl" else None
    pid = PositionPID()
    indi_pos = IndiPosCtl(DT)
    indi_att = PrimaryAxisAttitudeController(DT)
    indi = INDIController(DT)
    lag = ActuatorLag(float(env.delay))
    ins = AttitudeINS(quat_to_matrix(env.quat[0]))
    obs_est = LeverArmObserver(gyro_limit_dps=dps) if src == "override" else None
    if obs_est is not None and preset_k is not None:
        obs_est.k = float(preset_k); obs_est.auto_calibrated = True
    tp = np.array(target, float)
    last_action = -np.ones(4)
    f_real_prev = np.zeros((3, 1))
    calib = {k: [] for k in ("g", "a", "t")}
    z, xy, tilt_err, ang_err, sat = [], [], [], [], []
    rate_err_sat, sat_flags = [], []
    last_counter = -1

    for i in range(steps):
        env._computeObs()                     # sample the IMU once per control step
        gyro, accel, tom = env.gyro_meas.copy(), env.accel_meas.copy(), env.thrust_over_mass

        # ---- rate source: the same value can also drive the attitude INS (that is the point) ----
        if src == "truth":
            rate = env.omega_true.copy()
        elif src == "measured":
            rate = gyro
        else:
            if i < calib_steps:
                calib["g"].append(gyro); calib["a"].append(accel); calib["t"].append(tom)
                rate = gyro
            else:
                if (obs_est.k is None or preset_k is None) and not obs_est.auto_calibrated:
                    try:
                        obs_est.auto_calibrate(calib["g"], calib["a"], calib["t"], DT)
                        obs_est.auto_calibrated = True
                    except ValueError:
                        obs_est.auto_calibrated = True      # nothing saturated -> gyro == truth
                rate = obs_est.step(gyro, accel, tom) if obs_est.k is not None else gyro

        if env.step_counter != last_counter:
            last_counter = env.step_counter
            ins.update(rate if ins_rate == "estimate" else gyro, DT)
        R_true = quat_to_matrix(env.quat[0])
        R = ins.R if att == "ins" else R_true

        if stack == "rl":
            q_att = Rotation.from_matrix(R).as_quat()
            ta_scalar, z_body = pid.step(DT, env.pos[0], q_att, env.vel[0], tp)
            r_xy = np.hypot(z_body[0], z_body[1])
            if r_xy > 0.26:
                s = 0.26 / r_xy
                z_body = np.array([z_body[0] * s, z_body[1] * s,
                                   np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
            obs = build_obs(R, rate, (ta_scalar, z_body), tom, last_action * mask, mask)
            action = policy.select_action(obs, deterministic=True)
            last_action = action.copy()
            forces = np.clip((action + 1.0) * 7.5, 0.0, 15.0) * mask
            thrust = lag.step(forces, DT)
            env.step(action)                  # the env re-computes its own (ignored) obs
        else:
            acc_I_des, n_des_I, n_des_I_dot = indi_pos.calc(tp.reshape(3, 1),
                                                            env.pos[0].reshape(3, 1),
                                                            env.vel[0].reshape(3, 1))
            p_des, q_des, f_z_des, p_des_dot, q_des_dot = indi_att.calc(
                R, rate[2], acc_I_des, n_des_I, n_des_I_dot)
            u = indi.calc(rate[0], p_des, p_des_dot, rate[1], q_des, q_des_dot,
                          float(np.sum(f_real_prev)) / indi.mass, f_z_des, f_real_prev)
            f_motors = np.zeros(4)
            f_motors[VALID_MOTORS] = u.flatten()
            f_motors[FAILED_MOTOR] = 0.0
            f_motors = np.clip(f_motors, 0, 6)
            f_real_prev = u
            # invert the env's action->thrust map: (a+1)*7.5 clipped to [0,15]
            env.step(f_motors.reshape(1, 4) / 7.5 - 1.0)

        z.append(float(env.pos[0][2]))
        xy.append(float(np.hypot(env.pos[0][0], env.pos[0][1])))
        tilt_err.append(AttitudeINS.tilt_error(R, R_true))
        ang_err.append(AttitudeINS.angle_error(R, R_true))
        is_sat = bool(np.any(np.abs(env.omega_true) > np.deg2rad(dps)))
        sat.append(is_sat)
        rate_err_sat.append(float(np.linalg.norm(np.asarray(rate, float) - env.omega_true)))
        sat_flags.append(is_sat)

    env.close()
    z, xy = np.array(z), np.array(xy)
    w = max(20, len(z) // 4)
    re_s = np.array(rate_err_sat)[np.array(sat_flags)] if any(sat_flags) else np.array([])
    return dict(z=float(z[-w:].mean()), zstd=float(z[-w:].std()), xy=float(xy[-w:].mean()),
                tilt=float(np.rad2deg(np.mean(tilt_err[-w:]))),
                angle=float(np.rad2deg(np.mean(ang_err[-w:]))),
                sat=100.0 * float(np.mean(sat)),
                rate_err=float(re_s.mean()) if re_s.size else 0.0,
                n_sat=int(re_s.size))


def main():
    worst = faithfulness_check()
    print(f"[check] re-implemented obs vs env obs (truth attitude+rate, zero noise): "
          f"max |diff| = {worst:.2e}  -> {'OK' if worst < 1e-5 else 'MISMATCH'}\n")

    ref = {}
    for stack, flag in (("rl", 0), ("rl", 2), ("rl", 3), ("indi", 0)):
        ref[(stack, flag)] = run(stack=stack, flag=flag, dps=4000.0, att="truth", src="truth")

    print(f"{'stack':<6} {'flag':>4} {'dps':>6} {'attitude':<9} {'rate':<9} "
          f"{'z':>7} {'std':>6} {'xy':>7} {'tilt err':>9} {'att err':>8} {'sat%':>5}  verdict")
    for stack, flag in (("rl", 0), ("rl", 2), ("rl", 3), ("indi", 0)):
        zr = ref[(stack, flag)]["z"]
        for dps in (4000.0, 1000.0):
            for att in ("truth", "ins"):
                for src in ("truth", "measured"):
                    r = run(stack=stack, flag=flag, dps=dps, att=att, src=src)
                    stable = r["z"] > 0.3 and abs(r["z"] - zr) < 0.6
                    print(f"{stack:<6} {flag:>4} {dps:>6.0f} {att:<9} {src:<9} "
                          f"{r['z']:>7.2f} {r['zstd']:>6.2f} {r['xy']:>7.3f} "
                          f"{r['tilt']:>9.1f} {r['angle']:>8.1f} {r['sat']:>5.0f}  "
                          f"{'holds' if stable else 'LOST'}")
        print(f"       (reference for flag={flag}: truth attitude+rate -> z={zr:.2f})")


if __name__ == "__main__":
    main()
