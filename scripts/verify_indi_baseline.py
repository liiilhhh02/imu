"""Traditional-control baseline (`baseline_INDI.ipynb`) under gyro saturation.

The senior's INDI + control-allocation baseline is run for the single-rotor-failure case
(`FAILED_MOTOR = 0`) with the controller fed either the ground-truth body rate, the *saturated*
gyro, or the lever-arm observer's reconstruction.  Everything else is the notebook's own loop.

Question answered: does a traditional controller suffer badly from the over-range gyro?

Run:  PYTHONPATH=<repo>:<me> python scripts/verify_indi_baseline.py
"""
import os
import sys
import time

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pybullet as p  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from gym_pybullet_drones.envs.MetaShutDown_baseline import MetaAviary as BaselineEnv  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.imu import IMU, IMUConfig  # noqa: E402
from gpd_me.indi import (FAILED_MOTOR, INDIController, LPF, PositionController,  # noqa: E402
                         PrimaryAxisAttitudeController, VALID_MOTORS)
from gpd_me.observer import LeverArmObserver  # noqa: E402

LEVER_ARM = (-0.012, -0.0055, 0.0)
FREQ = 250
MASK_FOR_RUN = [0, 1, 1, 1]


def run(dps=1000.0, src="measured", steps=None, target=(0.0, 0.0, 1.0), seed=0,
        calib_steps=200, verbose=False):
    ts = 1.0 / FREQ
    steps = steps or FREQ * 6
    env = BaselineEnv(drone_model=DroneModel.CF2X, num_drones=1,
                      initial_xyzs=np.array([[0.0, 0.0, 1.0]]), physics=Physics("pyb"),
                      freq=FREQ, gui=False, record=False, obstacles=False)
    env.eval = False
    np.random.seed(seed)
    env.shut_down_rotors(0)
    env.shut_down = np.array(MASK_FOR_RUN, dtype=float)

    pos_controller = PositionController(ts)
    pat_controller = PrimaryAxisAttitudeController(ts)
    imu = IMU(IMUConfig(gyro_range_dps=dps, lever_arm=LEVER_ARM,
                        gyro_noise_std=0.05, accel_noise_std=0.02))
    obs_est = LeverArmObserver(gyro_limit_dps=dps) if src == "override" else None

    pos_desired = np.array(target, float).reshape(3, 1)
    valid = [i for i in range(4) if int(env.shut_down[i]) == 1]
    indi = INDIController(ts, allocation_failed=None, valid_motors=valid)
    f_real_prev = np.zeros((len(valid), 1))
    prev_omega = None
    calib = {k: [] for k in ("g", "a", "t")}
    z, xy, wt, wu = [], [], [], []

    for i in range(steps):
        sv = env._getDroneStateVector(0)
        pos = sv[0:3].reshape(3, 1)
        vel = sv[10:13].reshape(3, 1)
        R = Rotation.from_quat(sv[3:7]).as_matrix()
        omega_body = R.T @ sv[13:16].reshape(3, 1)
        wdot = np.zeros(3) if prev_omega is None else ((omega_body[:, 0] - prev_omega) / ts)
        prev_omega = omega_body[:, 0].copy()
        tom = float(np.sum(f_real_prev)) / indi.mass

        gyro, accel = imu.measure(omega_body[:, 0], wdot, tom, ts, force=True)
        if src == "truth":
            rate = omega_body[:, 0]
        elif src == "measured":
            rate = gyro
        else:
            if i < calib_steps:
                calib["g"].append(gyro); calib["a"].append(accel); calib["t"].append(tom)
                rate = gyro
            else:
                if obs_est.k is None:
                    obs_est.auto_calibrate(calib["g"], calib["a"], calib["t"], ts)
                rate = obs_est.step(gyro, accel, tom)
        p_rate, q_rate, r_rate = rate[0], rate[1], rate[2]

        acc_I_des, n_des_I, n_des_I_dot = pos_controller.calc(pos_desired, pos, vel)
        p_des, q_des, f_z_des, p_des_dot, q_des_dot = pat_controller.calc(
            R, r_rate, acc_I_des, n_des_I, n_des_I_dot)
        u_target = indi.calc(p_rate, p_des, p_des_dot, q_rate, q_des, q_des_dot,
                             tom, f_z_des, f_real_prev)

        f_motors = np.zeros(4)
        f_motors[valid] = u_target.flatten()
        f_motors = np.clip(f_motors, 0, 6)
        f_real_prev = u_target
        env.step(f_motors.reshape(1, 4))

        z.append(float(env.pos[0][2]))
        xy.append(float(np.hypot(env.pos[0][0], env.pos[0][1])))
        wt.append(env.omega_true.copy() if hasattr(env, "omega_true") else omega_body[:, 0].copy())
        wu.append(np.asarray(rate, float).copy())

    km = float(env.KM)
    env.close()
    z, xy, wt, wu = map(np.array, (z, xy, wt, wu))
    w = len(z) // 4
    return dict(z=float(z[-w:].mean()), zstd=float(z[-w:].std()), xy=float(xy[-w:].mean()),
                wmag=float(np.linalg.norm(wt[-w:], axis=1).mean()), km=km,
                wxy=float(np.abs(wt[-w:, :2]).max()), z_trajectory=z)


def main():
    print("=== senior's INDI + allocation baseline, single rotor failure (FAILED_MOTOR=0) ===")
    print(f"{'gyro range':>11} {'rate source':<10} {'z (last 1.5 s)':>16} {'std':>7} "
          f"{'xy':>7} {'|w|':>7} {'|w_xy|max':>10}   verdict")
    for dps in (4000.0, 1000.0, 300.0, 150.0, 100.0, 60.0):
        for src in ("truth", "measured"):
            r = run(dps=dps, src=src)
            ok = "holds" if abs(r["z"] - 1.0) < 0.5 else "LOST"
            print(f"{dps:>11.0f} {src:<10} {r['z']:>16.2f} {r['zstd']:>7.2f} "
                  f"{r['xy']:>7.3f} {r['wmag']:>7.1f} {r['wxy']:>10.2f}   {ok}")

    print("\n=== control-allocation feasibility per failure mask (A is 3 x n_valid) ===")
    from gpd_me.indi import ALLOCATION_FULL
    MASK = {0: [0, 1, 1, 1], 1: [0, 1, 0, 1], 2: [0, 0, 0, 1], 3: [0, 0, 1, 1]}
    for flag, mask in MASK.items():
        valid = [i for i in range(4) if mask[i] == 1]
        A = ALLOCATION_FULL[:, valid]
        rank = np.linalg.matrix_rank(A)
        extra = ""
        if len(valid) == 3:
            extra = f"det={np.linalg.det(A):+.5f}"
        print(f"  flag={flag} mask={mask} valid={valid}  rank={rank}/3  "
              f"{'INDI well-posed' if rank == 3 else 'RANK DEFICIENT -> must surrender an objective (e.g. yaw)'}"
              f"  {extra}")


if __name__ == "__main__":
    main()