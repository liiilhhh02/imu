"""Verifies the IMU model against a pybullet finite-difference ground truth.

Two rollouts of the 1-motor-failure case (`shutdown_real_7`, flag 0):

  1. ``rate_source="truth"``  the controller is fed the ground-truth rate, so the vehicle stays
     airborne and the accelerometer predictions can be compared in a sane flight regime:

         model :  a_model = (T/M) e_z + (dw/dt) x r + w x (w x r)          [gpd_me.imu]
         truth :  a_true  = R^T ( d v_IMU / dt + [0,0,+G] ),  v_IMU = v_CoM + w x (R r)

     ``w`` is recomputed from pybullet (``R^T w_world``), never read from the env, and all
     derivatives are central differences over the logged series.

  2. ``rate_source="measured"``  the controller only gets the saturated, noisy gyro -> the vehicle
     loses control.  This run quantifies the problem the estimator has to solve.

Run:  PYTHONPATH=<repo>:<me> python scripts/verify_imu.py [flag] [gyro_range_dps]
"""
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pybullet as p  # noqa: E402
import torch  # noqa: E402

from gym_pybullet_drones.algo.ACRL import ACRL  # noqa: E402
from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402


class ACRLArgs:
    state_dim = 15
    num_critics = 2
    hidden_dim_actor = 64
    hidden_dim_critic = 128
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    gamma = 0.99
    tau = 0.005
    eta = 0


def quat_to_matrix(q):
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


def central_diff(x, dt):
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def load_policy(ckpt="shutdown_real_7"):
    dev = torch.device("cpu")
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": k.get("map_location", dev)})
    policy = ACRL(state_dim=15, action_dim=4, max_action=1, device=dev, args=ACRLArgs())
    policy.load(ckpt, os.path.join(REPO, "gym_pybullet_drones", "model"))
    return policy


def rollout(flag=0, steps=1200, freq=200, gyro_dps=1000.0, rate_source="truth",
            lever_arm=(-0.012, -0.0055, 0.0), ckpt="shutdown_real_7",
            noise=True):
    policy = load_policy(ckpt)
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=freq,
        gui=False, record=False, obstacles=False,
        rate_source=rate_source,
        imu_cfg=IMUConfig(gyro_range_dps=gyro_dps, lever_arm=lever_arm,
                          gyro_noise_std=0.05 if noise else 0.0,
                          accel_noise_std=0.02 if noise else 0.0))
    env.eval = False
    ctrl = RLControl(DroneModel.CF2X)
    ctrl.reset()
    env.shut_down_rotors(flag)

    tp = np.array([0.0, 0.0, 1.0])
    rows = []
    for _ in range(steps):
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

        v_w, w_w = p.getBaseVelocity(env.DRONE_IDS[0], physicsClientId=env.CLIENT)
        _, quat = p.getBasePositionAndOrientation(env.DRONE_IDS[0], physicsClientId=env.CLIENT)
        rows.append(dict(v=np.array(v_w), w_world=np.array(w_w), quat=np.array(quat),
                         thrust=float(np.atleast_1d(env.last_acc)[0]),
                         gyro=env.gyro_meas.copy(), accel=env.accel_meas.copy(),
                         lever=env.lever_arm.copy(),
                         z=float(env.pos[0][2]), dt=env.TIMESTEP * env.AGGR_PHY_STEPS,
                         G=float(env.G)))
    env.close()
    return rows


def analyse(rows, lim_rad_s, lo_frac=0.25):
    n = len(rows)
    dt = rows[0]["dt"]
    g = np.array([0.0, 0.0, rows[0]["G"]])
    R = np.stack([quat_to_matrix(r["quat"]) for r in rows])
    v = np.stack([r["v"] for r in rows])
    w_world = np.stack([r["w_world"] for r in rows])
    thrust = np.array([r["thrust"] for r in rows])
    gyro = np.stack([r["gyro"] for r in rows])
    r_arm = rows[0]["lever"] if "lever" in rows[0] else None
    w = np.einsum("nji,nj->ni", R, w_world)
    lo = n // 4
    return dict(n=n, dt=dt, R=R, v=v, w=w, thrust=thrust, gyro=gyro, lo=lo, g=g, r_arm=r_arm)


def main():
    flag = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gyro_dps = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
    lim = np.deg2rad(gyro_dps)
    r_arm = np.array([-0.012, -0.0055, 0.0])

    # ---------- 1) physics check in a stable flight (controller gets the truth) ----------------
    rows = rollout(flag=flag, gyro_dps=gyro_dps, rate_source="truth")
    a = analyse(rows, lim)
    dt, lo, R, v, w = a["dt"], a["lo"], a["R"], a["v"], a["w"]
    thrust, gyro, g = a["thrust"], a["gyro"], a["g"]
    wdot = central_diff(w, dt)
    v_imu = np.stack([v[i] + np.cross(R[i] @ w[i], R[i] @ r_arm) for i in range(a["n"])])
    a_imu = central_diff(v_imu, dt)
    a_true = np.stack([R[i].T @ (a_imu[i] + g) for i in range(a["n"])])
    a_model = np.stack([np.array([0, 0, thrust[i]]) + np.cross(wdot[i], r_arm)
                        + np.cross(w[i], np.cross(w[i], r_arm)) for i in range(a["n"])])
    dv = central_diff(v, dt)
    a_com_true = np.stack([R[i].T @ (dv[i] + g) for i in range(a["n"])])

    err = np.linalg.norm(a_model[lo:] - a_true[lo:], axis=1)
    err_com = np.abs(a_com_true[lo:, 2] - thrust[lo:])
    mag = np.linalg.norm(w[lo:], axis=1)
    lev = np.linalg.norm(np.cross(w[lo:], np.cross(w[lo:], r_arm)), axis=1)
    sat = (np.abs(gyro[lo:]) >= lim - 1e-6).any(axis=1)
    clip_err = np.linalg.norm(np.clip(w[lo:], -lim, lim) - w[lo:], axis=1)
    z = np.array([r["z"] for r in rows])

    print(f"=== IMU model vs pybullet ground truth (flag={flag}, {a['n']} steps @ {1/dt:.0f} Hz) ===")
    print(f"  lever arm r = {r_arm}, |r| = {np.linalg.norm(r_arm)*100:.2f} cm,"
          f"  gyro range = {gyro_dps:.0f} dps = {lim:.2f} rad/s")
    print(f"  accel model vs finite-difference truth : mean {err.mean():.4f}  max {err.max():.4f} m/s^2"
          f"   (signal |a| ~ {np.linalg.norm(a_true[lo:], axis=1).mean():.2f})")
    print(f"  CoM check (T/M) vs R^T(dv/dt+g)_z      : mean {err_com.mean():.4f}  max {err_com.max():.4f} m/s^2")
    print(f"  lever-arm term |w x (w x r)|           : {lev.mean():.2f} m/s^2  (thrust/m ~ {thrust[lo:].mean():.2f})")
    print(f"  flight stays bounded                   : z(1s)={z[a['n']//5]:.2f}  z(end)={z[-1]:.2f} m")
    print(f"  |w| = {mag.mean():.1f} rad/s ({np.rad2deg(mag.mean()):.0f} dps);"
          f"  gyro pinned on {100*sat.mean():.1f}% of steps;"
          f"  clipped-gyro error = {clip_err.mean():.2f} rad/s ({np.rad2deg(clip_err.mean()):.0f} dps)")

    assert err.mean() < 0.15, "IMU model disagrees with pybullet ground truth"
    assert err_com.mean() < 0.02, "thrust/M does not match the CoM specific force"

    # ---------- 2) what saturation does to the closed loop ------------------------------------
    rows_bad = rollout(flag=flag, steps=600, gyro_dps=gyro_dps, rate_source="measured")
    z_bad = np.array([r["z"] for r in rows_bad])
    print(f"\n=== same controller fed the *saturated* gyro ({gyro_dps:.0f} dps), 3 s ===")
    print(f"  z: 1s={z_bad[199]:6.2f}  2s={z_bad[399]:7.2f}  3s={z_bad[599]:8.2f} m   -> "
          f"{'LOST CONTROL' if z_bad[-1] < 0.5 else 'ok'}")
    print("OK")


if __name__ == "__main__":
    main()