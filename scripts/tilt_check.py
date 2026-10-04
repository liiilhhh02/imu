"""A/B: does `gpd_me.tilt.TiltObserver` actually bound the INS's drift in the closed loop?

Same fault draw for both arms, same estimator, same controller; the only difference is whether the
acceleration-derived thrust axis is blended into the INS before the controller consumes it.  Reports
the tilt error of the *thrust axis* (what the controller actually uses), since that is the quantity
the observer can fix; the heading is unobservable from this measurement by construction.

    $PY scripts/tilt_check.py --seeds 3 --steps 1500 --ckpt results/e2e_v12.pt
"""
import argparse
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME, os.path.join(ME, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pybullet as p  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from gpd_me.e2e import NetRate, deploy_obs, WINDOW  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.ins import AttitudeINS  # noqa: E402
from gpd_me.policy import PositionPID, load_policy, quat_to_matrix  # noqa: E402
from gpd_me.tilt import TiltObserver  # noqa: E402
from e4_closed_loop import DT, LEVER, identification  # noqa: E402

MASK = np.array([0.0, 0.0, 1.0, 1.0])


def fly(seed, steps, ckpt, dps, use_tilt, tau, prior_pool, rl_name=None):
    prior, diag, fault_rng, ic, pre = prior_pool
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MetaAviaryFaulty(
        drone_model=__import__("gym_pybullet_drones.utils.enums", fromlist=["DroneModel"]).DroneModel.CF2X,
        num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=__import__("gym_pybullet_drones.utils.enums", fromlist=["Physics"]).Physics("pyb"),
        aggregate_phy_steps=1, freq=200, gui=False, record=False, obstacles=False,
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=LEVER, seed=seed),
        rate_source="truth", att_source="truth")
    env.eval = False
    env.rate_override = np.zeros(3)
    env.reset()
    np.random.set_state(fault_rng)
    env.shut_down_rotors(3)
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], ic[0].tolist(), ic[1].tolist(),
                                      physicsClientId=env.CLIENT)
    p.resetBaseVelocity(env.DRONE_IDS[0], ic[2].tolist(), ic[3].tolist(), physicsClientId=env.CLIENT)
    env._updateAndStoreKinematicInformation()
    env.shut_down = MASK.copy()

    net = NetRate(ckpt, dev, prior, dps)
    net.preseed(pre["g"][-WINDOW:], pre["a"][-WINDOW:], pre["u"][-WINDOW:], pre["mask"][-WINDOW:])
    policy = load_policy(rl_name or "shutdown_real_7_4")
    pid = PositionPID()
    ins = AttitudeINS(np.eye(3))
    tob = TiltObserver(tau=tau) if use_tilt else None
    last = -np.ones(4)
    tp = np.array([0.0, 0.0, 1.0])
    tilt_err, hold = [], steps
    for t in range(steps):
        env._computeObs()
        gyro, accel = env.gyro_meas.copy(), env.accel_meas.copy()
        tom = float(env.thrust_over_mass)
        u_cmd = np.clip((last + 1.0) * 7.5, 0.0, 15.0)
        rate = net.step(accel, gyro, u_cmd, env.shut_down)
        ins.update(rate, DT)
        if tob is not None:
            ins.R = tob.correct(ins.R, env.vel[0], DT)
        R = ins.R
        ta, z_body = pid.step(DT, env.pos[0], Rotation.from_matrix(R).as_quat(), env.vel[0], tp)
        r_xy = float(np.hypot(z_body[0], z_body[1]))
        if r_xy > 0.26:
            s = 0.26 / r_xy
            z_body = np.array([z_body[0] * s, z_body[1] * s,
                               np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
        obs = deploy_obs(R.T @ z_body, rate, ta, tom, last, env.shut_down)
        act = policy.select_action(obs, deterministic=True)
        last = np.asarray(act, float).ravel()
        env.target_a, env.target_z_body = float(ta), z_body
        env.step(act)
        R_true = quat_to_matrix(env.quat[0])
        # the thrust axis is what the controller consumes; the heading is unobservable here
        c = float(np.clip(R[:, 2] @ R_true[:, 2], -1.0, 1.0))
        tilt_err.append(float(np.rad2deg(np.arccos(c))))
        if hold == steps and env.pos[0][2] < 0.3:
            hold = t
    used = tob.n_used if tob is not None else 0
    rej = tob.n_rejected if tob is not None else 0
    env.close()
    te = np.asarray(tilt_err)
    return dict(first=float(te[:300].mean()), all=float(te.mean()), p90=float(np.percentile(te, 90)),
                hold=hold, used=used, rejected=rej)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ckpt", default=os.path.join(ME, "results", "e2e_v12.pt"))
    ap.add_argument("--dps", type=float, default=1000.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--rl_name", default=None)
    a = ap.parse_args()
    print(f"{'seed':>4} {'arm':>10} {'tilt<1.5s':>10} {'tilt_mean':>10} {'tilt_p90':>9} "
          f"{'lost@':>6} {'used':>6} {'rej':>5}")
    for s in range(a.seeds):
        pool = identification(seed=s, dps=a.dps, flag=3)
        for arm, use in (("ins", False), ("ins+tilt", True)):
            r = fly(s, a.steps, a.ckpt, a.dps, use, a.tau, pool, a.rl_name)
            print(f"{s:4d} {arm:>10} {r['first']:10.2f} {r['all']:10.2f} {r['p90']:9.2f} "
                  f"{r['hold']:6d} {r['used']:6d} {r['rejected']:5d}", flush=True)


if __name__ == "__main__":
    main()