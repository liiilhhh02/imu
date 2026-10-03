"""Truth-rate hold-rate measurement for the adjacent-pair dual failure (task-1 denominator).

The closed-loop numbers in `scripts/e4_closed_loop.py` are a *distribution* over fault draws, because
this operating point is chaotic: the supervisor's own reference script (`repro_shutdown.py`, same
policy, same outer PID, `env._computeObs()`) holds ~2/3 of its draws.  Before any controller claim can
be made, the denominator has to be measured with a real sample: N independent draws of the environment's
own fault injection, same policy, same loop.

This script uses the env's **own** observation (`env._computeObs()`) and the **supervisor's** outer
position PID (`RLControl.RLShutDownControl`), i.e. it is the reference loop, not a re-implementation.

    held := final z > 0.3 m  and  std(z) < 3 m over the run      (same criterion as E4)

Run:
  PYTHONPATH=<repo> python scripts/study_truth_rate.py --draws 20 --freq 200
  PYTHONPATH=<repo> python scripts/study_truth_rate.py --draws 20 --freq 50 --rl_ckpt <name>
"""
import argparse
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pybullet as p  # noqa: E402

from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gym_pybullet_drones.envs.MetaShutDown7 import MetaAviary  # noqa: E402
from gpd_me.policy import load_policy  # noqa: E402

MASK = {0: [0, 1, 1, 1], 1: [0, 1, 0, 1], 2: [0, 0, 0, 1], 3: [0, 0, 1, 1]}
CKPT = {0: "shutdown_real_7", 1: "shutdown_real_7", 2: "shutdown_real_7_4", 3: "shutdown_real_7_4"}


def make_env(freq: int):
    e = MetaAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
                   physics=Physics("pyb"), aggregate_phy_steps=1, freq=freq, gui=False, record=False,
                   obstacles=False)
    e.eval = False
    e.reset()                      # applies the training-time M/KM/delay/inertia overrides
    return e


def one_draw(policy, flag: int, freq: int, steps: int, target=(0.0, 0.0, 1.0), corrupt=None):
    """One independent fault draw, flown by the reference loop. Returns per-run statistics."""
    env = make_env(freq)
    tp = np.array(target, dtype=float)
    env.shut_down_rotors(flag)             # the env's own IC: reset + mask + preset initial spin
    ctrl = RLControl(DroneModel.CF2X)
    ctrl.reset()
    zs, atts, rate_err = [], [], []
    for _ in range(steps):
        raw = env._getDroneStateVector(0)
        ta, tz = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                        cur_quat=raw[3:7], cur_vel=raw[10:13], target_pos=tp)
        r_xy = np.hypot(tz[0], tz[1])
        if r_xy > 0.26:                    # the notebook's tilt clamp
            s = 0.26 / r_xy
            tz = np.array([tz[0] * s, tz[1] * s, np.sqrt(1 - (tz[0] * s) ** 2 - (tz[1] * s) ** 2)])
        env.target_a, env.target_z_body = float(ta), tz
        obs = env._computeObs()            # the env's own 15-D construction -- what the policy was fed
        if corrupt is not None:
            obs = corrupt(obs, env)
        action = policy.select_action(obs, deterministic=True)
        env.step(action)
        zs.append(float(env.pos[0][2]))
        atts.append(float(np.rad2deg(env.att_rad_error)))
        # the base env's ang_vel IS the true body rate (this harness always feeds the truth to the
        # controller), so there is no rate error to report here -- record the spin magnitude instead
        rate_err.append(float(np.linalg.norm(np.asarray(env.ang_vel, float).ravel())))
    env.close()
    zs = np.array(zs)
    w = max(50, len(zs) // 5)
    return dict(z=float(zs[-w:].mean()), z_std=float(zs[-w:].std()), z_final=float(zs[-1]),
                att=float(np.mean(atts[-w:])), rate_err=float(np.mean(rate_err[-w:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", type=int, default=3)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--freq", type=int, default=200)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rl_ckpt", default=None, help="checkpoint name under gym_pybullet_drones/model")
    ap.add_argument("--rl_dir", default=os.path.join(REPO, "gym_pybullet_drones", "model"))
    ap.add_argument("--thresh_z", type=float, default=0.3)
    ap.add_argument("--thresh_std", type=float, default=3.0)
    a = ap.parse_args()

    name = a.rl_ckpt or CKPT[a.flag]
    policy = load_policy(name, a.rl_dir)
    print(f"policy={name}  flag={a.flag} mask={MASK[a.flag]}  freq={a.freq} Hz  "
          f"steps={a.steps}  draws={a.draws}  (reference loop: env obs + supervisor outer PID)")

    rows = []
    for i in range(a.draws):
        r = one_draw(policy, a.flag, a.freq, a.steps)
        held = (r["z_final"] > a.thresh_z) and (r["z_std"] < a.thresh_std)
        rows.append((held, r))
        print(f"  draw {i:2d}: held={int(held)}  z={r['z']:7.2f}  z_std={r['z_std']:6.2f}  "
              f"z_end={r['z_final']:8.2f}  att={r['att']:6.1f}deg")

    held = [h for h, _ in rows]
    z_ok = np.array([r["z"] for h, r in rows if h])
    att_ok = np.array([r["att"] for h, r in rows if h])
    print(f"\nHELD {sum(held)}/{len(held)}  ({100*sum(held)/len(held):.0f} %)   "
          f"z_med {np.median(z_ok) if len(z_ok) else float('nan'):.2f} m   "
          f"att_med {np.median(att_ok) if len(att_ok) else float('nan'):.1f} deg")
    print("held criterion: z_final > %.2f m and std(z) < %.2f m" % (a.thresh_z, a.thresh_std))


if __name__ == "__main__":
    main()