"""Fine-tune the adjacent-pair failure policy so it tolerates the estimator's error.

Motivation (measured, `docs/STATUS.md` §5/§9): the supervisor's policy holds 19/20 independent fault
draws when it is fed the **true** rate, but the moment the rate is replaced by anything a real IMU
pipeline produces it dives -- with the true rate *and* the INS attitude it holds (z=2.9 m), while with a
*moderate* corruption (bias 1.5 rad/s, 6 % scale, 8 deg direction, 1 rad/s noise, 15 ms lag -- all
below the errors the v6 network actually has) it fails.  So the policy, not the estimator, is the
binding constraint on ">= 90 % of the truth-rate performance", and it must be trained against the error.

What this script does differently from `train_shutdown.py`
  * runs on **our** env (`gpd_me.env_faulty.MetaAviaryFaulty`) with `rate_source="corrupt"` and
    `att_source="ins"`: the controller input AND the attitude it is given both come from the corrupted
    (surrogate-estimator) rate -- no true rate, no true attitude reaches the policy;
  * a **curriculum** over the corruption level: each episode draws a level `lambda`, scaled by a ceiling
    that ramps from ~0.3 to 1.0, so the truth-rate skill is never destroyed;
  * evaluation reports a **hold-rate curve** over fault draws at several corruption levels, which is the
    quantity the acceptance criterion is about.

Honesty notes (do not remove):
  * the corruption is a *surrogate*; every number that is reported is re-measured with the **real
    network** in the loop (`scripts/e4_closed_loop.py`), never with this model;
  * the *reward* uses the simulator's true state (standard for RL training), but no true state enters
    the observation;
  * the outer position loop still uses the simulator's position/velocity (a real vehicle would use its
    position estimator) -- that assumption is stated, not hidden.

Run:
  PYTHONPATH=<repo>:<me> python scripts/train_robust.py --hours 3 --out results/rl/robust_flag3
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from gym_pybullet_drones.algo.ACRL import ACRL  # noqa: E402
from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.MetaBuffer import ReplayBuffer  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.policy import ACRLArgs  # noqa: E402

# Corruption at level 1.0 = the error the trained estimator actually shows (docs/STATUS.md §4/§9):
# per-flight bias ~1.3 rad/s, magnitude error ~6-15 % (the `k` identification), direction error up to
# ~20 deg when several axes saturate, jitter of a few rad/s, and a 10-30 ms estimator lag.
CORRUPT_FULL = dict(bias=4.5, scale=0.20, dir_deg=20.0, noise=4.0, lag_s=0.030)
MASK3 = [0, 0, 1, 1]


def log(msg, path):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def make_env(freq, dps, level, seed, fault=True):
    cfg = {k: v * float(level) for k, v in CORRUPT_FULL.items()}
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=freq, gui=False, record=False,
        obstacles=False,
        rate_source="truth" if level <= 0.0 else "corrupt",
        att_source="ins",
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=(0.013, 0.004, 0.002), seed=seed),
        corrupt_cfg=(cfg if level > 0.0 else None))
    env.eval = False
    env.reset()
    if fault:
        env.shut_down_rotors(3)
    return env


def rollout(env, policy, ctrl, steps, deterministic, target=(0.0, 0.0, 1.0), rng=None,
            level=None, collect=None):
    """One episode: the supervisor's outer PID sets (target thrust, target body-z), the policy the rest.

    Returns the episode's statistics; `collect` (if given) is a ReplayBuffer that gets (obs, action,
    reward, next_obs, done) tuples -- the reward is the environment's own.
    """
    tp = np.array(target, dtype=float)
    ctrl.reset()
    zs, atts = [], []
    obs = env._computeObs()
    prev = obs
    rew_sum = 0.0
    for _ in range(steps):
        raw = env._getDroneStateVector(0)
        ta, tz = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                        cur_quat=raw[3:7], cur_vel=raw[10:13], target_pos=tp)
        r_xy = float(np.hypot(tz[0], tz[1]))
        if r_xy > 0.26:
            s = 0.26 / r_xy
            tz = np.array([tz[0] * s, tz[1] * s, np.sqrt(1 - (tz[0] * s) ** 2 - (tz[1] * s) ** 2)])
        env.target_a, env.target_z_body = float(ta), tz
        obs = env._computeObs()
        action = policy.select_action(obs, deterministic=deterministic)
        nxt, rew, done, _info = env.step(action)
        if collect is not None:
            collect.add((prev, action, rew, nxt, float(done)))
        prev = nxt
        rew_sum += float(np.mean(rew))
        zs.append(float(env.pos[0][2]))
        atts.append(float(np.rad2deg(env.att_rad_error)))
    zs = np.asarray(zs)
    atts = np.asarray(atts)
    w = max(50, len(zs) // 5)
    return dict(z=float(zs[-w:].mean()), z_std=float(zs[-w:].std()), z_final=float(zs[-1]),
                att=float(atts[-w:].mean()), reward=rew_sum / max(steps, 1))


def held(r, thresh_z=0.3, thresh_std=3.0):
    return bool(r["z_final"] > thresh_z and r["z_std"] < thresh_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_in", default="shutdown_real_7_4")
    ap.add_argument("--fw_dir", default=os.path.join(REPO, "gym_pybullet_drones", "model"))
    ap.add_argument("--out", default=os.path.join(ME, "results", "rl", "robust_flag3"))
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--episodes", type=int, default=10 ** 9)
    ap.add_argument("--num_env", type=int, default=4)
    ap.add_argument("--train_freq", type=int, default=50)
    ap.add_argument("--len_episode", type=int, default=250)
    ap.add_argument("--dps", type=float, default=1000.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lam_max0", type=float, default=0.3, help="corruption ceiling for episode 0")
    ap.add_argument("--lam_ramp", type=int, default=1500, help="episodes to reach ceiling 1.0")
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_draws", type=int, default=4)
    ap.add_argument("--eval_steps", type=int, default=1200)
    ap.add_argument("--eval_levels", default="0,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "model"), exist_ok=True)
    runname = os.path.basename(a.out.rstrip("/"))
    logp = os.path.join(a.out, "train.log")
    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"run={runname} warm_start={a.policy_in} device={dev} train_freq={a.train_freq} "
        f"dps={a.dps} lam_max0={a.lam_max0} ramp={a.lam_ramp} ep_len={a.len_episode}",
        logp)
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(vars(a), fh, indent=2)

    policy = ACRL(state_dim=15, action_dim=4, max_action=1, device=dev, args=ACRLArgs())
    policy.load(a.policy_in, a.fw_dir)
    log(f"warm start from {a.fw_dir}/{a.policy_in}", logp)
    for opt in (policy.actor_optimizer, policy.critic_optimizer, policy.alpha_optimizer):
        for g in opt.param_groups:
            g["lr"] = a.lr
    policy.target_entropy = -2.0

    envs = [make_env(a.train_freq, a.dps, 1e-9, a.seed + i) for i in range(a.num_env)]
    ctrls = [RLControl(DroneModel.CF2X) for _ in range(a.num_env)]
    ev_env = make_env(200, a.dps, 0.0, 999)
    ev_ctrl = RLControl(DroneModel.CF2X)
    buffer = ReplayBuffer(state_dim=15, action_dim=4, max_buffer_size=1_000_000)
    t0 = time.time()
    deadline = t0 + a.hours * 3600.0

    def evaluate(levels, draws):
        out = {}
        for lam in levels:
            res, env0 = [], None
            for k in range(draws):
                env = make_env(200, a.dps, lam, a.seed + 1000 + k)
                r = rollout(env, policy, ev_ctrl, a.eval_steps, True)
                env.close()
                res.append(r)
            out[lam] = (sum(held(r) for r in res), draws, res)
        return out

    ev = evaluate([float(x) for x in a.eval_levels.split(",")], a.eval_draws)
    for lam, (h, n, res) in sorted(ev.items()):
        log(f"EVAL ep     0 lam={lam:.2f}: held {h}/{n}  z_med "
            f"{np.median([r['z'] for r in res]):.2f}  att_med {np.median([r['att'] for r in res]):.1f}",
            logp)

    ep = 0
    while ep < a.episodes and time.time() < deadline:
        lam_ceiling = min(1.0, a.lam_max0 + (1.0 - a.lam_max0) * ep / max(a.lam_ramp, 1))
        for i, e in enumerate(envs):
            lam = float(rng.uniform(0.0, lam_ceiling))
            # scale the *attributes* (not the object): `shut_down_rotors` -> `reset()` re-draws the
            # corruptor's bias/scale/direction from these attributes, so one episode = one fresh draw
            # at this level.  Reusing the environments keeps the pybullet client alive.
            for k, v in CORRUPT_FULL.items():
                setattr(e.corruptor, k, v * lam)
            e.RATE_SOURCE = "truth" if lam <= 0.0 else "corrupt"
            e.shut_down_rotors(3)
            ctrls[i].reset()
        obs_prev = [e._computeObs() for e in envs]
        ep_rew, ep_att = 0.0, 0.0
        for _ in range(a.len_episode):
            obs_batch = np.stack([e._computeObs() for e in envs])
            act = policy.select_action(obs_batch, deterministic=False)
            for i, e in enumerate(envs):
                raw = e._getDroneStateVector(0)
                ta, tz = ctrls[i].RLShutDownControl(control_timestep=e.TIMESTEP, cur_pos=raw[0:3],
                                                    cur_quat=raw[3:7], cur_vel=raw[10:13],
                                                    target_pos=np.array([0.0, 0.0, 1.0]))
                r_xy = float(np.hypot(tz[0], tz[1]))
                if r_xy > 0.26:
                    s = 0.26 / r_xy
                    tz = np.array([tz[0] * s, tz[1] * s,
                                   np.sqrt(1 - (tz[0] * s) ** 2 - (tz[1] * s) ** 2)])
                e.target_a, e.target_z_body = float(ta), tz
                obs_batch[i] = e._computeObs()
                nxt, rew, done, _ = e.step(act[i])
                buffer.add((obs_prev[i], act[i], rew, nxt, float(done)))
                obs_prev[i] = nxt
                ep_rew += float(np.mean(rew))
                ep_att += float(np.rad2deg(e.att_rad_error))
            if buffer.buffer_size >= a.batch:
                policy.train(buffer, iterations=1)
        ep += 1
        el = time.time() - t0
        log(f"ep {ep:5d} lam<={lam_ceiling:.2f} rew={ep_rew/a.len_episode:7.2f} "
            f"att={ep_att/a.len_episode:5.1f}deg buf={buffer.buffer_size} ep/s={ep/el:5.2f} "
            f"eta={str(timedelta(seconds=int(max(0, deadline-time.time()))))}", logp)
        if ep % a.eval_every == 0:
            ev = evaluate([float(x) for x in a.eval_levels.split(",")], a.eval_draws)
            for lam, (h, n, res) in sorted(ev.items()):
                log(f"EVAL ep {ep:5d} lam={lam:.2f}: held {h}/{n}  z_med "
                    f"{np.median([r['z'] for r in res]):.2f}  "
                    f"att_med {np.median([r['att'] for r in res]):.1f}", logp)
            policy.save(f"{runname}_latest", os.path.join(a.out, "model"))
            policy.save(f"{runname}_ep{ep}", os.path.join(a.out, "model"))
    policy.save(f"{runname}_final", os.path.join(a.out, "model"))
    log(f"finished ep={ep} elapsed={timedelta(seconds=int(time.time()-t0))} -> {a.out}", logp)


if __name__ == "__main__":
    main()