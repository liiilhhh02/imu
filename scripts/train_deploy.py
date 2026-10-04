#!/usr/bin/env python3
"""Fine-tune the senior's adjacent-pair failure policy on the *deployment* chain.

Why this file exists.  ``scripts/train_robust.py`` trained against a **synthetic** corruption model
(white noise + static bias + direction error + lag) and all four recipes failed: no anchor ->
catastrophic forgetting (nominal 4/4 -> 0/4); slow ramp -> keeps nominal but 0/4 robust; fast ramp ->
1/4 robust and loses nominal; eval-length episodes -> degraded within 80 episodes.  The estimator's
error is not white: it is low-frequency, axis-structured and correlated with the trajectory
(docs/STATUS.md 4.2), so the surrogate was the wrong object to train against.

Here the policy trains against the **real estimator**: the same checkpoint, the same ``NetRate``
wrapper, the same online prior identification (reused from ``scripts/e4_closed_loop.py``, not
reimplemented) and the same ``AttitudeINS`` integration, with the observation built by
``gpd_me.e2e.deploy_obs`` -- field for field the chain ``scripts/e4_closed_loop.py`` evaluates, so a
fine-tuned policy has nowhere to hide a train/eval mismatch.

What the policy may consume (and therefore what appears in its observation): the estimated body rate,
the estimated attitude, the measured accelerometer, the *commanded* rotor thrusts and the fault mask.
Never the true rate, attitude, lever arm or thrust.  Ground truth is used for the plant, for the
reward and for the anchor's rate source -- all of which are simulation-only privileges.

Usage (measured acceptance stays in e4_closed_loop.py, which already accepts --rl_dir/--rl_ckpt):
    $PY scripts/train_deploy.py --ckpt results/e2e_v12.pt --hours 3 --anchor 0.3
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
for _p in (REPO, ME, os.path.join(ME, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from gym_pybullet_drones.algo.ACRL import ACRL  # noqa: E402
from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.MetaBuffer import ReplayBuffer  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402

from gpd_me.e2e import NetRate, deploy_obs  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.ins import AttitudeINS  # noqa: E402
from gpd_me.policy import ACRLArgs, load_policy  # noqa: E402

from e4_closed_loop import DT, identification  # noqa: E402  (the deployment procedure, reused)

MASK3 = np.array([0.0, 0.0, 1.0, 1.0])


def log(msg, path):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def make_env(dps, seed, lever=(0.013, 0.004, 0.002)):
    """Training env: a faulty aviary whose IMU is the deployment IMU.

    ``rate_source``/``att_source`` are set to "truth" on purpose: this script builds the policy's
    observation itself with ``deploy_obs`` (the acceptance harness's function), so the env's own
    observation assembly must stay out of the way.  ``_computeObs`` is still called every step -- it
    is what advances the IMU with the one-sample-per-control-step guard.
    """
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200, gui=False, record=False,
        obstacles=False,
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=lever, seed=seed),
        rate_source="truth", att_source="truth")
    env.eval = False
    env.reset()
    return env


def draw_fault(env, seed, dps, prior_pool):
    """Apply one *real* fault draw: replay the RNG state captured by the identification procedure."""
    prior, diag, fault_rng, ic, _pre = prior_pool
    np.random.set_state(fault_rng)
    env.shut_down_rotors(3)
    import pybullet as p
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], ic[0].tolist(), ic[1].tolist(),
                                      physicsClientId=env.CLIENT)
    p.resetBaseVelocity(env.DRONE_IDS[0], ic[2].tolist(), ic[3].tolist(), physicsClientId=env.CLIENT)
    env._updateAndStoreKinematicInformation()
    return prior, diag


def episode(envs, nets, ins, pids, ctrls, policy, buffer, a, anchor_flags):
    """One parallel episode over all training envs, on the deployment chain.

    Faithful to ``e4_closed_loop.run``: _computeObs -> read the IMU -> the estimator -> the INS ->
    the supervisor's outer PID on the *estimated* attitude -> deploy_obs -> policy -> env.step.

    The transitions stored in the replay buffer use ``deploy_obs`` on **both** ends.  ``env.step``'s
    own returned observation is the senior's assembly, not this one: using it as ``next_obs`` would
    put a different convention into the critic's target and quietly make the fine-tune
    non-transferable -- the exact class of bug this file exists to avoid.
    """
    n = len(envs)
    last_action = [-np.ones(4) for _ in range(n)]
    prev_obs = [None] * n
    rew_sum = 0.0
    for i, e in enumerate(envs):
        ctrls[i].reset()
        pids[i].reset()
        ins[i] = AttitudeINS(np.eye(3))
        e.target_a, e.target_z_body = 9.81, np.array([0.0, 0.0, 1.0])
        e._computeObs()
    for _ in range(a.len_episode):
        obs_batch = np.zeros((n, 15), dtype=np.float32)
        for i, e in enumerate(envs):
            e._computeObs()                                   # advances the IMU (fresh sample)
            gyro, accel = e.gyro_meas.copy(), e.accel_meas.copy()
            tom = float(e.thrust_over_mass)
            u_cmd = np.clip((last_action[i] + 1.0) * 7.5, 0.0, 15.0)     # the command we sent
            if anchor_flags[i]:
                rate = e.omega_true.copy()                    # nominal anchor: the truth rate
            else:
                rate = nets[i].step(accel, gyro, u_cmd, e.shut_down)
            ins[i].update(rate, DT)
            R = ins[i].R
            raw = e._getDroneStateVector(0)                   # pos/vel only (see the module docstring)
            ta, z_body = ctrls[i].RLShutDownControl(
                control_timestep=e.TIMESTEP, cur_pos=raw[0:3],
                cur_quat=Rotation.from_matrix(R).as_quat(), cur_vel=raw[10:13],
                target_pos=np.array([0.0, 0.0, 1.0]))
            r_xy = float(np.hypot(z_body[0], z_body[1]))
            if r_xy > 0.26:                                   # the harness's tilt clamp
                s = 0.26 / r_xy
                z_body = np.array([z_body[0] * s, z_body[1] * s,
                                   np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
            obs_batch[i] = deploy_obs(R.T @ z_body, rate, ta, tom, last_action[i], e.shut_down)
            e.target_a, e.target_z_body = float(ta), z_body
        act = policy.select_action(obs_batch, deterministic=False)
        prev_act = [np.asarray(x, dtype=float).ravel() for x in act]   # t's action
        for i, e in enumerate(envs):
            _nxt_env_obs, rew, done, _ = e.step(act[i])       # discarded on purpose (see docstring)
            if prev_obs[i] is not None:
                buffer.add((prev_obs[i], prev_act[i], rew, obs_batch[i], float(done)))
            prev_obs[i] = obs_batch[i].copy()
            last_action[i] = np.asarray(act[i], dtype=float).ravel()
            rew_sum += float(np.mean(rew))
    zs = np.array([float(e.pos[0][2]) for e in envs])
    return dict(z=float(zs.mean()), rew=rew_sum / a.len_episode,
                anchors=int(sum(anchor_flags)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ME, "results", "e2e_v12.pt"),
                    help="the trained estimator that will fly (the real one, not a surrogate)")
    ap.add_argument("--policy_in", default="shutdown_real_7_4")
    ap.add_argument("--fw_dir", default=os.path.join(REPO, "gym_pybullet_drones", "model"))
    ap.add_argument("--out", default=os.path.join(ME, "results", "rl", "deploy_flag3"))
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--episodes", type=int, default=10 ** 9)
    ap.add_argument("--num_env", type=int, default=8)
    ap.add_argument("--len_episode", type=int, default=600)
    ap.add_argument("--dps", type=float, default=1000.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--anchor", type=float, default=0.3,
                    help="fraction of environments kept on the truth rate (anti-forgetting)")
    ap.add_argument("--setup_episodes", type=int, default=40,
                    help="episodes per (prior, fault-draw) pair before re-identifying")
    ap.add_argument("--pool", type=int, default=8,
                    help="identification runs to cycle through (the deployment procedure is slow)")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_draws", type=int, default=4)
    ap.add_argument("--eval_steps", type=int, default=1000)
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
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(vars(a), fh, indent=2)
    log(f"run={runname} estimator={a.ckpt} warm_start={a.policy_in} device={dev} "
        f"anchor={a.anchor} ep_len={a.len_episode} envs={a.num_env}", logp)

    # --- the deployment procedure's first half, once per pool slot -------------------------------
    t_ident = time.time()
    pool = [identification(seed=a.seed + 100 * k, dps=a.dps, flag=3) for k in range(a.pool)]
    # each pool entry is (prior, diag, fault_rng, ic, pre); `pre` is the measured pre-fault
    # history the estimator's window should start from (see NetRate.preseed)
    ks = [float(np.linalg.norm(p[0][:3])) for p in pool]
    log(f"identified {a.pool} draws in {time.time() - t_ident:.0f}s: k_med {np.median(ks):.3f}m "
        f"k_range [{min(ks):.3f},{max(ks):.3f}]", logp)

    policy = ACRL(state_dim=15, action_dim=4, max_action=1, device=dev, args=ACRLArgs())
    policy.load(a.policy_in, a.fw_dir)
    for opt in (policy.actor_optimizer, policy.critic_optimizer, policy.alpha_optimizer):
        for g in opt.param_groups:
            g["lr"] = a.lr
    policy.target_entropy = -2.0

    envs = [make_env(a.dps, a.seed + i) for i in range(a.num_env)]
    ctrls = [RLControl(DroneModel.CF2X) for _ in range(a.num_env)]
    nets = [NetRate(a.ckpt, dev, pool[i][0], a.dps) for i in range(a.num_env)]
    ins = [AttitudeINS(np.eye(3)) for _ in range(a.num_env)]
    from gpd_me.policy import PositionPID  # noqa: E402  (local import: keeps the header short)
    pids = [PositionPID() for _ in range(a.num_env)]   # never share one: the integral term
    buffer = ReplayBuffer(state_dim=15, action_dim=4, max_buffer_size=1_000_000)
    t0 = time.time()
    deadline = t0 + a.hours * 3600.0

    def evaluate(draws, steps, rate_mode):
        """In-loop check on fresh draws: hold rate with the estimator and with the truth rate."""
        held = 0
        for k in range(draws):
            env = make_env(a.dps, a.seed + 5000 + k)
            prior, diag = draw_fault(env, a.seed + 5000 + k, a.dps, pool[k % a.pool])
            env._computeObs()
            net = NetRate(a.ckpt, dev, prior, a.dps)
            i_ins = AttitudeINS(np.eye(3))
            ctrl = RLControl(DroneModel.CF2X)
            ctrl.reset()
            last = -np.ones(4)
            zs = []
            for _ in range(steps):
                env._computeObs()
                gyro, accel, tom = env.gyro_meas.copy(), env.accel_meas.copy(), float(env.thrust_over_mass)
                u_cmd = np.clip((last + 1.0) * 7.5, 0.0, 15.0)
                rate = env.omega_true.copy() if rate_mode == "truth" else net.step(accel, gyro, u_cmd,
                                                                                  env.shut_down)
                i_ins.update(rate, DT)
                R = i_ins.R
                raw = env._getDroneStateVector(0)
                ta, z_body = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                                    cur_quat=Rotation.from_matrix(R).as_quat(),
                                                    cur_vel=raw[10:13], target_pos=np.array([0., 0., 1.]))
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
                zs.append(float(env.pos[0][2]))
            env.close()
            zs = np.array(zs)
            w = max(1, len(zs) // 3)
            if zs[-w:].mean() > 0.3 and zs[-w:].std() < 3.0:
                held += 1
        return held, draws

    log("EVAL ep 0: " + "  ".join(
        f"{m} {evaluate(a.eval_draws, a.eval_steps, m)[0]}/{a.eval_draws}"
        for m in ("truth", "net")), logp)

    ep = 0
    while ep < a.episodes and time.time() < deadline:
        # one episode per (prior, draw) slot; re-draw the fault every episode, re-identify rarely
        anchor_flags = [bool(rng.random() < a.anchor) for _ in range(a.num_env)]
        for i, e in enumerate(envs):
            slot = (ep // a.setup_episodes + i) % a.pool
            prior, diag = draw_fault(e, a.seed + 100 * slot, a.dps, pool[slot])
            nets[i].prior = np.asarray(prior, float)
            nets[i].g_T = float(prior[3]); nets[i].tau = float(prior[7]); nets[i].k = float(prior[8])
            e._computeObs()
        st = episode(envs, nets, ins, pids, ctrls, policy, buffer, a, anchor_flags)
        for _ in range(4 if buffer.buffer_size >= a.batch else 0):
            policy.train(buffer, iterations=1)
        ep += 1
        el = time.time() - t0
        log(f"ep {ep:5d} z={st['z']:5.2f} rew={st['rew'] / a.len_episode:7.2f} "
            f"anchors={st['anchors']}/{a.num_env} buf={buffer.buffer_size} ep/s={ep/el:5.2f} "
            f"eta={str(timedelta(seconds=int(max(0, deadline - time.time()))))}", logp)
        if ep % a.eval_every == 0:
            res = {m: evaluate(a.eval_draws, a.eval_steps, m) for m in ("truth", "net")}
            log(f"EVAL ep {ep:5d}: truth {res['truth'][0]}/{res['truth'][1]}  "
                f"net {res['net'][0]}/{res['net'][1]}", logp)
            policy.save(f"{runname}_latest", os.path.join(a.out, "model"))
            policy.save(f"{runname}_ep{ep}", os.path.join(a.out, "model"))
    policy.save(f"{runname}_final", os.path.join(a.out, "model"))
    log(f"finished ep={ep} elapsed={timedelta(seconds=int(time.time() - t0))} -> {a.out}", logp)


if __name__ == "__main__":
    main()