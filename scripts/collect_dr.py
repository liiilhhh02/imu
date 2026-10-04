"""Domain-randomised data collection for the end-to-end estimator (2 rotors or fewer failed).

Per episode it randomises the airframe, the sensor, the fault and the excitation, flies in PyBullet,
logs the *raw* signals plus ground truth, and identifies the priors from the same log with the
deployment procedure (`gpd_me.priors`).  Shards are written as npz so collection (CPU, parallel) and
training (GPU) are decoupled.

Randomisation (per episode)
  airframe : lever arm r (3-35 mm, any direction), M, KM, delay, J (3 axes), angular damping
  sensor   : gyro range 200-4000 dps, gyro bias/scale, gyro/accel noise, sample rate
  fault    : which rotor(s) fail (1 or 2, 3-failure excluded by decision), fault onset time,
             initial spin and attitude, abrupt vs gradual
  flight   : excitation mode -- random command excitation / the senior's RL policy (truth rate) /
             the senior's INDI baseline; plus aggressive and gentle target trajectories

Run:  PYTHONPATH=<repo>:<me> python scripts/collect_dr.py --episodes 64 --workers 12 --steps 2000
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MASKS = {  # 1 failure and 2 failures only (3-failure case excluded by decision)
    0: [0, 1, 1, 1],
    1: [0, 1, 0, 1],
    3: [0, 0, 1, 1],
}
CKPT = {0: "shutdown_real_7", 3: "shutdown_real_7_4"}
FREQS = (100, 200, 250, 400)


def _excite_random(rng, n, mask, steps):
    """Random-walk thrust commands + per-rotor sinusoids: covers the input space, no policy needed."""
    base = rng.uniform(1.0, 4.0)
    u = np.zeros((steps, 4))
    walk = rng.normal(0.0, 0.05, 4)
    freqs = rng.uniform(0.2, 8.0, 4)
    phases = rng.uniform(0, 2 * np.pi, 4)
    amps = rng.uniform(0.0, 3.0, 4)
    t = np.arange(steps) / n
    for i in range(steps):
        walk = np.clip(walk * 0.995 + rng.normal(0, 0.03, 4), -2.0, 2.0)
        u[i] = np.clip(base + walk + amps * np.sin(2 * np.pi * freqs * t[i] + phases), 0.0, 12.0)
        if rng.random() < 0.002:                      # occasional large step (aggressive manoeuvre)
            u[i] = rng.uniform(0.0, 12.0, 4)
    return u * np.array(mask, float)


def one_episode(job):
    """Isolates per-episode failures so one bad rollout cannot kill the pool."""
    try:
        return _one_episode_impl(job)
    except Exception as e:
        import traceback
        print(f"[worker-fail] ep{job[0]}: {type(e).__name__}: {e}")
        print(traceback.format_exc().splitlines()[-3:])
        return None


def _one_episode_impl(job):
    idx, seed, steps, out, pin_dps = job
    rng = np.random.default_rng(seed)
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gpd_me.env_faulty import MetaAviaryFaulty
    from gpd_me.imu import IMUConfig
    from gpd_me.priors import identify_priors
    from gpd_me.policy import PositionPID, load_policy, quat_to_matrix
    from scipy.spatial.transform import Rotation

    flag = int(rng.choice(list(MASKS.keys())))
    mask = np.array(MASKS[flag], float)
    dps = float(pin_dps) if pin_dps > 0 else float(rng.choice(
        [100.0, 150.0, 200.0, 300.0, 400.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0]))
    # PIN_DPS > 0 collects a single gyro range instead of the 12-range mixture.  Use it to build a
    # dps=2000 dataset (the professor's operating point): at 2000 dps the 43 rad/s fault spin still
    # saturates the yaw axis (34.9 rad/s limit) but NOT the tilt axes (24-26 rad/s), so the algebraic
    # front end is no longer fighting three saturated axes at once.
    freq = int(rng.choice(FREQS)); dt = 1.0 / freq
    lever = rng.normal(size=3); lever /= np.linalg.norm(lever); lever *= rng.uniform(0.002, 0.045)
    M, KM, delay = float(rng.uniform(0.5, 1.6)), float(rng.uniform(0.004, 0.022)), float(rng.uniform(0.008, 0.055))
    J = np.array([rng.uniform(0.0015, 0.014), rng.uniform(0.0015, 0.014), rng.uniform(0.002, 0.020)])
    damping = float(rng.uniform(0.003, 0.10))

    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=freq, gui=False, record=False,
        obstacles=False, rate_source="truth",
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=tuple(lever),
                          gyro_noise_std=float(rng.uniform(0.01, 0.10)),
                          accel_noise_std=float(rng.uniform(0.005, 0.05)),
                          gyro_bias_std=float(rng.uniform(0.0, 0.04)),
                          gyro_scale_std=float(rng.uniform(0.0, 0.03)),
                          seed=int(seed % (2 ** 31))))     # else every episode draws the same bias/noise
    env.eval = False
    env.reset()
    import pybullet as p
    # NOTE: MetaShutDown7.shut_down_rotors() calls reset(), and reset() writes its own
    # M/KM/delay/inertia/damping back.  The randomised plant must therefore be applied *after* it,
    # otherwise the airframe randomisation is silently inert (it was, until this was found).
    env.shut_down_rotors(flag)
    # `shut_down_rotors` does reset() + mask + (flags 2/3) an *immediate* initial yaw spin.  That spin
    # must NOT be present during the nominal identification phase: it saturates the gyro for most of
    # that phase at low/mid range (measured on dr_1e7_v2: 98 %/72 %/38 %/13 % of nominal frames have a
    # saturated axis at 100/300/1000/1500 dps for flag 3), which silently poisons every prior that is
    # identified from it -- the nominal segment is supposed to be a clean hover + yaw manoeuvre.
    # So: record the env's own spin, clear it for the nominal phase, and re-apply it at the fault.
    spin_env = np.asarray(env.ang_vel, float).ravel().copy()
    env.shut_down = np.ones(4)               # healthy during the nominal phase
    p.resetBaseVelocity(objectUniqueId=env.DRONE_IDS[0], linearVelocity=[0, 0, 0],
                        angularVelocity=[0, 0, 0], physicsClientId=env.CLIENT)
    env._updateAndStoreKinematicInformation()
    p.changeDynamics(env.DRONE_IDS[0], -1, mass=M, localInertiaDiagonal=J.tolist(),
                     angularDamping=damping, physicsClientId=env.CLIENT)
    env.M, env.KM, env.delay = M, KM, delay
    M_applied = float(p.getDynamicsInfo(env.DRONE_IDS[0], -1, physicsClientId=env.CLIENT)[0])
    # the fault-time spin is the environment's own draw (kept from `shut_down_rotors` above), so the
    # training distribution matches exactly what the supervisor's fault injection produces
    spin_at_fault = spin_env if flag in (1, 3) else np.zeros(3)

    # --- episode structure: a NOMINAL phase first (hover + a deliberate yaw-identification
    # manoeuvre, which is what makes every prior identifiable from flight data), then the fault ---
    t_fault = int(rng.integers(int(0.12 * steps), int(0.6 * steps)))
    mode = str(rng.choice(["random", "random", "policy", "random"]))
    policy = pid = None
    if mode == "policy" and flag in CKPT:
        policy = load_policy(CKPT[flag])
        pid = PositionPID()
    u_plan = _excite_random(rng, freq, mask, steps) if mode == "random" else None

    L = {k: [] for k in ("a", "g", "u", "w", "q", "v", "tom", "sat", "mask")}
    # nominal yaw-identification excitation: a differential that spins the body at 2-12 rad/s
    # (inside every gyro range we sample), so |s| = k|w|^2 carries a usable lever-arm signal
    # two-stage identification manoeuvre, OPEN LOOP (a closed loop would bias the ARX):
    #   stage A (0.2 s) apply a known differential and measure the yaw-rate response -> gain estimate
    #   stage B apply an open-loop profile whose amplitude targets ~30 % of the gyro range
    yid_f = rng.uniform(0.3, 2.0)
    yid_target = 0.55 * np.deg2rad(dps)      # was 0.30*range: at 100-300 dps that is 0.5-1.6 rad/s,
    #                                          where |s| = k|w|^2 is at the accelerometer noise level
    yid_d0 = rng.uniform(0.3, 1.0)
    yid_tA = int(0.2 / dt)
    yid_amp, yid_w0, yid_t0 = None, None, None
    hover_u = M * 9.81 / 4.0
    for i in range(steps):
        if i == t_fault:                      # inject the failure (and the resulting spin)
            env.shut_down = mask
            if any(spin_at_fault):
                p.resetBaseVelocity(objectUniqueId=env.DRONE_IDS[0], linearVelocity=[0, 0, 0],
                                    angularVelocity=list(spin_at_fault), physicsClientId=env.CLIENT)
                env._updateAndStoreKinematicInformation()
        env._computeObs()
        q = env.quat[0]
        if i < t_fault:                        # nominal: hover + yaw identification
            z = float(env.pos[0][2])
            base = hover_u + 2.0 * (1.0 - z)
            if i == yid_tA:                                  # stage A finished -> read the yaw response
                yid_w0 = float(env.gyro_meas[2])
            if i < yid_tA:
                yaw_exc = yid_d0
            else:
                if yid_amp is None:                      # stage A done -> open-loop amplitude
                    dtA = max(i - yid_tA, 1) * dt
                    gain = abs(float(env.gyro_meas[2]) - yid_w0) / max(abs(yid_d0) * dtA, 1e-6)
                    yid_amp = float(np.clip(yid_target / max(gain, 1e-3), 0.05, 3.0))
                yaw_exc = yid_amp * np.sin(2 * np.pi * yid_f * (i - yid_tA) * dt)
            u_cmd = np.clip(np.array([base - yaw_exc, base + yaw_exc,
                                      base - yaw_exc, base + yaw_exc]), 0.0, 15.0)
            L["a"].append(env.accel_meas.copy()); L["g"].append(env.gyro_meas.copy())
            L["u"].append(u_cmd); L["w"].append(env.omega_true.copy())
            L["q"].append(np.asarray(q, float).copy()); L["v"].append(env.vel[0].copy())
            L["tom"].append(np.asarray(env.thrust[0], float).sum() / M)
            L["sat"].append(np.abs(env.gyro_meas) >= np.deg2rad(dps) - 1e-9)
            L["mask"].append(np.asarray(env.shut_down, float).copy())
            env.target_a, env.target_z_body = 9.81, np.array([0.0, 0.0, 1.0])
            env.step(u_cmd / 7.5 - 1.0)
            continue
        if mode == "random":
            u_cmd = u_plan[i]
        elif mode == "policy" and policy is not None:
            qa = Rotation.from_quat(q).as_quat()
            ta, z_body = pid.step(dt, env.pos[0], qa, env.vel[0], np.array([0.0, 0.0, 1.0]))
            r_xy = np.hypot(z_body[0], z_body[1])
            if r_xy > 0.26:
                sc = 0.26 / r_xy
                z_body = np.array([z_body[0] * sc, z_body[1] * sc,
                                   np.sqrt(1 - (z_body[0] * sc) ** 2 - (z_body[1] * sc) ** 2)])
            R_att = quat_to_matrix(q)
            rel = R_att.T @ z_body
            obs = np.array([rel[0], rel[1], env.omega_true[0] / 10, env.omega_true[1] / 10,
                            env.gyro_meas[2] / 50, (ta - 9.8) / 3,
                            (env.thrust_over_mass - 9.8) / 3,
                            *(env.last_action[0] * mask), *(mask * 2 - 1)], dtype=np.float32)
            action = policy.select_action(obs, deterministic=True)
            u_cmd = np.clip((action + 1) * 7.5, 0, 15) * mask
        else:
            u_cmd = _excite_random(rng, freq, mask, 1)[0]

        L["a"].append(env.accel_meas.copy()); L["g"].append(env.gyro_meas.copy())
        L["u"].append(np.asarray(u_cmd, float)); L["w"].append(env.omega_true.copy())
        L["q"].append(np.asarray(q, float).copy()); L["v"].append(env.vel[0].copy())
        L["tom"].append(np.asarray(env.thrust[0], float).sum() / M)
        L["sat"].append(np.abs(env.gyro_meas) >= np.deg2rad(dps) - 1e-9)
        L["mask"].append(np.asarray(env.shut_down, float).copy())

        env.target_a, env.target_z_body = 9.81, np.array([0.0, 0.0, 1.0])
        env.step(np.array(u_cmd, float) / 7.5 - 1.0)       # inverse of the env's action->thrust map
    d = {k: np.array(v) for k, v in L.items()}
    nmin = min(len(v) for v in d.values())          # guard: every per-step list must be in step
    if any(len(v) != nmin for v in d.values()):
        print(f"[warn] length mismatch in episode {idx}: " +
              str({k: len(v) for k, v in d.items() if len(v) != nmin}))
        d = {k: v[:nmin] for k, v in d.items()}
    from gpd_me.priors import identify_priors as _ip
    prior, diag = _ip(d["g"], d["a"], d["u"], d["w"], d["mask"], dps, dt, vel=d["v"],
                                 tom=d["tom"])
    # validation of the identified priors (privileged checks, diagnostics only)
    diag["mass_est"] = float(1.0 / prior[3])                     # command is in N, so g_T = 1/M
    diag["mass_err_pct"] = float(100.0 * (1.0 / prior[3] - M) / M)
    inr = ~np.asarray(d["sat"], bool).any(axis=1)
    if diag.get("G") and np.isfinite(diag["G"]) and inr.sum() > 60:
        x = (np.array([1, -1, 1, -1.0])[None, :] * d["u"]).sum(1)
        pred = d["w"][:, 2] + dt * (diag["G"] * x - d["w"][:, 2]) / max(diag["T"], 1e-3)
        err = pred[:-1] - d["w"][1:, 2]                       # one-step prediction error
        m = inr[:-1] & inr[1:]
        diag["yaw_pred_rmse"] = float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")
    diag["nominal_frac"] = float(t_fault / steps)
    env.close()
    payload = dict(gyro=d["g"].astype(np.float32), accel=d["a"].astype(np.float32),
                   mask_t=d["mask"].astype(np.float32),
                   u=d["u"].astype(np.float32), tom=d["tom"].astype(np.float32),
                   omega=d["w"].astype(np.float32), quat=d["q"].astype(np.float32),
                   sat=d["sat"], mask=mask, dps=dps, dt=dt, prior=prior,
                   true=dict(lever=lever, M=M_applied, KM=KM, delay=delay, J=J,
                             damping=damping, M_requested=M),
                   diag=diag, mode=mode, flag=flag)
    fn = os.path.join(out, f"ep{idx:05d}.npz")
    np.savez_compressed(fn, **{k: v for k, v in payload.items() if k != "true" and k != "diag"},
                        diag_keys=np.array(list(diag.keys()), dtype=object),
                        diag_vals=np.array([float(diag[k]) if isinstance(diag[k], (int, float, np.floating)) else np.nan for k in diag], dtype=float),
                        true=np.array([list(payload["true"].values())], dtype=object),
                        meta=np.array([flag, dps, mode], dtype=object))
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(ME, "results", "dr_shards"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dps", type=float, default=0.0,
                    help="pin the gyro range (0 = the 12-range mixture); e.g. --dps 2000")
    a = ap.parse_args()
    global PIN_DPS
    PIN_DPS = float(a.dps)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, a.seed * 100000 + i, a.steps, a.out, float(a.dps)) for i in range(a.episodes)]
    t0 = time.time()
    print(f"collecting {a.episodes} episodes x {a.steps} steps with {a.workers} workers "
          f"({a.episodes*a.steps} steps total) -> {a.out}")
    with Pool(a.workers) as pool:
        for n, fn in enumerate(pool.imap_unordered(one_episode, jobs, chunksize=1), 1):
            if n % 8 == 0 or n == a.episodes:
                el = time.time() - t0
                print(f"  {n}/{a.episodes} episodes  {n*a.steps} steps  "
                      f"{el:5.0f}s  ({n*a.steps/el:6.0f} steps/s)")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()