"""E4 — closed-loop main experiment with the learned estimator in the loop.

Episode structure (identical to the data collector, so the deployment procedure is exercised):
    1. nominal phase: hover + the two-stage OPEN-LOOP yaw-identification manoeuvre
    2. fault injection (adjacent-pair dual failure by default) + `POST_FAULT_S` of further
       OPEN-LOOP excitation
       -> every prior the estimator needs is identified from this flight data only (gyro,
          accelerometer, commanded thrust, per-sample alive mask, velocity).  The post-fault window
          is part of the identification on purpose: it is where the body spins fast enough to
          saturate the gyro, so the lever arm's `|r_perp|` becomes observable (`id_lever_arm`'s
          saturated scan / the in-range LS both need it).  The nominal hover alone never saturates
          at >=1000 dps and `k` then degrades to `‖r_LS‖`.  Open loop, not the policy: a closed loop
          would correlate the yaw command with the yaw rate and bias `id_yaw_channel`'s ARX.
    3. closed loop: outer position PID -> ACRL policy -> thrusts, with the attitude expressed by an
       INS and the controller's rate input taken from one of

           truth     ground truth                        (upper bound)
           clipped   the raw saturated gyro              (what a real IMU gives)
           net       the trained end-to-end estimator    (the proposal)

       For `net`, w_hat drives **both** the attitude INS and the controller rate input — the earlier
       matrix showed that injecting it in only one place is actively harmful.

IMPORTANT — read the results as a distribution over initial spins, not as a pass/fail table.
The adjacent-pair case (flag 3, mask [0,0,1,1]) with checkpoint `shutdown_real_7_4` sits **on the
stability boundary**: from a 20-draw sweep of the env's own `shut_down_rotors(3)` injection, ~15 %
of the spins diverge even when the controller is handed the TRUE body rate.  It is genuinely
chaotic, not a clean basin — perturbing the injected spin by 0.002 rad/s flips the outcome, and
feeding the TRUE attitude instead of the INS (or integrating the INS trapezoidally) does not
rescue the escapes (17/20 and 18/20 respectively).  The loop itself is byte-identical to the
repo's reference `scripts/verify_ins_attitude.run(stack="rl", att="ins", src="truth")`: given the
same post-fault state they produce identical trajectories.  Hand-picking seeds that survive would
be cherry-picking, so E4 reports the escape rate instead.

Metrics (per `src`, paired across srcs on the same N fault draws): the number of draws that held
(z > 0.3 m and std(z) < 3 m over the run), the median and IQR of the final altitude over the
surviving draws, the mean |tilt| error of the attitude estimate over the run, and the mean
per-flight rate error (|rate - omega_true|) over the run.  The `net` row is invalid until the
estimator is retrained on the current 28-feature / 9-prior layout: `results/e2e_v5.pt` predates it,
so the checkpoint cannot be loaded and the row falls back to the algebraic front end.

Run:  PYTHONPATH=<repo>:<me> python scripts/e4_closed_loop.py --ckpt results/e2e_v5.pt --flag 3
      --ranges 400,1000,4000 --seeds 3 --steps 2000
      (defaults: --ranges 1000 --seeds 20, i.e. the distributional report)
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
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.e2e import (WINDOW, E2ENet, algebraic_estimate, coarse_summary, fine_features,  # noqa: E402
                        deploy_obs)
from gpd_me.e2e import NetRate  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.ins import AttitudeINS  # noqa: E402
from gpd_me.policy import PositionPID, load_policy, quat_to_matrix  # noqa: E402
from gpd_me.priors import identify_priors, tom_from_command  # noqa: E402
from gpd_me.tilt import TiltObserver  # noqa: E402

MASK = {0: [0, 1, 1, 1], 3: [0, 0, 1, 1]}
CKPT_RL = {0: "shutdown_real_7", 3: "shutdown_real_7_4"}
# `--rl_ckpt`/`--rl_dir` override the supervisor's checkpoint, so the *robustly trained* policy
# (e.g. results/rl/robust_v2/model/robust_v2_latest) can be flown through the very same paired
# distributional harness -- that is how the ">= 90 % of the truth-rate performance" comparison is run.
RL_NAME = None
RL_DIR = None
# Tolerance probe: feed the TRUE rate plus a controlled perturbation.  This is not an estimator -- it
# measures how good a rate source must be for this operating point, i.e. it converts "the network is
# not good enough" into a number the next iteration can be judged against.
PN_BIAS = 0.0
PN_SIGMA = 0.0
PN_RNG = np.random.default_rng(0)
# Causal one-pole low-pass on the NETWORK estimate (fc in Hz, 0 = off).  The proper 480 ms spectrum
# (docs/STATUS.md 4.2) shows 54.5 % of the network error power above 10 Hz and 22.2 % above 20 Hz,
# while the true rate has only 1.8 % above 20 Hz -- i.e. a large part of the error IS filterable.
# The filter state is seeded from the first sample of each flight (a local in `run`, never
# carried over), so there is no start-up transient and no leakage between flights.
NET_LP = 0.0
DT = 1.0 / 200.0
POST_FAULT_S = 1.5       # open-loop logging appended after the fault for identification (seconds)
#                          (this is where the gyro saturates, so the lever arm's |r_perp| becomes
#                          identifiable; the healthy hover alone never saturates at >=1000 dps)
_WARNED = set()          # checkpoints already reported as incompatible (loud once, not per run)


def nominal_phase(env, steps=400, dps=1000.0, seed=0):
    """Hover + two-stage open-loop yaw identification; returns the logs needed for the priors."""
    rng = np.random.default_rng(seed)
    dt = DT
    yid_f = float(rng.uniform(0.3, 2.0))
    # 0.55 x range, matching scripts/collect_dr.py: the closed-loop harness used 0.30 x, which is a
    # weaker excitation than the identification the training and the dataset assume -- and the
    # measured effect was an online k_hat of 0.65-0.90 cm against a true |r_perp| of ~1.3 cm
    # (about 50 % low), i.e. the estimator was handicapped inside the loop.
    yid_target = 0.55 * np.deg2rad(dps)
    yid_d0 = float(rng.uniform(0.3, 1.0)); yid_tA = int(0.2 / dt)
    yid_amp, yid_w0 = None, None
    hover_u = float(env.M) * 9.81 / 4.0
    L = {k: [] for k in ("a", "g", "u", "w", "v", "tom", "mask")}
    for i in range(steps):
        env._computeObs()
        z = float(env.pos[0][2])
        base = hover_u + 2.0 * (1.0 - z)
        if i == yid_tA:
            yid_w0 = float(env.gyro_meas[2])      # measured gyro (unsaturated in the nominal phase)
        if i < yid_tA:
            exc = yid_d0
        else:
            if yid_amp is None:
                dtA = max(i - yid_tA, 1) * dt
                gain = abs(float(env.gyro_meas[2]) - yid_w0) / max(abs(yid_d0) * dtA, 1e-6)
                yid_amp = float(np.clip(yid_target / max(gain, 1e-3), 0.05, 3.0))
            exc = yid_amp * np.sin(2 * np.pi * yid_f * (i - yid_tA) * dt)
        u_cmd = np.clip(np.array([base - exc, base + exc, base - exc, base + exc]), 0.0, 15.0)
        L["a"].append(env.accel_meas.copy()); L["g"].append(env.gyro_meas.copy())
        L["u"].append(u_cmd); L["w"].append(env.omega_true.copy())
        L["v"].append(env.vel[0].copy()); L["tom"].append(float(np.atleast_1d(env.last_acc)[0]))
        L["mask"].append(np.ones(4))
        env.target_a, env.target_z_body = 9.81, np.array([0.0, 0.0, 1.0])
        env.step(u_cmd / 7.5 - 1.0)
    d = {k: np.array(v) for k, v in L.items()}
    return d


# The IMU's lever arm, i.e. the vehicle the identification measures against.  Every consumer
# (training included) must use THIS one, or the identified k = |r_perp| describes a
# different vehicle (the opus audit caught exactly that mismatch).
LEVER = (-0.012, -0.0055, 0.0)


def make_env(dps, seed=0):
    """`seed` is the fault draw's seed: it seeds the IMU's noise RNG, so the N draws have
    independent sensor-noise realisations (B11) instead of replaying `IMUConfig(seed=0)`."""
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200, gui=False, record=False,
        obstacles=False, rate_source="truth",
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=LEVER,
                          gyro_noise_std=0.05, accel_noise_std=0.02, seed=int(seed)))
    env.eval = False
    return env


def open_loop_transient(env, steps, seed):
    """Open-loop excitation for the first seconds after the fault — identification data only.

    This is where the lever arm is actually identifiable: right after the failure the body spins
    fast enough to saturate the gyro, and `id_lever_arm`'s saturated-window scan then recovers
    `k = |r_perp|` directly.  The nominal hover alone never saturates (at 1000 dps), so the scan
    raises and `k` degrades to `‖r_LS‖` — the BLOCKER-4 failure mode.  The flight here must stay
    **open loop**: closing the senior's policy around the vehicle would correlate the yaw command
    with the yaw rate and bias `id_yaw_channel`'s ARX.  Everything logged (gyro, accelerometer,
    commanded thrust, per-sample alive mask, velocity) is measurable on the real aircraft.
    """
    rng = np.random.default_rng(seed * 7919 + 13)
    mask = np.asarray(env.shut_down, float)
    base = float(env.M) * 9.81 / max(float(mask.sum()), 1.0)      # 2 alive rotors carry the weight
    walk = rng.normal(0.0, 0.05, 4)
    freqs = rng.uniform(0.2, 8.0, 4)
    phases = rng.uniform(0, 2 * np.pi, 4)
    amps = rng.uniform(0.0, 3.0, 4)
    L = {k: [] for k in ("a", "g", "u", "w", "v", "tom", "mask")}
    for i in range(steps):
        env._computeObs()
        walk = np.clip(walk * 0.995 + rng.normal(0.0, 0.05, 4), -1.5, 1.5)
        u_cmd = np.clip(base + walk + amps * np.sin(2 * np.pi * freqs * i * DT + phases), 0.0, 15.0)
        L["a"].append(env.accel_meas.copy()); L["g"].append(env.gyro_meas.copy())
        L["u"].append(u_cmd); L["w"].append(env.omega_true.copy())
        L["v"].append(env.vel[0].copy()); L["tom"].append(float(np.atleast_1d(env.last_acc)[0]))
        L["mask"].append(np.asarray(env.shut_down, float).copy())
        env.target_a, env.target_z_body = 9.81, np.array([0.0, 0.0, 1.0])
        env.step(u_cmd / 7.5 - 1.0)
    return {k: np.array(v) for k, v in L.items()}


def identification(seed=0, dps=1000.0, flag=3, post_s=None):
    """The deployment procedure's first half: hover + the open-loop yaw manoeuvre, then the fault
    and its open-loop transient -> priors.

    Returns ``(prior, diag, fault_rng, ic)``.  `fault_rng` is the global numpy RNG state captured
    immediately *before* ``shut_down_rotors(flag)`` (the env's own fault injection) and `ic` the
    resulting post-fault state.  Replaying that into a fresh environment reproduces one fault draw
    exactly, which is what makes the per-`src` comparison paired: every `src` sees the same initial
    spin, the same sensor-noise stream and the same commanded sequence.

    `post_s` (default `POST_FAULT_S`, 0 disables) is the length of open-loop logging after the fault
    that is appended to the identification log.
    """
    post_s = POST_FAULT_S if post_s is None else post_s
    np.random.seed(seed)
    env = make_env(dps, seed)
    env.reset()
    env.shut_down = np.ones(4)                       # healthy for the identification phase
    d = nominal_phase(env, steps=int(3.0 / DT), dps=dps, seed=seed)   # was 1.6 s: give the two-stage
                                                              # manoeuvre the same room the
                                                              # collector's episodes have
    fault_rng = np.random.get_state()
    env.shut_down_rotors(flag)      # reset() + mask + (flags 2/3) the initial yaw spin
    assert np.array_equal(env.shut_down, MASK[flag]), (env.shut_down, MASK[flag])
    ic = (env.pos[0].copy(), env.quat[0].copy(), env.vel[0].copy(),
          np.asarray(env.ang_vel, float).copy().ravel())
    n_pre = len(d["g"])                                  # the *nominal* (pre-fault) samples
    if post_s > 0:                                   # saturated window -> identifiable |r_perp|
        d2 = open_loop_transient(env, int(post_s / DT), seed)
        d = {k: np.concatenate([d[k], d2[k]], axis=0) for k in d}
    # the pre-fault history: measured, unsaturated, and available to a real vehicle -- it is what a
    # deployed estimator should start its window from instead of repeating the first post-fault sample
    pre = {k: np.asarray(d[k][:n_pre], float) for k in ("g", "a", "u", "mask")}
    env.close()
    prior, diag = identify_priors(d["g"], d["a"], d["u"], d["w"], d["mask"], dps, DT,
                                  vel=d["v"], tom=d["tom"])
    return prior, diag, fault_rng, ic, pre


def run(flag=3, dps=1000.0, src="net", ckpt=None, seed=0, steps=2000, target=(0.0, 1.0),
        prior=None, diag=None, fault_rng=None, ic=None, pre=None, preseed=True,
        tilt_obs=False, tilt_tau=0.5):
    """One closed-loop flight.  `prior`/`fault_rng`/`ic` come from `identification`, so several
    `src` values can be flown from the *same* fault draw (paired); calling with `prior=None`
    performs the identification and the fault draw itself.
    """
    if prior is None:
        prior, diag, fault_rng, ic, pre = identification(seed=seed, dps=dps, flag=flag)
    np.random.seed(seed)
    env = make_env(dps, seed)
    env.reset()
    np.random.set_state(fault_rng)                   # the env's own fault injection, same draw
    env.shut_down_rotors(flag)
    # NB: from here on the loop is byte-identical to scripts/verify_ins_attitude.run(...) with the
    # same post-fault state.  It is not loop fidelity that decides survival -- the closed loop is on
    # the stability boundary and ~15 % of the env's own injected spins escape regardless (see the
    # module docstring); that is why the caller reports a distribution over draws, never one seed.
    assert np.array_equal(env.shut_down, MASK[flag]), (env.shut_down, MASK[flag])
    if ic is not None:                               # exact same post-fault state for every src
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], ic[0].tolist(), ic[1].tolist(),
                                          physicsClientId=env.CLIENT)
        p.resetBaseVelocity(env.DRONE_IDS[0], ic[2].tolist(), ic[3].tolist(),
                            physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NetRate(ckpt, dev, prior, dps) if src == "net" else None
    if net is not None and preseed and pre is not None:
        # start the rolling window from the real pre-fault history (measured, unsaturated)
        net.preseed(pre["g"][-WINDOW:], pre["a"][-WINDOW:], pre["u"][-WINDOW:],
                    pre["mask"][-WINDOW:])
    policy = (load_policy(RL_NAME or CKPT_RL[flag], RL_DIR) if RL_DIR
              else load_policy(RL_NAME or CKPT_RL[flag]))
    pid = PositionPID()
    ins = AttitudeINS(quat_to_matrix(env.quat[0]))
    tob = TiltObserver(tau=tilt_tau) if tilt_obs else None   # one observer per flight (stateful)
    tp = np.array([0.0, 0.0, target[1]])
    z, xy, thr_e, err_n = [], [], [], []
    last_action = -np.ones(4)
    lp_state = None                                  # one-pole filter state for `--net_lp`
    for _ in range(steps):
        env._computeObs()
        gyro, accel, tom = env.gyro_meas.copy(), env.accel_meas.copy(), env.thrust_over_mass
        w_true = env.omega_true.copy()
        u_cmd = np.clip((last_action + 1.0) * 7.5, 0.0, 15.0)   # the raw command we sent (N/rotor)
        if src == "truth":
            rate = w_true
        elif src == "truth_noisy":
            rate = w_true + PN_BIAS + PN_RNG.normal(0.0, PN_SIGMA, 3)
        elif src == "clipped":
            rate = gyro
        else:
            w_hat = net.step(accel, gyro, u_cmd, env.shut_down)
            if NET_LP > 0:
                # B7: the causal one-pole low-pass on the network estimate.  `a = exp(-2*pi*fc*DT)`
                # is the pole of the equivalent RC filter; the state is seeded from this flight's
                # FIRST sample (so no start-up transient) and is used for the INS, the PID and the
                # observation alike -- exactly what `--net_lp` was meant to test.
                if lp_state is None:
                    lp_state = np.asarray(w_hat, float).copy()
                a_lp = np.exp(-2.0 * np.pi * NET_LP * DT)
                lp_state = a_lp * lp_state + (1.0 - a_lp) * np.asarray(w_hat, float)
                rate = lp_state.copy()
            else:
                rate = w_hat
        ins.update(rate, DT)
        if tob is not None:
            # bound the INS's drift with the acceleration-derived thrust axis BEFORE the PID and
            # before the observation are built, for every src (uniform, so the A/B is clean)
            ins.R = tob.correct(ins.R, env.vel[0], DT, omega=rate)   # half-step alignment (frame-corrected)
        R = ins.R
        q_att = Rotation.from_matrix(R).as_quat()
        ta, z_body = pid.step(DT, env.pos[0], q_att, env.vel[0], tp)
        r_xy = np.hypot(z_body[0], z_body[1])
        if r_xy > 0.26:
            s = 0.26 / r_xy
            z_body = np.array([z_body[0] * s, z_body[1] * s,
                               np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
        rel = R.T @ z_body
        # ONE observation builder for every consumer (the senior's convention, dims 7:11 = 
        # last_action[0] broadcast through the mask).  This line used to be a hand-written copy that
        # kept the element-wise form long after gpd_me.e2e.deploy_obs was fixed -- found by the user's
        # code review; scripts/test_obs_convention.py now also asserts these call sites.
        obs = deploy_obs(rel, rate, ta, tom, last_action, env.shut_down)
        action = policy.select_action(obs, deterministic=True)
        last_action = np.asarray(action, dtype=float).ravel()
        env.target_a, env.target_z_body = float(ta), z_body
        env.step(action)

        err_n.append(float(np.linalg.norm(np.asarray(rate, float) - w_true)))
        z.append(float(env.pos[0][2])); xy.append(float(np.hypot(env.pos[0][0], env.pos[0][1])))
        thr_e.append(float(np.rad2deg(AttitudeINS.tilt_error(R, quat_to_matrix(env.quat[0])))))
    z, xy = np.array(z), np.array(xy)
    w = max(1, len(z) // 3)
    env.close()
    return dict(z=float(z[-w:].mean()), zstd=float(z[-w:].std()),
                zmin=float(z.min()), zmax=float(z.max()), xy=float(xy[-w:].mean()),
                tilt=float(np.mean(thr_e)), rate_err=float(np.mean(err_n)),
                # the same error restricted to the *pre-divergence* window: averaging over a failed
                # episode's tumbling tail makes a healthy estimator look broken
                rate_err_early=float(np.mean(err_n[:300])),
                tilt_early=float(np.mean(thr_e[:300])),
                hold_steps=int(next((i for i, zz in enumerate(z) if zz < 0.3), len(z))),
                # B8: the algebraic front end uses prior[8] = |r_perp| (gpd_me/e2e.py:202), NOT
                # ||prior[:3]||.  Report the k actually in use (`k_perp`, which `--k_override`
                # writes) alongside the lever-arm norm it was being confused with.
                k_perp=float(prior[8]), k_hat_norm=float(np.linalg.norm(prior[:3])), diag=diag,
                tilt_used=(tob.n_used if tob is not None else None),
                tilt_rejected=(tob.n_rejected if tob is not None else None),
                net_compat=getattr(net, "compat", None))


SRCS = ["truth", "clipped", "net"]      # "truth_noisy" is appended when the tolerance
                                        # probe is active (see PN_BIAS/PN_SIGMA in main)


def summarize(rows):
    """Distributional summary of one `src` over N fault draws (paired across srcs)."""
    held = [r for r in rows if r["z"] > 0.3 and r["zstd"] < 3.0]
    zs = np.array([r["z"] for r in held]) if held else np.array([np.nan])
    return dict(n=len(rows), n_hold=len(held),
                z_med=float(np.median(zs)),
                z_iqr=float(np.percentile(zs, 75) - np.percentile(zs, 25)) if held else float("nan"),
                tilt=float(np.mean([r["tilt"] for r in held])) if held else float("nan"),
                rate_err=float(np.mean([r["rate_err"] for r in held])) if held else float("nan"),
                k_perp=float(np.mean([r["k_perp"] for r in rows])),
                k_hat_norm=float(np.mean([r["k_hat_norm"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ME, "results", "e2e_v5.pt"))
    ap.add_argument("--flag", type=int, default=3, choices=[0, 3])
    ap.add_argument("--ranges", default="1000",
                    help="comma-separated gyro range(s) in dps")
    ap.add_argument("--seeds", type=int, default=20,
                    help="number of independent fault draws (seeds 0..N-1 through the env's own "
                         "shut_down_rotors); each src is flown from the *same* draws")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rl_ckpt", default=None, help="override the inner-loop policy checkpoint name "
                                                   "(e.g. robust_v2_latest)")
    ap.add_argument("--dump_rows", action="store_true",
                    help="print every draw's raw stats, including the ones that did not hold "
                         "(rate_err/tilt/zmin/zmax) -- the only way to explain a failure")
    ap.add_argument("--k_override", type=float, default=None,
                    help="DIAGNOSTIC ONLY, NOT DEPLOYABLE: replace the identified |r_perp| "
                         "with this value to isolate how much of the in-loop estimator error "
                         "is the online identification of k")
    ap.add_argument("--no_preseed", action="store_true",
                    help="A/B switch: do NOT start the net's window from the real pre-fault "
                         "history (the old behaviour was to repeat the first post-fault sample)")
    ap.add_argument("--rl_dir", default=None, help="directory that holds it")
    ap.add_argument("--pn_bias", type=float, default=0.0,
                    help="tolerance probe: constant bias added to the TRUE rate [rad/s]")
    ap.add_argument("--pn_sigma", type=float, default=0.0,
                    help="tolerance probe: white noise std added to the TRUE rate [rad/s]")
    ap.add_argument("--net_lp", type=float, default=0.0,
                    help="causal low-pass cutoff [Hz] applied to the net estimate (0 = off)")
    ap.add_argument("--wilcoxon", type=int, default=0,
                    help="0 = off.  With N > 0, run N extra paired fault draws and print, per src "
                         "pair, the median of hold_steps(src) - hold_steps(truth) and the Wilcoxon "
                         "signed-rank p-value (the audit's section 5 item 1: the continuous paired "
                         "metric is the primary quantity, not the binary hold)")
    ap.add_argument("--tilt_obs", action="store_true",
                    help="audit section-5.2: bound the INS drift with the acceleration-derived "
                         "thrust axis (gpd_me.tilt.TiltObserver) before the PID/observation")
    ap.add_argument("--tilt_tau", type=float, default=0.5,
                    help="tilt observer blend time constant [s] (only with --tilt_obs)")
    ap.add_argument("--verbose", action="store_true", help="also print the per-draw final z of "
                                                           "every src (audit the escape rate)")
    a = ap.parse_args()
    global RL_NAME, RL_DIR
    RL_NAME, RL_DIR = a.rl_ckpt, a.rl_dir
    global PN_BIAS, PN_SIGMA, NET_LP
    PN_BIAS, PN_SIGMA = float(a.pn_bias), float(a.pn_sigma)
    NET_LP = float(a.net_lp)
    if PN_BIAS or PN_SIGMA:
        SRCS.append("truth_noisy")
    print("=== E4 closed loop: w_hat drives BOTH the attitude INS and the controller rate input ===")
    print(f"ckpt={a.ckpt}  mask={MASK[a.flag]}  {a.seeds} fault draws x {a.steps} steps  "
          f"(each src re-flown from the same draw)\n")
    print("Read every row as a distribution over independent initial spins, never as single-seed")
    print("pass/fail: the adjacent-pair case with this policy sits at the stability boundary, so")
    print("the escape rate is the measurement.  'held' = final z > 0.3 m and std(z) < 3 m.\n")
    print(f"{'dps':>6} {'src':<8} {'held':>7} {'z_med':>7} {'z_IQR':>6} {'tilt':>6} {'rate_err':>9} "
          f"{'k_perp':>7} {'k_norm':>7}  note")
    for dps in [float(x) for x in a.ranges.split(",")]:
        rows = {s: [] for s in SRCS}
        for seed in range(a.seeds):
            # one identification + one fault draw per draw index, shared by every src (paired)
            prior, diag, fault_rng, ic, pre = identification(seed=seed, dps=dps, flag=a.flag)
            if a.k_override is not None:
                prior = np.asarray(prior, float).copy()
                prior[8] = float(a.k_override)     # |r_perp|: diagnostic override, never at deployment
            for src in SRCS:
                row = run(flag=a.flag, dps=dps, src=src, ckpt=a.ckpt, seed=seed,
                          steps=a.steps, prior=prior, diag=diag, fault_rng=fault_rng, ic=ic,
                          pre=pre, preseed=not a.no_preseed,
                          tilt_obs=a.tilt_obs, tilt_tau=a.tilt_tau)
                rows[src].append(row)
                if a.dump_rows:      # printing only: the failures are the rows worth explaining
                    ts = ("" if row["tilt_used"] is None else
                          f" | tilt_obs n_used={row['tilt_used']} n_rejected={row['tilt_rejected']}")
                    print(f"[rows] seed={seed} src={src:6s} held={int(row['z'] > 0.3 and row['zstd'] < 3.0)} "
                          f"z={row['z']:6.2f} zstd={row['zstd']:6.2f} zmin={row['zmin']:6.2f} "
                          f"tilt={row['tilt']:6.1f} rate_err={row['rate_err']:6.2f} "
                          f"| first1.5s: rate_err={row['rate_err_early']:5.2f} tilt={row['tilt_early']:5.1f} "
                          f"| lost@step={row['hold_steps']:4d} "
                          f"k_perp={row['k_perp']*100:5.2f}cm k_norm={row['k_hat_norm']*100:5.2f}cm"
                          + ts, flush=True)
        for src in SRCS:
            m = summarize(rows[src])
            note = ""
            if src == "net" and rows[src][0].get("net_compat") is False:
                note = "ckpt on OLD 26-feature layout -> falls back to w_alg (retrain)"
            print(f"{dps:>6.0f} {src:<8} {m['n_hold']:>3d}/{m['n']:<3d} {m['z_med']:>7.2f} "
                  f"{m['z_iqr']:>6.2f} {m['tilt']:>6.1f} {m['rate_err']:>9.2f} "
                  f"{m['k_perp']*100:>6.2f}cm {m['k_hat_norm']*100:>6.2f}cm  {note}")
        if a.verbose:
            for i in range(a.seeds):
                print(f"        draw {i:3d}  " + "  ".join(
                    f"{s}={rows[s][i]['z']:7.2f}" for s in SRCS))
        if a.wilcoxon > 0:
            # Audit section 5 item 1: the primary quantity is the PAIRED continuous hold_steps
            # (the step at which z first drops below 0.3 m), not the binary hold.  Draws already
            # flown above are reused; only the excess is identified and flown now.
            n_w = int(a.wilcoxon)
            hs = {s: [] for s in SRCS}
            for seed in range(n_w):
                if seed < len(rows["truth"]):
                    for src in SRCS:
                        hs[src].append(rows[src][seed]["hold_steps"])
                    continue
                prior_w, diag_w, rng_w, ic_w, pre_w = identification(seed=seed, dps=dps, flag=a.flag)
                if a.k_override is not None:
                    prior_w = np.asarray(prior_w, float).copy()
                    prior_w[8] = float(a.k_override)
                for src in SRCS:
                    row = run(flag=a.flag, dps=dps, src=src, ckpt=a.ckpt, seed=seed,
                              steps=a.steps, prior=prior_w, diag=diag_w, fault_rng=rng_w,
                              ic=ic_w, pre=pre_w, preseed=not a.no_preseed,
                              tilt_obs=a.tilt_obs, tilt_tau=a.tilt_tau)
                    hs[src].append(row["hold_steps"])
            base = np.asarray(hs["truth"], float)
            print(f"\n  paired continuous metric (Wilcoxon signed-rank, {n_w} draws, "
                  f"paired on the same fault draws):")
            print(f"    {'pair':<26} {'median d(hold_steps)':>22} {'p-value':>10}")
            for src in SRCS:
                if src == "truth":
                    continue
                d = np.asarray(hs[src], float) - base
                med = float(np.median(d))
                if np.all(d == 0):
                    p = float("nan")          # wilcoxon() raises on an all-zero difference vector
                else:
                    try:
                        p = float(wilcoxon(d).pvalue)
                    except ValueError:
                        p = float("nan")
                print(f"    {f'hold_steps({src}) - hold_steps(truth)':<26} {med:>22.1f} {p:>10.4g}")


if __name__ == "__main__":
    main()