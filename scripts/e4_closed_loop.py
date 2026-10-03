"""E4 — closed-loop main experiment with the learned estimator in the loop.

Episode structure (identical to the data collector, so the deployment procedure is exercised):
    1. nominal phase: hover + the two-stage OPEN-LOOP yaw-identification manoeuvre
       -> every prior the estimator needs is identified from this flight data only
    2. fault injection (adjacent-pair dual failure by default)
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

from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.e2e import WINDOW, E2ENet, algebraic_estimate, coarse_summary, fine_features  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.ins import AttitudeINS  # noqa: E402
from gpd_me.policy import PositionPID, load_policy, quat_to_matrix  # noqa: E402
from gpd_me.priors import identify_priors, tom_from_command  # noqa: E402

MASK = {0: [0, 1, 1, 1], 3: [0, 0, 1, 1]}
CKPT_RL = {0: "shutdown_real_7", 3: "shutdown_real_7_4"}
DT = 1.0 / 200.0
_WARNED = set()          # checkpoints already reported as incompatible (loud once, not per run)


class NetRate:
    """Causal wrapper around the trained estimator: keeps a rolling window and returns w_hat."""

    def __init__(self, ckpt, dev, prior, dps):
        self.dev = dev
        self.prior = np.asarray(prior, float)
        # prior = [r(3), g_T, G, T, range_rad, tau, k]; the algebraic front end needs k = |r_perp|
        self.g_T = float(self.prior[3]); self.tau = float(self.prior[7]); self.k = float(self.prior[8])
        self.lim = np.deg2rad(dps)
        self.net = E2ENet().to(dev).eval()
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        try:
            self.net.load_state_dict(ck["state"])
            self.compat = True
            # the checkpoint stores numpy float64 normalisation constants -> cast, or the first
            # Linear sees float64 and torch aborts with "mat1 and mat2 must have the same dtype"
            self.xf_m = torch.tensor(ck["xf_m"], device=dev, dtype=torch.float32)
            self.xf_s = torch.tensor(ck["xf_s"], device=dev, dtype=torch.float32)
            self.xp_m = torch.tensor(ck["xp_m"], device=dev, dtype=torch.float32)
            self.xp_s = torch.tensor(ck["xp_s"], device=dev, dtype=torch.float32)
        except RuntimeError as exc:
            # The checkpoint predates the current feature/prior layout (its fine vector is 26 wide
            # with 7 priors; the current one is 28/9).  It cannot be loaded, so the residual heads
            # stay at their zero init and w_hat reduces to the algebraic physics front end -- the
            # network output is only valid again after retraining.  Kept loud, not silent.
            self.compat = False
            msg = next((ln.strip() for ln in str(exc).splitlines() if "size mismatch" in ln),
                       str(exc).splitlines()[0].strip())
            if ckpt not in _WARNED:
                _WARNED.add(ckpt)
                print(f"[e4] WARNING: {ckpt} was trained on the OLD 26-feature / 7-prior layout and "
                      f"cannot be loaded ({msg}).  The 'net' row falls back to the algebraic "
                      f"estimate; retrain to make it valid.")
            F = fine_features(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), bool),
                              np.zeros((1, 4)), np.zeros(1), self.prior, np.zeros((1, 3))).shape[1]
            self.xf_m = torch.zeros(F, device=dev); self.xf_s = torch.ones(F, device=dev)
            self.xp_m = torch.zeros(len(self.prior), device=dev)
            self.xp_s = torch.ones(len(self.prior), device=dev)
        self.buf = {k: [] for k in ("a", "g", "u", "m")}

    def step(self, accel, gyro, u_cmd, mask):
        """One causal estimator step: returns w_hat for the current instant.

        `u_cmd` is the *commanded* per-rotor thrust (N) this controller sent and `mask` the
        per-sample alive-rotor mask; both feed the identified actuator model, so the specific thrust
        the estimator uses is `tom_from_command(...)`, never the simulator's true thrust.
        """
        for key, val in (("a", accel), ("g", gyro), ("u", u_cmd), ("m", mask)):
            self.buf[key].append(np.asarray(val, float))
        H = min(WINDOW, len(self.buf["a"]))
        pad = WINDOW - H
        rep = lambda x: np.concatenate([np.repeat(x[:1], pad, 0), x], 0) if pad else x
        a_ = rep(np.stack(self.buf["a"][-H:])); g_ = rep(np.stack(self.buf["g"][-H:]))
        u_ = rep(np.stack(self.buf["u"][-H:])); m_ = rep(np.stack(self.buf["m"][-H:]))
        sat_ = np.abs(g_) >= self.lim - 1e-9
        tom_ = tom_from_command(u_, m_, DT, self.g_T, self.tau)
        w_alg = algebraic_estimate(g_, a_ - np.array([0.0, 0.0, 1.0]) * tom_[:, None],
                                   self.k, sat_, self.lim)
        # coarse summary over the trailing 2 s (padded for the first samples)
        Na = len(self.buf["a"]); Hc = min(400, Na); repc = lambda x: np.concatenate(
            [np.repeat(x[:1], 400 - Hc, 0), x], 0) if 400 - Hc else x
        cg = repc(np.stack(self.buf["g"][-Hc:])); cm = repc(np.stack(self.buf["m"][-Hc:]))
        cu = repc(np.stack(self.buf["u"][-Hc:]))
        ctom_ = tom_from_command(cu, cm, DT, self.g_T, self.tau)
        cs = coarse_summary(cg, np.abs(cg) >= self.lim - 1e-9, cu, ctom_, self.prior)
        T_ = lambda x: torch.tensor(np.asarray(x)[None], dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            ft = ((T_(fine_features(a_, g_, sat_, u_, tom_, self.prior, w_alg)) - self.xf_m)
                  / self.xf_s).clamp(-50, 50)
            pt = (T_(self.prior) - self.xp_m) / self.xp_s
            w_hat, _ = self.net(ft, T_(cs), pt, T_(w_alg), torch.tensor(sat_[None], device=self.dev))
        return w_hat[0, -1].cpu().numpy().astype(float)


def nominal_phase(env, steps=400, dps=1000.0, seed=0):
    """Hover + two-stage open-loop yaw identification; returns the logs needed for the priors."""
    rng = np.random.default_rng(seed)
    dt = DT
    yid_f = float(rng.uniform(0.3, 2.0)); yid_target = 0.30 * np.deg2rad(dps)
    yid_d0 = float(rng.uniform(0.3, 1.0)); yid_tA = int(0.2 / dt)
    yid_amp, yid_w0 = None, None
    hover_u = float(env.M) * 9.81 / 4.0
    L = {k: [] for k in ("a", "g", "u", "w", "v", "tom", "mask")}
    for i in range(steps):
        env._computeObs()
        z = float(env.pos[0][2])
        base = hover_u + 2.0 * (1.0 - z)
        if i == yid_tA:
            yid_w0 = float(env.omega_true[2])
        if i < yid_tA:
            exc = yid_d0
        else:
            if yid_amp is None:
                dtA = max(i - yid_tA, 1) * dt
                gain = abs(float(env.omega_true[2]) - yid_w0) / max(abs(yid_d0) * dtA, 1e-6)
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


def make_env(dps):
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200, gui=False, record=False,
        obstacles=False, rate_source="truth",
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=(-0.012, -0.0055, 0.0),
                          gyro_noise_std=0.05, accel_noise_std=0.02))
    env.eval = False
    return env


def identification(seed=0, dps=1000.0, flag=3):
    """The deployment procedure's first half: hover + the open-loop yaw manoeuvre -> priors.

    Returns ``(prior, diag, fault_rng, ic)``.  `fault_rng` is the global numpy RNG state captured
    immediately *before* ``shut_down_rotors(flag)`` (the env's own fault injection) and `ic` the
    resulting post-fault state.  Replaying that into a fresh environment reproduces one fault draw
    exactly, which is what makes the per-`src` comparison paired: every `src` sees the same initial
    spin, the same sensor-noise stream and the same commanded sequence.
    """
    np.random.seed(seed)
    env = make_env(dps)
    env.reset()
    env.shut_down = np.ones(4)                       # healthy for the identification phase
    d = nominal_phase(env, steps=int(1.6 / DT), dps=dps, seed=seed)
    prior, diag = identify_priors(d["g"], d["a"], d["u"], d["w"], d["mask"], dps, DT,
                                  vel=d["v"], tom=d["tom"])
    fault_rng = np.random.get_state()
    env.shut_down_rotors(flag)      # reset() + mask + (flags 2/3) the initial yaw spin
    assert np.array_equal(env.shut_down, MASK[flag]), (env.shut_down, MASK[flag])
    ic = (env.pos[0].copy(), env.quat[0].copy(), env.vel[0].copy(),
          np.asarray(env.ang_vel, float).copy().ravel())
    env.close()
    return prior, diag, fault_rng, ic


def run(flag=3, dps=1000.0, src="net", ckpt=None, seed=0, steps=2000, target=(0.0, 1.0),
        prior=None, diag=None, fault_rng=None, ic=None):
    """One closed-loop flight.  `prior`/`fault_rng`/`ic` come from `identification`, so several
    `src` values can be flown from the *same* fault draw (paired); calling with `prior=None`
    performs the identification and the fault draw itself.
    """
    if prior is None:
        prior, diag, fault_rng, ic = identification(seed=seed, dps=dps, flag=flag)
    np.random.seed(seed)
    env = make_env(dps)
    env.reset()
    np.random.set_state(fault_rng)                   # the env's own fault injection, same draw
    env.shut_down_rotors(flag)
    assert np.array_equal(env.shut_down, MASK[flag]), (env.shut_down, MASK[flag])
    if ic is not None:                               # exact same post-fault state for every src
        p.resetBasePositionAndOrientation(env.DRONE_IDS[0], ic[0].tolist(), ic[1].tolist(),
                                          physicsClientId=env.CLIENT)
        p.resetBaseVelocity(env.DRONE_IDS[0], ic[2].tolist(), ic[3].tolist(),
                            physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NetRate(ckpt, dev, prior, dps) if src == "net" else None
    policy = load_policy(CKPT_RL[flag])
    pid = PositionPID()
    ins = AttitudeINS(quat_to_matrix(env.quat[0]))
    tp = np.array([0.0, 0.0, target[1]])
    z, xy, thr_e, err_n = [], [], [], []
    last_action = -np.ones(4)
    for _ in range(steps):
        env._computeObs()
        gyro, accel, tom = env.gyro_meas.copy(), env.accel_meas.copy(), env.thrust_over_mass
        w_true = env.omega_true.copy()
        u_cmd = np.clip((last_action + 1.0) * 7.5, 0.0, 15.0)   # the raw command we sent (N/rotor)
        if src == "truth":
            rate = w_true
        elif src == "clipped":
            rate = gyro
        else:
            rate = net.step(accel, gyro, u_cmd, env.shut_down)
        ins.update(rate, DT)
        R = ins.R
        q_att = Rotation.from_matrix(R).as_quat()
        ta, z_body = pid.step(DT, env.pos[0], q_att, env.vel[0], tp)
        r_xy = np.hypot(z_body[0], z_body[1])
        if r_xy > 0.26:
            s = 0.26 / r_xy
            z_body = np.array([z_body[0] * s, z_body[1] * s,
                               np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
        rel = R.T @ z_body
        obs = np.array([rel[0], rel[1],
                        rate[0] / 10, rate[1] / 10, rate[2] / 50,
                        (ta - 9.8) / 3, (tom - 9.8) / 3,
                        *(last_action * env.shut_down), *(env.shut_down * 2 - 1)], dtype=np.float32)
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
                k=float(np.linalg.norm(prior[:3])), diag=diag,
                net_compat=getattr(net, "compat", None))


SRCS = ("truth", "clipped", "net")


def summarize(rows):
    """Distributional summary of one `src` over N fault draws (paired across srcs)."""
    held = [r for r in rows if r["z"] > 0.3 and r["zstd"] < 3.0]
    zs = np.array([r["z"] for r in held]) if held else np.array([np.nan])
    return dict(n=len(rows), n_hold=len(held),
                z_med=float(np.median(zs)),
                z_iqr=float(np.percentile(zs, 75) - np.percentile(zs, 25)) if held else float("nan"),
                tilt=float(np.mean([r["tilt"] for r in held])) if held else float("nan"),
                rate_err=float(np.mean([r["rate_err"] for r in held])) if held else float("nan"),
                k=float(np.mean([r["k"] for r in rows])))


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
    ap.add_argument("--verbose", action="store_true", help="also print the per-draw final z of "
                                                           "every src (audit the escape rate)")
    a = ap.parse_args()
    print("=== E4 closed loop: w_hat drives BOTH the attitude INS and the controller rate input ===")
    print(f"ckpt={a.ckpt}  mask={MASK[a.flag]}  {a.seeds} fault draws x {a.steps} steps  "
          f"(each src re-flown from the same draw)\n")
    print("Read every row as a distribution over independent initial spins, never as single-seed")
    print("pass/fail: the adjacent-pair case with this policy sits at the stability boundary, so")
    print("the escape rate is the measurement.  'held' = final z > 0.3 m and std(z) < 3 m.\n")
    print(f"{'dps':>6} {'src':<8} {'held':>7} {'z_med':>7} {'z_IQR':>6} {'tilt':>6} {'rate_err':>9} "
          f"{'k_hat':>7}  note")
    for dps in [float(x) for x in a.ranges.split(",")]:
        rows = {s: [] for s in SRCS}
        for seed in range(a.seeds):
            # one identification + one fault draw per draw index, shared by every src (paired)
            prior, diag, fault_rng, ic = identification(seed=seed, dps=dps, flag=a.flag)
            for src in SRCS:
                rows[src].append(run(flag=a.flag, dps=dps, src=src, ckpt=a.ckpt, seed=seed,
                                     steps=a.steps, prior=prior, diag=diag,
                                     fault_rng=fault_rng, ic=ic))
        for src in SRCS:
            m = summarize(rows[src])
            note = ""
            if src == "net" and rows[src][0].get("net_compat") is False:
                note = "ckpt on OLD 26-feature layout -> falls back to w_alg (retrain)"
            print(f"{dps:>6.0f} {src:<8} {m['n_hold']:>3d}/{m['n']:<3d} {m['z_med']:>7.2f} "
                  f"{m['z_iqr']:>6.2f} {m['tilt']:>6.1f} {m['rate_err']:>9.2f} "
                  f"{m['k']*100:>6.2f}cm  {note}")
        if a.verbose:
            for i in range(a.seeds):
                print(f"        draw {i:3d}  " + "  ".join(
                    f"{s}={rows[s][i]['z']:7.2f}" for s in SRCS))


if __name__ == "__main__":
    main()