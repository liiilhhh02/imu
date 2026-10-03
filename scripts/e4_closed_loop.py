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

Metrics: stable bias and jitter of the rate error (the bias is what integrates into attitude error),
the INS tilt error versus truth, altitude / horizontal drift, and whether it flies.

Run:  PYTHONPATH=<repo>:<me> python scripts/e4_closed_loop.py --ckpt results/e2e_v4.pt
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

from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.e2e import COARSE, WINDOW, E2ENet, algebraic_estimate, att_errors_deg, coarse_summary, \
    fine_features, ins_rollout  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.ins import AttitudeINS  # noqa: E402
from gpd_me.policy import PositionPID, load_policy, quat_to_matrix  # noqa: E402
from gpd_me.priors import identify_priors  # noqa: E402

MASK = {0: [0, 1, 1, 1], 3: [0, 0, 1, 1]}
CKPT_RL = {0: "shutdown_real_7", 3: "shutdown_real_7_4"}
DT = 1.0 / 200.0


class NetRate:
    """Causal wrapper around the trained estimator: keeps a rolling window and returns w_hat."""

    def __init__(self, ckpt, dev, k_hat, prior, dps):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.net = E2ENet().to(dev).eval()
        self.net.load_state_dict(ck["state"])
        # the checkpoint stores numpy float64 normalisation constants -> cast, or the first Linear
        # sees float64 and torch aborts with "mat1 and mat2 must have the same dtype"
        self.xf_m = torch.tensor(ck["xf_m"], device=dev, dtype=torch.float32)
        self.xf_s = torch.tensor(ck["xf_s"], device=dev, dtype=torch.float32)
        self.xp_m = torch.tensor(ck["xp_m"], device=dev, dtype=torch.float32)
        self.xp_s = torch.tensor(ck["xp_s"], device=dev, dtype=torch.float32)
        self.dev, self.k, self.prior, self.lim = dev, k_hat, np.asarray(prior, float), np.deg2rad(dps)
        self.buf = {k: [] for k in ("a", "g", "s", "u", "tom")}

    def step(self, accel, gyro, u_cmd, tom):
        """One causal estimator step: returns w_hat for the current instant."""
        for key, val in (("a", accel), ("g", gyro), ("u", u_cmd), ("tom", tom)):
            self.buf[key].append(np.asarray(val, float))
        H = min(WINDOW, len(self.buf["a"]))
        pad = WINDOW - H
        rep = lambda x: np.concatenate([np.repeat(x[:1], pad, 0), x], 0) if pad else x
        a_ = rep(np.stack(self.buf["a"][-H:])); g_ = rep(np.stack(self.buf["g"][-H:]))
        u_ = rep(np.stack(self.buf["u"][-H:])); t_ = rep(np.array(self.buf["tom"][-H:]))
        sat_ = np.abs(g_) >= self.lim - 1e-9
        w_alg = algebraic_estimate(g_, a_ - np.array([0.0, 0.0, 1.0]) * t_[:, None],
                                   self.k, sat_, self.lim)
        # coarse summary over the trailing 2 s (padded for the first samples)
        Na = len(self.buf["a"]); Hc = min(400, Na); repc = lambda x: np.concatenate(
            [np.repeat(x[:1], 400 - Hc, 0), x], 0) if 400 - Hc else x
        ca = repc(np.stack(self.buf["a"][-Hc:])); cg = repc(np.stack(self.buf["g"][-Hc:]))
        cu = repc(np.stack(self.buf["u"][-Hc:])); ct_ = repc(np.array(self.buf["tom"][-Hc:]))
        cs = coarse_summary(cg, np.abs(cg) >= self.lim - 1e-9, cu, ct_, self.prior)
        T_ = lambda x: torch.tensor(np.asarray(x)[None], dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            ft = ((T_(fine_features(a_, g_, sat_, u_, t_, self.prior, w_alg)) - self.xf_m)
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


def run(flag=3, dps=1000.0, src="net", ckpt=None, seed=0, steps=2000, target=(0.0, 1.0)):
    np.random.seed(seed)
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200, gui=False, record=False,
        obstacles=False, rate_source="truth",
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=(-0.012, -0.0055, 0.0),
                          gyro_noise_std=0.05, accel_noise_std=0.02))
    env.eval = False
    env.reset()
    env.shut_down = np.ones(4)                       # healthy for the identification phase
    d = nominal_phase(env, steps=int(1.6 / DT), dps=dps, seed=seed)
    prior, diag = identify_priors(d["g"], d["a"], d["u"], d["w"], d["mask"], dps, DT,
                                  vel=d["v"], tom=d["tom"])

    env.shut_down_rotors(flag)
    env.shut_down = np.array(MASK[flag], float)
    p.resetBaseVelocity(objectUniqueId=env.DRONE_IDS[0], linearVelocity=[0, 0, 0],
                        angularVelocity=[0, 0, -np.random.uniform(10, 25)], physicsClientId=env.CLIENT)
    env._updateAndStoreKinematicInformation()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NetRate(ckpt, dev, float(np.linalg.norm(prior[:3])), prior, dps) if src == "net" else None
    policy = load_policy(CKPT_RL[flag])
    pid = PositionPID()
    ins = AttitudeINS(quat_to_matrix(env.quat[0]))
    tp = np.array([0.0, 0.0, target[1]])
    z, xy, bias_e, jit_e, thr_e = [], [], [], [], []
    err_hist = []
    for _ in range(steps):
        env._computeObs()
        gyro, accel, tom = env.gyro_meas.copy(), env.accel_meas.copy(), env.thrust_over_mass
        u_prev = np.clip((env.last_action[0] + 1) * 7.5, 0, 15) * env.shut_down
        if src == "truth":
            rate = env.omega_true.copy()
        elif src == "clipped":
            rate = gyro
        else:
            rate = net.step(accel, gyro, u_prev, tom)
        ins.update(rate, DT)
        R = ins.R
        q_att = Rotation.from_matrix(R).as_quat()
        ta, z_body = pid.step(DT, env.pos[0], q_att, env.vel[0], tp)
        r_xy = np.hypot(z_body[0], z_body[1])
        if r_xy > 0.26:
            s = 0.26 / r_xy
            z_body = np.array([z_body[0] * s, z_body[1] * s,
                               np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
        weight = np.zeros(4)
        rel = R.T @ z_body
        w_prev = 0.3 * (ins.R @ np.zeros(3))
        obs = np.array([rel[0], rel[1],
                        rate[0] / 10, rate[1] / 10, rate[2] / 50,
                        (ta - 9.8) / 3, (tom - 9.8) / 3,
                        *(env.last_action[0] * env.shut_down), *(env.shut_down * 2 - 1)], dtype=np.float32)
        action = policy.select_action(obs, deterministic=True)
        env.target_a, env.target_z_body = float(ta), z_body
        env.step(action)

        err = rate - env.omega_true
        err_hist.append(err.copy())
        z.append(float(env.pos[0][2])); xy.append(float(np.hypot(env.pos[0][0], env.pos[0][1])))
        thr_e.append(float(np.rad2deg(AttitudeINS.tilt_error(R, quat_to_matrix(env.quat[0])))))
    E = np.array(err_hist)
    w = len(z) // 3
    env.close()
    return dict(z=float(np.mean(z[-w:])), zstd=float(np.std(z[-w:])), xy=float(np.mean(xy[-w:])),
                bias=float(np.linalg.norm(E[-w:].mean(0))),
                jitter=float(np.linalg.norm(E[-w:] - E[-w:].mean(0), axis=1).mean()),
                tilt=float(np.mean(thr_e[-w:])), k=float(np.linalg.norm(prior[:3])), diag=diag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ME, "results", "e2e_v4.pt"))
    ap.add_argument("--flag", type=int, default=3, choices=[0, 3])
    ap.add_argument("--ranges", default="400,700,1000")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    print("=== E4 closed loop: w_hat drives BOTH the attitude INS and the controller rate input ===")
    print(f"ckpt={a.ckpt}  mask={MASK[a.flag]}  {a.seeds} seeds x {a.steps} steps\n")
    print(f"{'dps':>6} {'src':<8} {'z':>7} {'std':>6} {'xy':>7} {'bias':>6} {'jitter':>7} "
          f"{'tilt':>7} {'k_hat':>7}  verdict")
    for dps in [float(x) for x in a.ranges.split(",")]:
        for src in ("truth", "clipped", "net"):
            R = [run(flag=a.flag, dps=dps, src=src, ckpt=a.ckpt, seed=s, steps=a.steps)
                 for s in range(a.seeds)]
            m = lambda k: float(np.mean([r[k] for r in R]))
            z, zs = m("z"), m("zstd")
            ok = "holds" if (z > 0.3 and zs < 3.0) else "LOST"
            print(f"{dps:>6.0f} {src:<8} {z:>7.2f} {zs:>6.2f} {m('xy'):>7.3f} {m('bias'):>6.2f} "
                  f"{m('jitter'):>7.2f} {m('tilt'):>7.1f} {m('k')*100:>6.2f}cm  {ok}")


if __name__ == "__main__":
    main()