"""Collect data and train the learned out-of-range rate estimator.

Data collection (simulation, privileged ground truth available for supervision only)
----------------------------------------------------------------------------------
Episodes are flown with the controller fed the *ground-truth* rate, so the vehicle stays in the
mission-relevant regime while the IMU is saturated — i.e. the data cover exactly the state
distribution the estimator will face at deployment.  Each episode randomises
  * gyro range (1000-2500 dps), gyro bias/scale error, accel noise,
  * the lever arm r (direction and magnitude), mass, KM, actuator delay,
  * the fault mask and the injected initial spin,
so that the network cannot memorise one airframe.

The lever-arm scale k_hat is produced by the *same* procedure used at deployment: linear LS on the
in-range samples when there are enough of them, otherwise the 1-D scan (see gpd_me.observer).

Training
--------
Windows of H=12 frames, losses as in gpd_me.estimator.  Evaluation is against the two baselines that
matter: the clipped gyro (what a real IMU gives) and the algebraic lever-arm observer.

Run:  PYTHONPATH=<repo>:<me> python scripts/train_estimator.py [--episodes 24] [--iters 3000]
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.estimator import EstimatorNet, WINDOW, ins_rollout, tilt_error_deg  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.observer import LeverArmObserver  # noqa: E402
from gpd_me.policy import PositionPID, load_policy, quat_to_matrix  # noqa: E402

CKPT = {0: "shutdown_real_7", 2: "shutdown_real_7_4", 3: "shutdown_real_7_4"}
DT = 1.0 / 200.0
LIM_DPS = (1000.0, 1500.0, 2000.0, 2500.0)


def one_episode(flag, dps, rng, steps=1200):
    """Fly with the true rate, log everything, return the per-step arrays."""
    lever = rng.normal(size=3); lever /= np.linalg.norm(lever)
    lever *= rng.uniform(0.006, 0.030)
    mass, km, delay = rng.uniform(0.8, 1.2), rng.uniform(0.008, 0.012), rng.uniform(0.02, 0.032)
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"),
        aggregate_phy_steps=1, freq=200, gui=False, record=False, obstacles=False,
        rate_source="truth",
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=tuple(lever),
                          gyro_noise_std=0.05, accel_noise_std=0.02,
                          gyro_bias_std=rng.uniform(0.0, 0.02)))
    env.eval = False
    env.reset()
    env.shut_down_rotors(flag)
    env.M, env.KM, env.delay = mass, km, delay          # randomised plant
    env.shut_down = np.array({0: [0, 1, 1, 1], 2: [0, 0, 0, 1], 3: [0, 0, 1, 1]}[flag], float)
    policy = load_policy(CKPT[flag])
    pid = PositionPID()
    tp = np.array([0.0, 0.0, 1.0])
    L = {k: [] for k in ("s", "g", "a", "u", "w", "R", "tom", "sat")}
    for _ in range(steps):
        env._computeObs()
        q = env.quat[0]
        qa = Rotation.from_quat(q).as_quat()
        ta, z_body = pid.step(DT, env.pos[0], qa, env.vel[0], tp)
        r_xy = np.hypot(z_body[0], z_body[1])
        if r_xy > 0.26:
            sc = 0.26 / r_xy
            z_body = np.array([z_body[0] * sc, z_body[1] * sc,
                               np.sqrt(1 - (z_body[0] * sc) ** 2 - (z_body[1] * sc) ** 2)])
        tom = env.thrust_over_mass
        L["s"].append(env.accel_meas - np.array([0.0, 0.0, tom]))
        L["g"].append(env.gyro_meas.copy())
        L["a"].append(env.accel_meas.copy())
        L["tom"].append(tom)
        L["w"].append(env.omega_true.copy())
        L["R"].append(quat_to_matrix(q))
        L["sat"].append(np.abs(env.gyro_meas) >= np.deg2rad(dps) - 1e-9)
        env.target_a, env.target_z_body = float(ta), z_body
        R_att = quat_to_matrix(q)
        obs = np.array([(R_att.T @ z_body)[0], (R_att.T @ z_body)[1],
                        env.omega_true[0] / 10, env.omega_true[1] / 10, env.omega_true[2] / 50,
                        (ta - 9.8) / 3, (tom - 9.8) / 3,
                        *(env.last_action[0] * env.shut_down), *(env.shut_down * 2 - 1)],
                       dtype=np.float32)
        action = policy.select_action(obs, deterministic=True)
        L["u"].append(np.clip((action + 1) * 7.5, 0, 15) * env.shut_down)
        env.step(action)
    env.close()
    return {k: np.array(v) for k, v in L.items()}, dict(lever=lever, M=mass, KM=km, delay=delay)


def identify_k(gyro, s, lim, calib=250):
    """The deployment procedure: in-range LS first, else the 1-D scan."""
    ob = LeverArmObserver(gyro_limit_dps=np.rad2deg(lim))
    inr = ~np.abs(gyro).max(axis=1).__ge__(lim - 1e-9)
    try:
        if int((inr & (np.linalg.norm(gyro, axis=1) > 2.0)).sum()) >= 8:
            k = ob.calibrate_inrange(gyro[inr], (s + np.array([0.0, 0.0, 0.0]))[inr], np.zeros(inr.sum()))
            return k, "inrange"
    except Exception:
        pass
    try:
        ob.calibrate_scan(gyro[:calib], s[:calib] + 0.0, np.zeros(min(calib, len(gyro))))
        return ob.k, "scan"
    except Exception:
        return np.nan, "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(ME, "results", "estimator_gru.pt"))
    ap.add_argument("--load", default=None, help="skip training, load this checkpoint")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    print("=== collecting rollouts (controller fed the true rate, IMU saturated) ===")
    data = []
    t0 = time.time()
    for ep in range(a.episodes):
        flag = int(rng.choice([0, 2, 3]))
        dps = float(rng.choice(LIM_DPS))
        d, plant = one_episode(flag, dps, rng)
        lim = np.deg2rad(dps)
        k_hat, how = identify_k(d["g"], d["s"], lim)
        ob = LeverArmObserver(gyro_limit_dps=dps, lever_scale=k_hat)
        w_alg = np.stack([ob.step(d["g"][i], d["a"][i], d["tom"][i]) for i in range(len(d["g"]))])
        d.update(k_hat=k_hat, how=how, w_alg=w_alg, flag=flag, dps=dps, plant=plant)
        data.append(d)
        if ep % 4 == 0 or ep == a.episodes - 1:
            sat = d["sat"].any(axis=1)
            print(f"  ep{ep:3d} flag={flag} dps={dps:.0f} k_hat={k_hat*100:.3f}cm "
                  f"(true {np.linalg.norm(plant['lever'])*100:.3f}) [{how}] sat={100*sat.mean():.0f}% "
                  f"alg err={np.abs(w_alg[sat, 2]-d['w'][sat, 2]).mean():6.2f} rad/s")
    print(f"collected in {time.time()-t0:.0f}s")

    # ---- windows ----
    X, W_ALG, W_TRUE, R_TRUE, SAT, INR, R0, META = [], [], [], [], [], [], [], []
    for d in data:
        n = len(d["g"])
        feats = EstimatorNet.build_features(d["s"], d["g"], d["sat"], d["u"], d["tom"],
                                            d["w_alg"], d["k_hat"], np.deg2rad(d["dps"]))
        for i in range(0, n - WINDOW - 1, 4):
            X.append(feats[i:i + WINDOW])
            W_ALG.append(d["w_alg"][i:i + WINDOW])
            W_TRUE.append(d["w"][i:i + WINDOW])
            R_TRUE.append(d["R"][i:i + WINDOW])
            SAT.append(d["sat"][i:i + WINDOW])
            INR.append(~d["sat"][i:i + WINDOW].any(axis=1))
            R0.append(d["R"][i])
            META.append((d["flag"], d["dps"]))
    X = torch.tensor(np.stack(X)); W_ALG = torch.tensor(np.stack(W_ALG), dtype=torch.float32)
    W_TRUE = torch.tensor(np.stack(W_TRUE), dtype=torch.float32)
    R_TRUE = torch.tensor(np.stack(R_TRUE), dtype=torch.float32)
    R0 = torch.tensor(np.stack(R0), dtype=torch.float32)
    SAT = torch.tensor(np.stack(SAT)); INR = torch.tensor(np.stack(INR))
    n = len(X)
    perm = torch.randperm(n)
    ntr = int(0.8 * n)
    tr, va = perm[:ntr], perm[ntr:]
    print(f"windows: {n} (train {ntr}, val {n-ntr}), features {X.shape[-1]}, window {WINDOW}")

    net = EstimatorNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def losses(idx):
        x, wa, wt = X[idx], W_ALG[idx], W_TRUE[idx]
        s_hat = net(x, wa)
        sat, inr, rt, r0 = SAT[idx], INR[idx], R_TRUE[idx], R0[idx]
        l_rate = (((s_hat - wt) ** 2).sum(-1) * sat.any(-1))[sat.any(-1)].mean() if sat.any() else 0 * s_hat.sum()
        l_anchor = (((s_hat - wa) ** 2).sum(-1) * inr)[inr].mean() if inr.any() else 0 * s_hat.sum()
        R_hat = ins_rollout(s_hat, r0, DT)
        l_ins = tilt_error_deg(R_hat, rt).mean() / 30.0
        return l_rate + 0.3 * l_anchor + 0.02 * l_ins, l_rate, l_anchor, l_ins

    print("\n=== training ===")
    if a.load:
        net.load_state_dict(torch.load(a.load, map_location="cpu"))
        a.iters = 0
        print(f"  loaded {a.load} (training skipped)")
    for it in range(a.iters):
        idx = tr[torch.randint(0, len(tr), (a.batch,))]
        loss, l_rate, l_anchor, l_ins = losses(idx)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if it % 250 == 0 or it == a.iters - 1:
            with torch.no_grad():
                lv, lr_v, la_v, li_v = losses(va)
            print(f"  it{it:5d} train L={float(loss):8.3f} (rate {float(l_rate):7.3f} "
                  f"anchor {float(l_anchor):7.3f} ins {float(l_ins):6.3f}) | "
                  f"val L={float(lv):8.3f} rate {float(lr_v):7.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print(f"saved {a.out}")

    # ---- evaluation: rate error + INS tilt error, net vs the two baselines ----
    with torch.no_grad():
        s_hat = net(X[va], W_ALG[va])
        wt, wa = W_TRUE[va], W_ALG[va]
        sat = SAT[va].any(-1)
        e_net = ((s_hat - wt) ** 2).sum(-1)[sat].sqrt().mean()
        e_alg = ((wa - wt) ** 2).sum(-1)[sat].sqrt().mean()
        xclip = X[va][..., 3:6]                                   # the raw (clipped) gyro channel (features 3:6)
        e_c = ((xclip - wt) ** 2).sum(-1).sqrt()
        e_a = ((wa - wt) ** 2).sum(-1).sqrt()
        e_n = ((s_hat - wt) ** 2).sum(-1).sqrt()
        e_clip = e_c[sat].mean(); e_alg = e_a[sat].mean(); e_net = e_n[sat].mean()
        R_net = ins_rollout(s_hat, R0[va], DT)
        R_alg = ins_rollout(wa, R0[va], DT)
        R_cl = ins_rollout(xclip, R0[va], DT)
        t_net = tilt_error_deg(R_net, R_TRUE[va]).mean()
        t_alg = tilt_error_deg(R_alg, R_TRUE[va]).mean()
        t_cl = tilt_error_deg(R_cl, R_TRUE[va]).mean()
    print("\n=== held-out validation (saturated steps only) ===")
    print(f"  rate error   : clipped-gyro {float(e_clip):6.2f} | algebraic {float(e_alg):6.2f} | "
          f"network {float(e_net):6.2f} rad/s")
    print(f"  INS tilt err : clipped-gyro {float(t_cl):6.2f} | algebraic {float(t_alg):6.2f} | "
          f"network {float(t_net):6.2f} deg")

    # ---- breakdown by gyro range and by failure mask (frame level) ----
    meta = [META[i] for i in va.tolist()]
    H = SAT[va].shape[1]
    satf = SAT[va].any(-1).reshape(-1)
    ecf, eaf, enf = e_c.reshape(-1), e_a.reshape(-1), e_n.reshape(-1)
    metaf = np.repeat(np.array([m[1] for m in meta], float), H)
    flagf = np.repeat(np.array([m[0] for m in meta], float), H)
    inrf = INR[va].reshape(-1)
    print("\n  by gyro range (saturated frames):")
    print(f"    {'dps':>6} {'n':>7} {'clipped':>9} {'algebraic':>10} {'network':>9}")
    for dps in sorted(set(metaf.tolist())):
        m = satf & (metaf == dps)
        if int(m.sum()) == 0:
            continue
        print(f"    {dps:>6.0f} {int(m.sum()):>7} {float(ecf[m].mean()):>9.2f} "
              f"{float(eaf[m].mean()):>10.2f} {float(enf[m].mean()):>9.2f}")
    print("\n  by failure mask (saturated frames):")
    for flag in sorted(set(flagf.tolist())):
        m = satf & (flagf == flag)
        if int(m.sum()) == 0:
            continue
        print(f"    flag={int(flag)} n={int(m.sum()):>6}  clipped {float(ecf[m].mean()):>7.2f} | "
              f"algebraic {float(eaf[m].mean()):>7.2f} | network {float(enf[m].mean()):>7.2f} rad/s")
    inr = INR[va].reshape(-1)
    print(f"\n  in-range anchor check: |network - gyro| on in-range steps = "
          f"{float((((s_hat - xclip) ** 2).sum(-1).reshape(-1))[inr].sqrt().mean()) if inr.any() else float('nan'):.3f} rad/s")


if __name__ == "__main__":
    main()