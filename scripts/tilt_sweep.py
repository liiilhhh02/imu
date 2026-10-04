"""Closed-loop A/B sweep for the thrust-axis tilt observer -- the "What the A/B should report" section.

This is `scripts/tilt_check.py`'s machinery with the report fixed, per `review/opus_tilt_review.txt`:

  * survival (`hold` = first step with z < 0.3, else `steps`) is the PRIMARY endpoint, reported as
    the PAIRED per-seed delta `hold(arm) - hold(ins)` plus a Wilcoxon signed-rank p over seeds;
  * the tilting window is censored at `min(hold_arm, hold_ins)`, because averaging past the crash
    mixes free flight with post-crash ground tumbling (where the observer rejects everything) and
    makes arms with different `hold` incomparable; `first300` (1.5 s, both arms still airborne) is
    the clean headline;
  * a yaw/heading error column `rotvec(R_true.T @ R_est) . e_z` (degrees), because `AttitudeINS.yaw_error`
    returns the total angle, not the rotation about the thrust axis;
  * the innovation sequence (`tilt_err_used`) and the split rejection counts are surfaced;
  * `tau` is swept; a degraded-velocity arm and a perfect-measurement ceiling arm separate "the
    measurement is bad" from "the blend is too slow".

Arms, all flown from the SAME fault draw per seed (`identification()` is called once per seed, outside
the arm loop -- the same post-fault state, IMU noise stream and commanded sequence):

    ins                     baseline: INS (net rate) only, no observer
    tilt@{tau}              observer, measurement derived from the true-delay velocity, half-step aligned
    tilt-noalign@{tau}      same, but WITHOUT the omega half-step alignment -> measures its benefit
    oracle-att              DIAGNOSTIC: true attitude + the estimator's rate (a non-deployable ceiling)
    meas-ceil               DIAGNOSTIC: observer fed z_meas = R_true[:,2] ("the measurement made perfect")
    veldeg                  observer fed a white-noised, first-order-filtered velocity (deployability check)

    $PY scripts/tilt_sweep.py --seeds 3 --steps 800 --taus 0.02,0.1
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
from scipy.stats import wilcoxon  # noqa: E402

from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402

from gpd_me.e2e import NetRate, deploy_obs, WINDOW  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.ins import AttitudeINS  # noqa: E402
from gpd_me.policy import PositionPID, load_policy, quat_to_matrix  # noqa: E402
from gpd_me.tilt import TiltObserver  # noqa: E402
from e4_closed_loop import DT, LEVER, identification  # noqa: E402

MASK = np.array([0.0, 0.0, 1.0, 1.0])
Z_UP = np.array([0.0, 0.0, 1.0])


class MeasCeilObserver(TiltObserver):
    """Diagnostic ceiling: the observer's measurement is made perfect, `z_meas = R_true[:, 2]`.

    The blend, `tau`, magnitude weighting and rejection logic are the parent's; only the measured
    direction is replaced.  The true thrust axis is already timestamped at the current instant, so
    the half-step alignment is NOT applied (doing so would rotate a correct sample out of time).
    """

    def correct(self, R_ins, vel, dt, omega=None, z_true=None):
        v = np.asarray(vel, float).ravel()
        if self.v_prev is None:
            self.v_prev = v.copy()
            return R_ins
        if z_true is None:
            return super().correct(R_ins, vel, dt, omega)
        # The real velocity still defines the derivative stream `v_prev`; only the sample handed to
        # the parent is synthesised so that it derives exactly `n * z_true`.  This keeps the
        # magnitude weighting / gating identical to the real arm and only perfects the direction.
        a_i = (v - self.v_prev) / dt
        n = float(np.linalg.norm(a_i + Z_UP * self.g))
        t_syn = n * np.asarray(z_true, float).ravel()
        v_syn = self.v_prev + dt * (t_syn - Z_UP * self.g)
        self.v_prev = v.copy()
        out = super().correct(R_ins, v_syn, dt, None)
        self.v_prev = v.copy()          # undo the parent's internal advance to the synthesised sample
        return out


def parse_arm(arm):
    """'tilt@0.05' -> ('tilt', 0.05); 'tilt-noalign@0.05' -> ('tilt-noalign', 0.05); else (arm, None)."""
    if "@" in arm:
        head, tau = arm.split("@")
        return head, float(tau)
    return arm, None


def fly(seed, steps, ckpt, dps, kind, tau, prior_pool, vel_noise, vel_tau,
        rl_name=None, hold_ins=None):
    """One closed-loop flight for one arm.

    Machinery is byte-identical to `scripts/tilt_check.py:fly` (same env construction, same
    `preseed`, same control loop / 0.26 rad tilt clamp / `deploy_obs` / `hold` definition); the
    extras are the arm-specific observer input and the extra logged diagnostics.
    """
    use_tilt = kind in ("tilt", "tilt-noalign", "veldeg", "meas-ceil")
    oracle_att = kind == "oracle-att"
    prior, diag, fault_rng, ic, pre = prior_pool
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200, gui=False, record=False,
        obstacles=False, imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=LEVER, seed=seed),
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
    if kind == "meas-ceil":
        tob = MeasCeilObserver(tau=tau)
    elif use_tilt:
        tob = TiltObserver(tau=tau)
    else:
        tob = None
    noise_rng = np.random.default_rng(seed * 1000003 + 7919)
    vfilt = None                       # veldeg first-order filter state
    last = -np.ones(4)
    tp = np.array([0.0, 0.0, 1.0])
    tilt_err, yaw_err, meas_err, innov = [], [], [], []
    hold = steps
    for t in range(steps):
        env._computeObs()
        gyro, accel = env.gyro_meas.copy(), env.accel_meas.copy()
        tom = float(env.thrust_over_mass)
        u_cmd = np.clip((last + 1.0) * 7.5, 0.0, 15.0)
        rate = net.step(accel, gyro, u_cmd, env.shut_down)
        ins.update(rate, DT)
        if tob is not None:
            if kind == "veldeg":
                # the sim hands over a perfect world velocity; a real outer loop supplies a filtered
                # estimate, so this arm is the deployability check (review sec. 3, "truth velocity").
                noisy = env.vel[0] + noise_rng.normal(0.0, vel_noise, 3)
                vfilt = noisy.copy() if vfilt is None else vfilt + (DT / (vel_tau + DT)) * (noisy - vfilt)
                vel_obs = vfilt
            else:
                vel_obs = env.vel[0]
            if kind == "meas-ceil":
                R_true_now = quat_to_matrix(env.quat[0])
                ins.R = tob.correct(ins.R, vel_obs, DT, z_true=R_true_now[:, 2])
            elif kind == "tilt-noalign":
                ins.R = tob.correct(ins.R, vel_obs, DT)             # deliberately no omega
            else:
                ins.R = tob.correct(ins.R, vel_obs, DT, omega=rate)  # half-step measurement alignment
        R = quat_to_matrix(env.quat[0]) if oracle_att else ins.R
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
        # heading error about the thrust axis: rotvec in the (true) body frame, z component.  NOT
        # AttitudeINS.yaw_error, which returns the *total* rotation angle (review sec. 2).
        yaw_err.append(float(np.rad2deg(Rotation.from_matrix(R_true.T @ R).as_rotvec()[2])))
        if tob is not None and tob.z_meas is not None:
            # the measurement's own error: if this is not near zero the observer's premise is wrong
            cm = float(np.clip(tob.z_meas @ R_true[:, 2], -1.0, 1.0))
            meas_err.append(float(np.rad2deg(np.arccos(cm))))
        if hold == steps and env.pos[0][2] < 0.3:
            hold = t
    if tob is not None:
        innov = np.rad2deg(np.asarray(tob.tilt_err_used, float))
    env.close()
    return dict(
        tilt=np.asarray(tilt_err, float), yaw=np.asarray(yaw_err, float),
        meas=np.asarray(meas_err, float) if meas_err else np.asarray([float("nan")]),
        innov=innov, hold=int(hold),
        used=int(tob.n_used) if tob is not None else 0,
        rej=int(tob.n_rejected) if tob is not None else 0,
        rej_vclip=int(tob.n_rejected_vclip) if tob is not None else 0,
        rej_thrust=int(tob.n_rejected_thrust) if tob is not None else 0)


def rad_stats(a, mask=None):
    if mask is not None:
        a = a[mask]
    return (float(a.mean()), float(np.percentile(a, 90))) if a.size else (float("nan"), float("nan"))


def arm_list(taus):
    return (["ins"] + [f"tilt@{t}" for t in taus] + [f"tilt-noalign@{taus[0]}"]
            + ["oracle-att", "meas-ceil", "veldeg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ckpt", default=os.path.join(ME, "results", "e2e_v12.pt"))
    ap.add_argument("--dps", type=float, default=1000.0)
    ap.add_argument("--taus", default="0.02,0.05,0.1",
                    help="comma-separated tau values for the tilt@tau arms; the review notes the "
                         "design point tilt_err ~ tau*omega_err, so one value proves nothing.")
    ap.add_argument("--vel_noise", type=float, default=0.05,
                    help="white-noise sigma (m/s) added to the velocity the veldeg observer sees")
    ap.add_argument("--vel_tau", type=float, default=0.05,
                    help="first-order filter time constant (s) on that velocity")
    ap.add_argument("--rl_name", default=None)
    a = ap.parse_args()
    taus = [float(t) for t in a.taus.split(",") if t.strip()]
    arms = arm_list(taus)

    print(f"# tilt_sweep  seeds={a.seeds} steps={a.steps} dps={a.dps} taus={taus} "
          f"vel_noise={a.vel_noise} vel_tau={a.vel_tau} ckpt={os.path.basename(a.ckpt)}")
    hdr = (f"{'seed':>4} {'arm':>17} {'tilt_win':>9} {'yaw_win':>8} {'tilt300':>8} {'zmeas300':>9} "
           f"{'hold':>6} {'dhold':>6} {'used':>6} {'rej':>5} {'vclip':>6} {'thr':>5} "
           f"{'inno_mu':>8} {'inno_p90':>9}")
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for s in range(a.seeds):
        pool = identification(seed=s, dps=a.dps, flag=3)
        seed_res = {}
        for arm in arms:
            kind, tau = parse_arm(arm)
            if tau is None:
                tau = taus[0]          # meas-ceil / veldeg have no @tau; use the first swept value
            hold_ins = seed_res["ins"]["hold"] if "ins" in seed_res else None
            r = fly(s, a.steps, a.ckpt, a.dps, kind, tau, pool, a.vel_noise, a.vel_tau,
                    a.rl_name, hold_ins)
            w = min(r["hold"], hold_ins) if hold_ins is not None else r["hold"]
            r["win"] = int(w)
            r["tilt_win"] = float(r["tilt"][:w].mean()) if w > 0 else float("nan")
            r["yaw_win"] = float(r["yaw"][:w].mean()) if w > 0 else float("nan")
            r["tilt300"] = float(r["tilt"][:300].mean())
            r["zmeas300"] = float(r["meas"][:300].mean())
            r["inno_mu"], r["inno_p90"] = rad_stats(np.asarray(r["innov"], float))
            seed_res[arm] = r
            r["dhold"] = r["hold"] - seed_res["ins"]["hold"]
            print(f"{s:4d} {arm:>17} {r['tilt_win']:9.2f} {r['yaw_win']:8.2f} {r['tilt300']:8.2f} "
                  f"{r['zmeas300']:9.2f} {r['hold']:6d} {r['dhold']:6d} {r['used']:6d} "
                  f"{r['rej']:5d} {r['rej_vclip']:6d} {r['rej_thrust']:5d} "
                  f"{r['inno_mu']:8.2f} {r['inno_p90']:9.2f}", flush=True)
        results[s] = seed_res
        print("-" * len(hdr), flush=True)

    print(f"\n# paired summary over {a.seeds} seeds ({os.path.basename(a.ckpt)}); "
          f"delta = hold(arm) - hold(ins), + is better")
    shdr = (f"{'arm':>17} {'mean_dhold':>10} {'med_dhold':>9} {'wilcoxon_p':>11} "
            f"{'n_lt':>5} {'n_eq':>5} {'n_gt':>5} {'mean_tilt300':>12} {'mean_zmeas300':>13}")
    print(shdr)
    print("-" * len(shdr))
    for arm in arms:
        dl = np.array([results[s][arm]["hold"] - results[s]["ins"]["hold"] for s in range(a.seeds)],
                      float)
        try:
            _, pval = wilcoxon(dl) if not np.all(dl == 0) else (float("nan"), float("nan"))
        except ValueError:
            pval = float("nan")
        n_lt = int((dl < 0).sum()); n_eq = int((dl == 0).sum()); n_gt = int((dl > 0).sum())
        t300 = float(np.mean([results[s][arm]["tilt300"] for s in range(a.seeds)]))
        zm = float(np.mean([results[s][arm]["zmeas300"] for s in range(a.seeds)]))
        print(f"{arm:>17} {dl.mean():10.1f} {float(np.median(dl)):9.1f} {pval:11.4f} "
              f"{n_lt:5d} {n_eq:5d} {n_gt:5d} {t300:12.2f} {zm:13.2f}")

    # implausibility flags: a diagnostic ceiling that under-performs the ablation above it means the
    # comparison is confounded, not that the observer helped.
    mh = {arm: np.mean([results[s][arm]["hold"] for s in range(a.seeds)]) for arm in arms}
    if mh["meas-ceil"] < mh["oracle-att"]:
        print(f"\n# IMPLAUSIBLE: meas-ceil mean hold {mh['meas-ceil']:.1f} < oracle-att "
              f"{mh['oracle-att']:.1f} -- the perfect-measurement ceiling should not be worse than "
              f"the attitude ceiling.")
    if mh["veldeg"] > mh["oracle-att"]:
        print(f"\n# NOTE: veldeg mean hold {mh['veldeg']:.1f} exceeds oracle-att "
              f"{mh['oracle-att']:.1f} -- the degraded-velocity arm is not limited by velocity noise.")


if __name__ == "__main__":
    main()