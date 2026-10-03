"""Failure-mode study: what spin rate does each mask produce, and which gyro axes saturate?

Two questions from the user, answered with measurements:
  Q1 why does a *single* rotor failure tolerate gyro saturation?
  Q2 does a *two* rotor failure give the highest spin (the user wants a high spin)?

Part A flies each mask with the senior's policies (ground-truth rate, outer position PID) and
reports z, |w|, the per-axis true rates and which axes exceed 1000 dps.
Part B measures the *yaw torque budget*: the applied z-torque is
``KM * (T0 - T1 + T2 - T3)`` (see `MetaBaseAviary4._physics`), so for a mask we can compute both the
torque under the hover constraint (sum T = M g) and the maximum achievable torque (each rotor <= 15 N).

Run:  PYTHONPATH=<repo>:<me> python scripts/study_failure_modes.py
"""
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from gym_pybullet_drones.algo.ACRL import ACRL  # noqa: E402
from gym_pybullet_drones.control.RLAttitudeControl import RLControl  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402
from gpd_me.policy import ACRLArgs, PROP_XY  # noqa: E402

MASK = {0: [0, 1, 1, 1], 1: [0, 1, 0, 1], 2: [0, 0, 0, 1], 3: [0, 0, 1, 1]}
# yaw sign of each rotor as used by MetaBaseAviary4._physics: z_torque = KM*(T0 - T1 + T2 - T3)
YAW_SIGN = np.array([+1, -1, +1, -1])
CKPTS = {0: ["shutdown_real_7"], 1: ["shutdown_real_7", "shutdown_real_7_4", "shutdown_real_7_3"],
         2: ["shutdown_real_7_4"], 3: ["shutdown_real_7_4", "shutdown_real_7_3"]}
MODEL_DIR = os.path.join(REPO, "gym_pybullet_drones", "model")


def load(ck):
    dev = torch.device("cpu")
    _o = torch.load
    torch.load = lambda *a, **k: _o(*a, **{**k, "map_location": k.get("map_location", dev)})
    p = ACRL(state_dim=15, action_dim=4, max_action=1, device=dev, args=ACRLArgs())
    p.load(ck, MODEL_DIR)
    return p


def fly(flag, ck, dps=1000.0, steps=1600, spin_init=None, seed=0):
    policy = load(ck)
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200,
        gui=False, record=False, obstacles=False, rate_source="truth",
        imu_cfg=IMUConfig(gyro_range_dps=dps, lever_arm=(-0.012, -0.0055, 0.0),
                          gyro_noise_std=0.05, accel_noise_std=0.02))
    env.eval = False
    np.random.seed(seed)
    env.shut_down_rotors(flag)
    if spin_init is not None:
        import pybullet as p
        p.resetBaseVelocity(objectUniqueId=env.DRONE_IDS[0], linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, -float(spin_init)], physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
    ctrl = RLControl(DroneModel.CF2X)
    ctrl.reset()
    tp = np.array([0.0, 0.0, 1.0])
    z, wt, tom = [], [], []
    for _ in range(steps):
        raw = env._getDroneStateVector(0)
        ta, tz = ctrl.RLShutDownControl(control_timestep=env.TIMESTEP, cur_pos=raw[0:3],
                                        cur_quat=raw[3:7], cur_vel=raw[10:13], target_pos=tp)
        r = np.hypot(tz[0], tz[1])
        if r > 0.26:
            s = 0.26 / r
            tz = np.array([tz[0] * s, tz[1] * s, np.sqrt(1 - (tz[0] * s) ** 2 - (tz[1] * s) ** 2)])
        env.target_a, env.target_z_body = float(ta), tz
        obs = env._computeObs()
        env.step(policy.select_action(obs, deterministic=True))
        z.append(float(env.pos[0][2])); wt.append(env.omega_true.copy())
        tom.append(np.asarray(env.thrust[0], float).copy())
    km, m = float(env.KM), float(env.M)
    env.close()
    wt = np.array(wt); z = np.array(z); w = len(z) // 5
    inst = np.linalg.norm(wt[-w:], axis=1)
    ang = np.rad2deg(np.arccos(np.clip(np.abs(wt[-w:, 2]) / np.maximum(inst, 1e-9), -1, 1)))
    return dict(z=float(z[-w:].mean()), zstd=float(z[-w:].std()), w=wt[-w:].mean(axis=0),
                wmag=float(inst.mean()), ang=float(ang.mean()),
                steadiness=float(np.linalg.norm(wt[-w:].mean(axis=0)) / max(inst.mean(), 1e-9)),
                T=np.array(tom[-w:]).mean(axis=0), km=km, m=m)


def torque_budget(mask, km, mg):
    """(hover-constrained, maximum) z-torque for a mask, in N*m."""
    s = YAW_SIGN[np.array(mask) == 1]
    n = len(s)
    T = np.full(n, mg / n)
    tau_hover = km * float(s @ T)
    # maximum |sum(**sign_i * T_i)| over 0 <= T_i <= 15: put every rotor that shares one sign at 15
    tau_max = km * 15.0 * max(int((s > 0).sum()), int((s < 0).sum()))
    return tau_hover, tau_max


def main():
    print("=== Part A: what the senior's policies actually do (ground-truth rate, 1000 dps) ===")
    print(f"{'flag':>4} {'mask':<12} {'ckpt':<19} {'z [m]':>8} {'|w| [rad/s]':>12} {'wz':>8} "
          f"{'wx':>7} {'wy':>7} {'ang(w,z)':>9} {'steady':>7}  sat")
    rows = {}
    for flag, cks in CKPTS.items():
        for ck in cks:
            for seed in ((0, 1, 2) if flag == 1 else (0,)):
                try:
                    r = fly(flag, ck, seed=seed)
                except Exception as e:
                    print(f"{flag:>4} {str(MASK[flag]):<12} {ck:<19}  failed: {e}")
                    continue
                sat = [i for i in range(3) if abs(r["w"][i]) > np.deg2rad(1000.0)]
                rows[(flag, ck)] = r
                w = r["w"]
                print(f"{flag:>4} {str(MASK[flag]):<12} {ck:<19} {r['z']:>8.2f} {r['wmag']:>12.1f} "
                      f"{w[2]:>8.2f} {w[0]:>7.2f} {w[1]:>7.2f} {r['ang']:>8.1f} "
                      f"{r['steadiness']:>6.2f}  {''.join('xyz'[i] for i in sat) or '-'}"
                      f"  seed={seed}")
    print()
    print("=== Part B: yaw-torque budget per mask (z_torque = KM*(T0-T1+T2-T3)) ===")
    km, m = 0.01, 1.0
    mg = m * 9.81
    print(f"{'flag':>4} {'mask':<12} {'alive rots':<12} {'yaw signs':<12} "
          f"{'tau @hover [N*m]':>17} {'tau max [N*m]':>14} {'spin ratio vs 1 rotor':>22}")
    tau_single = km * 15.0          # 1 rotor, zero the other three
    for flag, mask in MASK.items():
        alive = [i for i in range(4) if mask[i] == 1]
        th, tm = torque_budget(mask, km, mg)
        print(f"{flag:>4} {str(mask):<12} {str(alive):<12} {str(YAW_SIGN[alive].tolist()):<12} "
              f"{th:>17.4f} {tm:>14.4f} {tm/tau_single:>21.2f}x")
    geometry()


def geometry():
    """How badly does clipping one axis distort the *direction* of the body-rate vector?"""
    print()
    print("=== Part C: direction distortion of w caused by clipping at 1000 dps ===")
    lim = np.deg2rad(1000.0)
    print(f"{'mask':<12} {'w [rad/s]':<26} {'|w|':>6} {'ang(w,z)':>9} {'|w_clip|':>9} "
          f"{'direction err':>14}")
    for flag, mask in MASK.items():
        w = {0: (-1.66, -0.64, -25.33), 1: (-8.26, 7.82, -16.92),
             2: (-11.94, 11.28, -41.77), 3: (-10.61, 13.52, -38.45)}[flag]
        w = np.array(w)
        wc = np.clip(w, -lim, lim)
        n, nc = np.linalg.norm(w), np.linalg.norm(wc)
        ang = np.rad2deg(np.arccos(np.clip(abs(w[2]) / n, -1, 1)))
        derr = np.rad2deg(np.arccos(np.clip((w @ wc) / (n * nc), -1, 1)))
        print(f"{str(mask):<12} {str(np.round(w, 2)):<26} {n:>6.1f} {ang:>9.1f} {nc:>9.1f} "
              f"{derr:>14.1f}")


if __name__ == "__main__":
    main()