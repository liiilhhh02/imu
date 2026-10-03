"""Empirical check of the yaw-mixer claim used to explain the failure-mode spin budget.

`MetaBaseAviary4._physics` applies ``z_torque = KM * (T0 - T1 + T2 - T3)``.  With ``J_zz = 0.007``
and ``KM = 0.01`` we therefore predict ``d(wz)/dt = tau_z / J_zz``.  Driving constant thrusts through
the real plant and measuring the resulting spin-up rate closes the loop on the analysis that
"two rotors on the same yaw sign can produce twice the single-rotor torque".

Run:  PYTHONPATH=<repo>:<me> python scripts/verify_mixer.py
"""
import os
import sys

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402

J_ZZ, KM = 0.007, 0.01
SETTLE = 200          # the actuator lag (delay=0.026 s) has settled by then


def measure(mask, action, steps=400, verbose=False):
    env = MetaAviaryFaulty(drone_model=DroneModel.CF2X, num_drones=1,
                           initial_xyzs=np.array([[0.0, 0.0, 1.0]]), physics=Physics("pyb"),
                           aggregate_phy_steps=1, freq=200, gui=False, record=False, obstacles=False,
                           rate_source="truth",
                           imu_cfg=IMUConfig(gyro_range_dps=1e6, lever_arm=(-0.012, -0.0055, 0.0)))
    env.eval = False
    env.reset()
    env.shut_down = np.array(mask, dtype=float)
    w, thr = [], []
    for _ in range(steps):
        env._computeObs()
        env.step(np.array(action, dtype=float))
        w.append(env.omega_true.copy())
        thr.append(np.asarray(env.thrust[0], float).copy())
    dt = env.TIMESTEP * env.AGGR_PHY_STEPS
    w = np.array(w)
    thr = np.array(thr)
    if verbose:
        print("      t[s]  wz      wx      wy      thrusts")
        for i in range(100, min(steps, 260), 40):
            print(f"      {i*dt:4.2f} {w[i,2]:7.2f} {w[i,0]:7.2f} {w[i,1]:7.2f}  {np.round(thr[i],2)}")
    env.close()
    i0, i1 = 150, 250                      # after the actuator lag, before damping dominates
    return (w[i1, 2] - w[i0, 2]) / ((i1 - i0) * dt), w, thr


def main():
    cases = [
        ("flag1 T1=T3=15 N   -> tau = -.30", [0, 1, 0, 1], [-1, 1, -1, 1]),
        ("flag1 T1=T3=9.8 N  -> tau = -.098", [0, 1, 0, 1], [-1, 0.30667, -1, 0.30667]),
        ("flag2 T3=15 N      -> tau = -.15", [0, 0, 0, 1], [-1, -1, -1, 1]),
        ("flag2 T3=9.8 N     -> tau = -.098", [0, 0, 0, 1], [-1, -1, -1, 0.30667]),
    ]
    print(f"{'case':<36} {'tau [N*m] (code)':>17} {'meas dwz/dt':>12} "
          f"{'max |w_xy| [rad/s]':>19} {'wz @1.1 s':>11}")
    for name, mask, act in cases:
        meas, w, thr = measure(mask, act)
        T = np.clip((np.array(act) + 1) * 7.5, 0, 15) * np.array(mask)
        tau = KM * (T[0] - T[1] + T[2] - T[3])
        wxy = float(np.abs(w[:, :2]).max())
        print(f"{name:<36} {tau:>17.3f} {meas:>12.1f} {wxy:>19.2f} {w[220, 2]:>11.2f}"
              f"{'   <-- pure spin, no tumbling' if wxy < 1e-6 else ''}")


if __name__ == "__main__":
    main()