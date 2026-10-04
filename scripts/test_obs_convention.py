"""Permanent guard: `deploy_obs` must agree with the *environment's own* observation assembly.

The external audit (review/opus_audit_2026-10-04.md, section 3a) pointed out that
`scripts/test_deploy_obs.py` is weak: it compares `deploy_obs` against a re-typed copy of the same
formula, so it can never notice the harness drifting away from the environment's convention.  This
test closes that hole by comparing against `MetaAviaryFaulty._computeObs()` itself, which is the
convention `scripts/train_robust.py` trains on and which the senior's checkpoints were trained with.

Two documented differences are expected and asserted explicitly, so that a *new* difference fails:
  * dim 6 (`(tom-9.8)/3`): the env fills it from `self.last_acc` (sampled at the top of
    `_computeObs`) while `deploy_obs` is given `env.thrust_over_mass` (refreshed by `_imu_step`,
    i.e. one step fresher).  Asserted to within one step's worth of change.
  * everything else must match exactly.

Run:  $PY scripts/test_obs_convention.py
"""
import os
import sys

import numpy as np

sys.path[:0] = [os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402
from gpd_me.e2e import deploy_obs  # noqa: E402
from gpd_me.env_faulty import MetaAviaryFaulty  # noqa: E402
from gpd_me.imu import IMUConfig  # noqa: E402

MASK3 = np.array([0.0, 0.0, 1.0, 1.0])
LEVER = (-0.012, -0.0055, 0.0)          # scripts/e4_closed_loop.py:LEVER


def main():
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1, freq=200, gui=False, record=False,
        obstacles=False, imu_cfg=IMUConfig(gyro_range_dps=1000.0, lever_arm=LEVER, seed=3),
        rate_source="override", att_source="ins")
    env.eval = False
    env.rate_override = np.zeros(3)     # reset() ends in _computeObs, so the override must exist
    env.reset()
    env.shut_down = MASK3.copy()
    rng = np.random.default_rng(0)
    worst_exact, worst_dim6 = 0.0, 0.0
    for trial in range(60):
        rate = rng.normal(0.0, 30.0, 3)
        ta = float(rng.uniform(7.0, 13.0))
        zb = rng.normal(0.0, 0.1, 3)
        zb = zb / np.linalg.norm(zb)
        last = rng.uniform(-1.0, 1.0, 4)
        env.rate_override = rate
        env.target_a, env.target_z_body = ta, zb
        env.last_action = last.copy()          # the env's dims 7:11 read this
        got = np.asarray(env._computeObs(), float)
        ref = np.asarray(deploy_obs(env.ins.R.T @ zb, rate, ta, env.thrust_over_mass, last,
                                    env.shut_down), float)
        d = np.abs(got - ref)
        worst_dim6 = max(worst_dim6, float(d[6]))
        d[6] = 0.0
        # dims 0:2: the env recomputes des_rad from the INS it just advanced; deploy_obs is handed the
        # same R, so these must agree to float32 precision.
        worst_exact = max(worst_exact, float(d.max()))
    env.close()
    print(f"max |env - deploy_obs| excluding dim 6: {worst_exact:.3e}")
    print(f"dim 6 (expected one-step T/M difference): {worst_dim6:.3e}  (tolerance 0.1)")
    assert worst_exact < 1e-5, f"observation convention drifted: {worst_exact}"
    assert worst_dim6 < 0.1, f"dim 6 differs by more than one step of thrust change: {worst_dim6}"
    print("test_obs_convention: PASS (env convention == deploy_obs, one documented one-step dim 6)")


if __name__ == "__main__":
    main()