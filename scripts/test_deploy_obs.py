"""Assert `gpd_me.e2e.deploy_obs` is bit-identical to the acceptance harness's inline obs.

Throwaway guard, run before any fine-tuning: the training rollout and scripts/e4_closed_loop.py must
agree field by field, otherwise a fine-tuned policy cannot transfer and the failure looks like
"the policy cannot learn the estimator's error".
"""
import os
import sys

import numpy as np

sys.path[:0] = [os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones"),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
from gpd_me.e2e import deploy_obs  # noqa: E402

rng = np.random.default_rng(0)
for trial in range(200):
    rel = rng.normal(0, 0.3, 2)
    rate = rng.normal(0, 40, 3)
    ta = float(rng.uniform(6, 15))
    tom = float(rng.uniform(6, 15))
    last_action = rng.uniform(-1, 1, 4)
    mask = rng.integers(0, 2, 4).astype(float)
    # the harness's inline form (scripts/e4_closed_loop.py, the `obs = np.array([...])` line)
    ref = np.array([rel[0], rel[1],
                    rate[0] / 10, rate[1] / 10, rate[2] / 50,
                    (ta - 9.8) / 3, (tom - 9.8) / 3,
                    *(last_action * mask), *(mask * 2 - 1)], dtype=np.float32)
    got = deploy_obs(rel, rate, ta, tom, last_action, mask)
    assert got.dtype == ref.dtype == np.float32, (got.dtype, ref.dtype)
    assert got.shape == ref.shape == (15,), (got.shape, ref.shape)
    assert np.array_equal(got, ref), (trial, got, ref)
print("test_deploy_obs: bit-identical over 200 random trials, 15-D float32")
