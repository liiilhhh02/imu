"""Re-identify the priors stored in the shards (in place) with the corrected identifiers.

The 10^7 shards were collected before the accelerometer DC (`T/M` along body z) was plumbed into the
lever-arm identification, so their `prior` field is contaminated by a 9.8 m/s^2 unmodelled offset.
Everything needed for the fix (`gyro`, `accel`, `tom`, `u`, `mask_t`, `dps`, `dt`) is already in the
shard, so they can be repaired offline instead of re-collecting.

Run:  PYTHONPATH=<repo>:<me> python scripts/fix_priors.py --dir results/dr_1e7 [--workers 12]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.environ.get("GPD_REPO", "/home/liiil/Downloads/gym-pybullet-drones")
ME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, ME):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def fix_one(fn):
    try:
        from gpd_me.priors import identify_priors
        z = dict(np.load(fn, allow_pickle=True))
        prior, diag = identify_priors(
            np.asarray(z["gyro"], float), np.asarray(z["accel"], float), np.asarray(z["u"], float),
            np.asarray(z["omega"], float), np.asarray(z["mask_t"], float), float(z["dps"]),
            float(z["dt"]), tom=np.asarray(z["tom"], float))
        z["prior"] = prior.astype(np.float32)
        np.savez_compressed(fn, **z)
        return (float(diag["k"]), str(diag["how"]), float(prior[7]),
                float(diag.get("tom_err_rel", np.nan)), float(diag.get("G", np.nan)))
    except Exception as e:
        return np.nan, f"fail:{type(e).__name__}", np.nan, np.nan, np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ME, "results", "dr_1e7"))
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.npz")))
    print(f"re-identifying priors in {len(files)} shards with {a.workers} workers")
    with Pool(a.workers) as pool:
        res = list(pool.imap_unordered(fix_one, files, chunksize=8))
    how = {}
    for _k, h, _t, _e, _g in res:
        how[h.split("(")[0]] = how.get(h.split("(")[0], 0) + 1
    ks = np.array([k for k, _h, _t, _e, _g in res if np.isfinite(k)])
    taus = np.array([t for _k, _h, t, _e, _g in res if np.isfinite(t)])
    toms = np.array([e for _k, _h, _t, e, _g in res if np.isfinite(e)])
    print(f"  done. route counts: {how}")
    print(f"  identified k = |r_perp|: median {np.median(ks)*100:.3f} cm, "
          f"5-95% [{np.percentile(ks,5)*100:.3f}, {np.percentile(ks,95)*100:.3f}] cm")
    print(f"  identified tau        : median {np.median(taus)*1000:.2f} ms, "
          f"5-95% [{np.percentile(taus,5)*1000:.2f}, {np.percentile(taus,95)*1000:.2f}] ms")
    print(f"  command-driven T/M vs the simulated truth: median rel err {np.median(toms)*100:.3f} %")


if __name__ == "__main__":
    main()