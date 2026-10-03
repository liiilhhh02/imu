"""Verify the identified priors against the ground truth stored in the shards.

This is the honesty check for the whole identification chain: the shards carry the *true* airframe
parameters (`true = [lever(3), M_applied, KM, delay, J(3), damping, M_requested]`), so every prior
can be scored against the quantity it is supposed to estimate -- without any of them being used to
produce the estimate.

Reported per shard:
  tau      identified actuator lag vs the true `delay`                   (was never identified before)
  T/M      command-based specific thrust vs the simulated one            (BLOCKER 3)
  k        identified |r_perp| vs the true |r_perp| on the spin axis      (BLOCKER 4)
  r_perp   identified lever arm vs the true one, perpendicular part       (axial part is unobservable)
  g_T      hover trim vs 1/M
  lag      per-flight |mean_t err| of the algebraic estimate vs the plain clip, on saturated frames

Run:  PYTHONPATH=<repo>:<me> python scripts/verify_priors.py [--dir results/dr_1e7] [--n 240]
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


def _mean_dir(omega, sat):
    m = np.asarray(sat, bool).any(axis=1)
    if not m.any():
        m = np.ones(len(omega), bool)
    v = np.asarray(omega, float)[m].mean(axis=0)
    nn = float(np.linalg.norm(v))
    return (v / nn) if nn > 1e-9 else None


def one(fn):
    try:
        from gpd_me.e2e import algebraic_estimate
        from gpd_me.priors import identify_priors, tom_from_command
        z = dict(np.load(fn, allow_pickle=True))
        gyro = np.asarray(z["gyro"], float); accel = np.asarray(z["accel"], float)
        u = np.asarray(z["u"], float); omega = np.asarray(z["omega"], float)
        mask_t = np.asarray(z["mask_t"], float); sat = np.asarray(z["sat"], bool)
        dps = float(z["dps"]); dt = float(z["dt"]); lim = np.deg2rad(dps)
        prior, diag = identify_priors(gyro, accel, u, omega, mask_t, dps, dt,
                                      tom=np.asarray(z["tom"], float))
        tr = z["true"][0]
        r_true = np.asarray(tr[0], float); M_true = float(tr[1]); tau_true = float(tr[3])
        d = _mean_dir(omega, sat)
        if d is None:
            return None
        perp = lambda r: r - (r @ d) * d
        k_true = float(np.linalg.norm(perp(r_true)))
        r_hat = np.asarray(prior[:3], float)
        tom_hat = tom_from_command(u, mask_t, dt, float(prior[3]), float(prior[7]))
        s = accel - np.array([0.0, 0.0, 1.0])[None, :] * tom_hat[:, None]
        w_alg = algebraic_estimate(gyro, s, float(prior[8]), sat, lim)
        sm = sat.any(axis=1)
        lag = lambda w: float(np.linalg.norm((w - omega)[sm].mean(0))) if sm.any() else np.nan
        return dict(dps=dps, flag=float(z["flag"]), how=diag["how"], torque_ok=diag["torque_ok"],
                    tau_hat=float(prior[7]), tau_true=tau_true,
                    tau_err=100.0 * (float(prior[7]) - tau_true) / max(tau_true, 1e-6),
                    tom_err=float(diag.get("tom_err_rel", np.nan)),
                    k_hat=float(prior[8]), k_true=k_true,
                    k_err=100.0 * (float(prior[8]) - k_true) / max(k_true, 1e-9),
                    r_perp_err=float(np.linalg.norm(perp(r_hat) - perp(r_true))) * 1e3,
                    r_err=float(np.linalg.norm(r_hat - r_true)) * 1e3,
                    gT_err=100.0 * (float(prior[3]) - 1.0 / M_true) * M_true,
                    lag_alg=lag(w_alg), lag_clip=lag(gyro), n_sat=int(sm.sum()))
    except Exception as e:
        import traceback
        if os.environ.get("VP_DEBUG"):
            traceback.print_exc()
        return dict(err=f"{type(e).__name__}: {e}")


def band(dps):
    return "100-300" if dps <= 300 else ("400-1000" if dps <= 1000 else "1500-6000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ME, "results", "dr_1e7"))
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.npz")))
    step = max(1, len(files) // a.n)
    files = files[::step][:a.n]
    print(f"verifying priors on {len(files)} shards of {a.dir} with {a.workers} workers\n")
    with Pool(a.workers) as pool:
        res = [r for r in pool.imap_unordered(one, files, chunksize=2) if r]
    bad = [r for r in res if "err" in r]
    res = [r for r in res if "err" not in r]
    if bad:
        print(f"  {len(bad)} shards failed, e.g. {bad[0]['err']}\n")

    def pct(v, p=50):
        v = np.asarray([x for x in v if np.isfinite(x)], float)
        return float(np.percentile(v, p)) if len(v) else float("nan")

    print(f"{'metric':>28} {'median':>9} {'5%':>9} {'95%':>9}   n")
    rows = [("tau err (%)", [r["tau_err"] for r in res]),
            ("T/M rel err", [r["tom_err"] for r in res]),
            ("k = |r_perp| err (%)", [r["k_err"] for r in res]),
            ("|r_perp| err (mm)", [r["r_perp_err"] for r in res]),
            ("|r| err (mm, incl. axial)", [r["r_err"] for r in res]),
            ("g_T err (%)", [r["gT_err"] for r in res]),
            ("lag algebraic (rad/s)", [r["lag_alg"] for r in res]),
            ("lag clipped (rad/s)", [r["lag_clip"] for r in res])]
    for name, v in rows:
        print(f"{name:>28} {pct(v):9.3f} {pct(v,5):9.3f} {pct(v,95):9.3f}   {len(v)}")

    print("\nby gyro range (this is where the low-range priors used to collapse):")
    print(f"{'band':>10} {'n':>4} {'tau%':>7} {'T/M':>7} {'k%':>7} {'r_perp mm':>10} "
          f"{'lag_alg':>8} {'lag_clip':>9}")
    for b in ("100-300", "400-1000", "1500-6000"):
        s = [r for r in res if band(r["dps"]) == b]
        if not s:
            continue
        print(f"{b:>10} {len(s):>4} {pct([r['tau_err'] for r in s]):7.1f} "
              f"{pct([r['tom_err'] for r in s]):7.4f} {pct([r['k_err'] for r in s]):7.1f} "
              f"{pct([r['r_perp_err'] for r in s]):10.2f} "
              f"{pct([r['lag_alg'] for r in s]):8.2f} {pct([r['lag_clip'] for r in s]):9.2f}")

    print("\nroutes:")
    how = {}
    for r in res:
        key = r["how"].split("(")[0]
        how[key] = how.get(key, 0) + 1
    for k, v in sorted(how.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:4d}  ({100*v/len(res):4.1f} %)")
    print(f"  torque_ok identified         {sum(1 for r in res if r['torque_ok']):4d}"
          f"  ({100*sum(1 for r in res if r['torque_ok'])/len(res):4.1f} %)")


if __name__ == "__main__":
    main()