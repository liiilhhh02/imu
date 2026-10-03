"""Attitude-error ratio against the ordinary method -- the acceptance metric for the estimator.

The acceptance criterion is *relative*: the network's attitude estimation error must be within 10 % of
the ordinary method's error.  "Ordinary method" is ambiguous, so this reports both honest baselines and
never mixes them:

    clip : dead-reckon the attitude from the *clipped* gyro            (the status quo on the vehicle)
    alg  : dead-reckon from the analytic lever-arm reconstruction      (physics front end only)

and the same for the network.  Errors are the INS tilt (angle between the estimated and true body z)
after a fixed horizon, evaluated on held-out **episodes** (never windows), restricted to the frames
where the window actually saturates -- the regime the estimator exists for.

Everything here is read-only with respect to the training code: it imports the dataset builder and the
model from `scripts/train_e2e.py` / `gpd_me/e2e.py` so the metric cannot drift from the one used
during training.

Run:
  PYTHONPATH=<repo>:<me> python scripts/eval_attitude.py --ckpt results/e2e_v6.pt --shards results/dr_1e7_v2
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

import torch  # noqa: E402

from gpd_me.e2e import E2ENet, GYRO_SLICE, WINDOW, att_errors_deg, ins_rollout_dt  # noqa: E402
import train_e2e as T  # noqa: E402  (the trainer's own dataset builder and window sampler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ME, "results", "e2e_v6.pt"))
    ap.add_argument("--shards", default=os.path.join(ME, "results", "dr_1e7_v2"))
    ap.add_argument("--limit", type=int, default=400, help="episodes to load (held-out split only)")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--horizon", type=int, default=12, help="frames (12 = 60 ms at 200 Hz)")
    ap.add_argument("--windows", type=int, default=4000, help="held-out windows to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = T.load_eps(a.shards, a.limit)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * a.val_frac))
    va_eps = [eps[i] for i in perm[:n_val]]          # same episode-level split rule as training
    D = T.build_dataset(va_eps, verbose=True)

    sd = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = E2ENet().to(dev)
    net.load_state_dict(sd["state"])
    net.eval()
    xf_m = torch.tensor(sd["xf_m"], dtype=torch.float32, device=dev)
    xf_s = torch.tensor(sd["xf_s"], dtype=torch.float32, device=dev)
    xp_m = torch.tensor(sd["xp_m"], dtype=torch.float32, device=dev)
    xp_s = torch.tensor(sd["xp_s"], dtype=torch.float32, device=dev)

    n_par = sum(p.numel() for p in net.parameters())
    print(f"checkpoint: {a.ckpt}")
    print(f"network size: {n_par:,} parameters  ({n_par*4/1024:.0f} KB fp32, {n_par/1024:.0f} KB int8)")

    idx = D["OKW"][:: max(1, len(D["OKW"]) // a.windows)][:a.windows]
    B = T.gather(D, idx, dev)
    with torch.no_grad():
        fine = ((B["fine"] - xf_m) / xf_s).clamp(-50.0, 50.0)
        w_hat, _ = net(fine, B["coarse"], (B["prior"] - xp_m) / xp_s, B["w_alg"], B["sat"])

        # inference latency (this is the deployability number: per-sample, single core, fp32)
        import time
        one = fine[:, :WINDOW]
        net(fine[:1], B["coarse"][:1], (B["prior"][:1] - xp_m) / xp_s, B["w_alg"][:1], B["sat"][:1])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        reps = 50
        for _ in range(reps):
            net(fine[:1], B["coarse"][:1], (B["prior"][:1] - xp_m) / xp_s, B["w_alg"][:1], B["sat"][:1])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms = 1000.0 * (time.time() - t0) / reps
        print(f"inference latency: {ms:.2f} ms per 48-frame window ({ms/48:.3f} ms per 5 ms control step)"
              f"  [{'cuda' if dev.type=='cuda' else 'cpu'}]")

        hs = a.horizon
        R_true = B["R_true"][:, :hs]
        tilt_clip = att_errors_deg(ins_rollout_dt(B["fine"][:, :hs, GYRO_SLICE], B["R0"], B["dt"]),
                                   R_true)[0]
        tilt_alg = att_errors_deg(ins_rollout_dt(B["w_alg"][:, :hs], B["R0"], B["dt"]), R_true)[0]
        tilt_net = att_errors_deg(ins_rollout_dt(w_hat[:, :hs], B["R0"], B["dt"]), R_true)[0]

        # only windows that actually saturate: the estimator's entire raison d'etre
        satw = B["sat"].any(-1).any(-1)
        if satw.any():
            tilt_clip, tilt_alg, tilt_net = (tilt_clip[satw], tilt_alg[satw], tilt_net[satw])
        # use the *final* tilt of the horizon (the integrated attitude error a controller would see)
        c, g, n = (float(tilt_clip[:, -1].mean()), float(tilt_alg[:, -1].mean()),
                   float(tilt_net[:, -1].mean()))
        # keep parallel flattened copies for the per-range breakdown
        satw_np = satw.cpu().numpy()
        tclip, talg, tnet = (tilt_clip.cpu().numpy(), tilt_alg.cpu().numpy(), tilt_net.cpu().numpy())
        dps_f = np.asarray(B["meta"][:, 1], float)[satw_np]

    print(f"\nheld-out windows with saturation: {int(satw.sum())} / {len(satw)}"
          f"   horizon {hs} frames ({hs*5} ms)")
    print(f"{'source':>22} {'tilt@horizon (deg)':>19} {'ratio vs clip':>14}")
    print(f"{'clipped gyro (ordinary)':>22} {c:19.2f} {1.00:14.2f}")
    print(f"{'analytic lever arm':>22} {g:19.2f} {g/c if c else float('nan'):14.2f}")
    print(f"{'network':>22} {n:19.2f} {n/c if c else float('nan'):14.2f}")
    print(f"\nACCEPTANCE: network/clip = {n/c if c else float('nan'):.3f}  "
          f"(target <= 0.10)   {'PASS' if c and n/c <= 0.10 else 'NOT YET'}")

    # per-range breakdown, because the low-range regime is where the direction is unobservable
    print(f"\nby gyro range (tilt@{hs*5}ms at the end of the horizon, deg):")
    print(f"{'dps':>6} {'n_win':>6} {'clip':>8} {'alg':>8} {'net':>8} {'net/clip':>9}")
    for d in sorted(set(dps_f.tolist())):
        m = dps_f == d
        if int(m.sum()) < 5:
            continue
        cc, gg, nn = (float(tclip[m][:, -1].mean()), float(talg[m][:, -1].mean()),
                      float(tnet[m][:, -1].mean()))
        print(f"{d:6.0f} {int(m.sum()):6d} {cc:8.2f} {gg:8.2f} {nn:8.2f} {nn/cc if cc else float('nan'):9.2f}")


if __name__ == "__main__":
    main()