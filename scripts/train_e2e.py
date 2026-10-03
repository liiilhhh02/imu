"""Train the end-to-end estimator v2 on the domain-randomised shards (GPU).

v2 changes versus the first attempt (see the design review in `gpd_me/e2e.py`):
  * output is physics-parameterised (`w_hat = w_alg + sat * delta`, starting exactly at the algebra)
  * the feature set uses the accelerometer *residual* (DC removed), its norm, the algebraic estimate
    and the norms of the gyro
  * a 48-frame (240 ms) horizon so the INS loss sees the attitude drift that actually matters
  * Huber losses (no ill-posed heteroscedastic NLL), grad clip 20, LR warmup
  * the slow parameter head is **detached in stage 1** and supervised in stage 2

Run:  PYTHONPATH=<repo>:<me> python scripts/train_e2e.py --shards results/dr_val --iters 4000
"""
import argparse
import glob
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
from scipy.spatial.transform import Rotation  # noqa: E402

from gpd_me.e2e import (COARSE, FINE_FEATURES, GYRO_SLICE, WINDOW, E2ENet,  # noqa: E402
                        algebraic_estimate, att_errors_deg, coarse_summary, fine_features,
                        huber, ins_rollout, ins_rollout_dt, make_dft)

YAW_SIGN = np.array([+1.0, -1.0, +1.0, -1.0])
KEYS = ("gyro", "accel", "u", "tom", "omega", "quat", "sat", "mask", "dps", "dt", "prior", "true")


def load_eps(pattern, limit=None):
    files = sorted(glob.glob(os.path.join(pattern, "*.npz")))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        out.append({k: z[k] for k in KEYS})
    return out


def build_dataset(eps, verbose=True):
    n_bad = 0
    acc = {k: [] for k in ("Xf", "Xc", "Xp", "Walg", "W", "R", "A", "U", "TOM", "SAT", "INR",
                           "CLIP", "META", "PRIOR_T", "DT", "EP")}
    ep_id = 0
    for e in eps:
        n = len(e["gyro"]); dt = float(e["dt"]); lim = np.deg2rad(float(e["dps"]))
        prior = np.asarray(e["prior"], float).ravel()
        # drop episodes that produced non-finite values (diverged integrations)
        chk = [np.asarray(e[k], float) for k in ("gyro", "accel", "u", "tom", "omega", "quat", "prior")]
        if not all(np.isfinite(c).all() for c in chk):
            n_bad += 1
            continue
        sat = np.asarray(e["sat"], bool)
        gyro = np.asarray(e["gyro"], float)
        k = float(np.linalg.norm(prior[:3]))
        s = np.asarray(e["accel"], float) - np.array([0.0, 0.0, 1.0])[None, :] * \
            np.asarray(e["tom"], float)[:, None]
        w_alg = algebraic_estimate(gyro, s, k, sat, lim)
        # causal 2 s coarse summary
        cs = np.zeros((n, COARSE), np.float32)
        runt = np.cumsum(np.asarray(e["tom"], float)); runtu = np.cumsum(np.asarray(e["tom"], float) ** 2)
        rung = np.cumsum(np.abs(gyro), 0); rungu = np.cumsum(gyro ** 2, 0)
        runu = np.cumsum(np.abs(np.asarray(e["u"], float)), 0)
        runs = np.cumsum(sat.astype(np.float64), 0)
        idx_prev = np.arange(n) - int(2.0 / dt)
        for i in range(n):
            a = idx_prev[i]
            sc = 1.0 / (i - a if a >= 0 else i + 1)
            sub = lambda run: (run[i] - (run[a] if a >= 0 else 0.0)) * sc
            mt, mt2 = float(np.atleast_1d(sub(runt))), float(np.atleast_1d(sub(runtu)))
            mg, mg2 = sub(rung), sub(rungu)
            cs[i] = np.concatenate([
                [mt, np.sqrt(max(mt2 - mt * mt, 0.0)), float(np.atleast_1d(mg).mean()),
                 float(np.sqrt(np.maximum(np.atleast_1d(mg2).mean()
                                          - np.atleast_1d(mg).mean() ** 2, 0.0))),
                 float(np.atleast_1d(sub(runu)).mean())],
                sub(runs), prior])
        acc["Xf"].append(fine_features(e["accel"], gyro, sat, e["u"], e["tom"], prior, w_alg))
        acc["Xc"].append(cs)
        acc["Xp"].append(np.tile(prior[None, :], (n, 1)))
        acc["Walg"].append(w_alg); acc["A"].append(e["accel"])
        acc["W"].append(e["omega"]); acc["SAT"].append(sat); acc["INR"].append(~sat.any(axis=1))
        acc["CLIP"].append(gyro); acc["U"].append(e["u"]); acc["TOM"].append(e["tom"])
        acc["R"].append(Rotation.from_quat(e["quat"]).as_matrix())
        acc["META"].append(np.tile(np.array([float(e["mask"][1]), float(e["dps"])]), (n, 1)))
        # per-frame control timestep and episode id: the shards are multi-rate (100-400 Hz) and
        # windows must not straddle an episode boundary
        acc["DT"].append(np.full(n, dt, np.float32))
        acc["EP"].append(np.full(n, ep_id, np.float32))
        ep_id += 1
        tr = e["true"][0]
        pt = np.array([*np.asarray(tr[0], float), 1.0 / float(tr[1]), 0.0, 0.0, lim])
        acc["PRIOR_T"].append(np.tile(pt[None, :], (n, 1)))
    D = {k: np.concatenate(v) for k, v in acc.items()}
    # drop frames that are non-finite anywhere critical (k=0 identification -> |s|/k explodes etc.)
    crit = ["Xf", "Xc", "Xp", "Walg", "W", "R", "A", "U", "TOM", "CLIP", "PRIOR_T"]
    ok = np.ones(len(D["Xf"]), bool)
    detail = {}
    for k in crit:
        f = np.isfinite(np.asarray(D[k], float)).reshape(len(ok), -1).all(axis=1)
        detail[k] = int((~f).sum())
        ok &= f
    if verbose and (~ok).any():
        print(f"    dropped {int((~ok).sum())} non-finite frames {detail}")
    for k in list(D.keys()):
        D[k] = D[k][ok]
    D["SAT"] = D["SAT"] > 0.5
    D["INR"] = D["INR"] > 0.5
    D = {k: (np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)
             if v.dtype.kind == "f" else v) for k, v in D.items()}
    # windows whose whole span lies inside one episode (otherwise the INS target mixes airframes,
    # ranges and sampling rates inside a single GRU window)
    ep = np.rint(D["EP"]).astype(np.int64)
    ok = np.zeros(len(ep), bool)
    ok[WINDOW:] = ep[1:len(ep) - WINDOW + 1] == ep[WINDOW:]   # window end i: ep[i-W+1] == ep[i]
    D["OKW"] = np.where(ok)[0]
    D["SAT"] = D["SAT"] > 0.5
    D["INR"] = D["INR"] > 0.5
    if verbose:
        if n_bad:
            print(f"    dropped {n_bad} non-finite episodes")
        print(f"    {len(D['Xf'])} frames, sat {100*D['SAT'].any(1).mean():.0f}%, "
              f"masks {np.unique(D['META'][:, 0])}, ranges {np.unique(D['META'][:, 1])}")
    return D


def gather(D, idx, dev):
    W = WINDOW
    t = lambda x, dt_=torch.float32: torch.tensor(x, dtype=dt_, device=dev)
    return dict(
        fine=t(np.stack([D["Xf"][i - W:i] for i in idx])),
        coarse=t(D["Xc"][idx - 1]), prior=t(D["Xp"][idx - 1]),
        w_alg=t(np.stack([D["Walg"][i - W:i] for i in idx])),
        sat=t(np.stack([D["SAT"][i - W:i] for i in idx]), torch.bool),
        w_true=t(np.stack([D["W"][i - W:i] for i in idx])),
        # the W rates at frames i-W..i-1 produce the attitudes at i-W+1..i, so the rollout must be
        # compared against that shifted window (comparing against i-W..i-1 was one period early and
        # biased w_hat toward ~0.97*w_true on a constant-rate window)
        R_true=t(np.stack([D["R"][i - W + 1:i + 1] for i in idx])), R0=t(D["R"][idx - W]),
        inr=t(np.stack([D["INR"][i - W:i] for i in idx]), torch.bool),
        a_m=t(np.stack([D["A"][i - W:i] for i in idx])),
        u=t(np.stack([D["U"][i - W:i] for i in idx])),
        tom=t(np.stack([D["TOM"][i - W:i] for i in idx])),
        prior_t=t(D["PRIOR_T"][idx - 1]),
        dt=t(D["DT"][idx - 1]), ep=t(D["EP"][idx - 1]),
        meta=D["META"][idx - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join(ME, "results", "dr_val"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--stage2", type=int, default=1500, help="iterations at which the slow head is enabled")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--tilt_frames", type=int, default=12,
                    help="INS evaluation horizon in frames (12 = 60 ms); the 240 ms rollout is "
                         "dominated by the spin magnitude above ~1000 dps and is not a usable proxy")
    # ---- step 3: ablation switches ----
    ap.add_argument("--ablate", default="", help="comma list of loss terms to drop: "
                                                  "bias,att,phys,torque,spec,anchor,prior")
    ap.add_argument("--no_slow", type=int, default=0, help="disable the slow parameter head")
    ap.add_argument("--mode", default="e2e", choices=["e2e", "residual", "noalg"],
                    help="e2e = physics-parameterised output + w_alg feature; "
                         "residual = w_alg baseline but unmasked delta; noalg = no w_alg at all")
    a = ap.parse_args()
    if a.out is None:
        tag = a.mode + ("" if not a.ablate else "_no" + a.ablate.replace(",", ""))
        tag += "_noslow" if a.no_slow else ""
        a.out = os.path.join(ME, "results", f"e2e_{tag}.pt")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}" + (f"  {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""))

    eps = load_eps(a.shards, a.limit)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(eps)); n_val = max(1, int(len(eps) * a.val_frac))
    tr_eps = [eps[i] for i in perm[n_val:]]; va_eps = [eps[i] for i in perm[:n_val]]
    print(f"episodes {len(eps)} -> train {len(tr_eps)} / val {len(va_eps)} (split by episode)")
    print("  train:"); Dtr = build_dataset(tr_eps)
    print("  val:"); Dva = build_dataset(va_eps)

    xf_m, xf_s = Dtr["Xf"].mean(0), Dtr["Xf"].std(0) + 1e-6
    xp_m, xp_s = Dtr["Xp"].mean(0), Dtr["Xp"].std(0) + 1e-6
    net = E2ENet().to(dev)
    if a.load:
        net.load_state_dict(torch.load(a.load, map_location=dev, weights_only=False)["state"]); print(f"loaded {a.load}")
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / 200.0))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(a.amp) and dev.type == "cuda")
    dc_, ds_ = make_dft(WINDOW, 0.005); dc_, ds_ = dc_.to(dev), ds_.to(dev)
    hi = (torch.arange(WINDOW // 2 + 1, device=dev) / (WINDOW * 0.005)) > 10.0
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    xf_m_t, xf_s_t, xp_m_t, xp_s_t = T(xf_m), T(xf_s), T(xp_m), T(xp_s)
    ys_t = torch.tensor(YAW_SIGN, device=dev)
    ab = {t.strip() for t in a.ablate.split(",") if t.strip()}
    def w(term, val):
        return 0.0 * val if term in ab else val

    def forward_all(B, stage1):
        fine = ((B["fine"] - xf_m_t) / xf_s_t).clamp(-50.0, 50.0)
        if a.mode == "noalg":                     # truly end-to-end: no physics baseline anywhere
            fine = fine.clone(); fine[..., 23:26] = 0.0
        base = torch.zeros_like(B["w_alg"]) if a.mode == "noalg" else B["w_alg"]
        satm = torch.ones_like(B["sat"]) if a.mode == "residual" else B["sat"]
        w_hat, corr = net(fine, B["coarse"], (B["prior"] - xp_m_t) / xp_s_t, base, satm)
        corr_eff = torch.zeros_like(corr) if (stage1 or a.no_slow) else corr
        r_id = B["prior"][:, :3] + corr_eff[:, :3] * 0.02
        g_T = B["prior"][:, 3] + corr_eff[:, 3] * 0.5
        G = B["prior"][:, 4] + corr_eff[:, 4] * 5.0
        Tt = B["prior"][:, 5] + corr_eff[:, 5] * 1.0
        satf = B["sat"].any(-1)
        l_rate = huber((w_hat - B["w_true"]).abs().sum(-1))[satf].mean() if satf.any() else w_hat.sum() * 0
        dtv = B["dt"]                                     # per-window control timestep (B,)
        R_hat = ins_rollout_dt(w_hat, B["R0"], dtv)
        tilt, full = att_errors_deg(R_hat, B["R_true"])
        l_att = (huber(tilt / 10.0).mean() + 0.2 * huber(full / 10.0).mean())
        wdot = (w_hat[:, 1:] - w_hat[:, :-1]) / dtv[:, None, None].clamp_min(1e-6)
        wm = w_hat[:, :-1]; rr = r_id[:, None, :].expand_as(wm)
        cent = torch.cross(wm, torch.cross(wm, rr, dim=-1), dim=-1)
        tan = torch.cross(wdot, rr, dim=-1)
        zz = torch.zeros_like(B["tom"][:, :-1])
        resid = B["a_m"][:, :-1] - torch.stack([zz, zz, B["tom"][:, :-1]], -1) - cent - tan
        l_phys = huber(resid.abs().sum(-1) / 10.0).mean()
        yaw_cmd = (B["u"][:, :-1] * ys_t[None, None, :]).sum(-1)
        l_torque = huber((Tt[:, None] * wdot[..., 2] + wm[..., 2] - G[:, None] * yaw_cmd) / 10.0).mean()
        errv = w_hat - B["w_true"]
        l_bias = huber(errv.mean(dim=1).abs().sum(-1)).mean()          # window-mean error = stable lag
        l_anchor = huber((w_hat - B["fine"][..., GYRO_SLICE]).abs().sum(-1))[B["inr"]].mean() \
            if B["inr"].any() else w_hat.sum() * 0
        z = w_hat[..., 2]; zc = z[:, None, :]
        p = (zc * dc_[None]).sum(-1) ** 2 + (zc * ds_[None]).sum(-1) ** 2
        l_spec = p[:, hi].sum(-1).mean() / p[:, ~hi].sum(-1).mean().clamp_min(1e-6)
        l_prior = torch.zeros((), device=dev)
        if not stage1:
            l_prior = (huber(r_id - B["prior_t"][:, :3]).mean() / 0.02
                       + huber(g_T - B["prior_t"][:, 3]).mean())
        total = (l_rate + w("att", 0.5 * l_att) + w("phys", 0.3 * l_phys)
                 + w("torque", 0.1 * l_torque) + w("anchor", 0.3 * l_anchor)
                 + w("spec", 0.05 * l_spec) + w("prior", 2.0 * l_prior) + w("bias", 1.0 * l_bias))
        return total, dict(rate=l_rate, att=l_att, phys=l_phys, torque=l_torque,
                           anchor=l_anchor, spec=l_spec, prior=l_prior, bias=l_bias), w_hat, R_hat

    print("\n=== training (stage 1: physics head only; stage 2: + parameter head) ===")
    t0 = time.time()
    for it in range(a.iters):
        stage1 = it < a.stage2
        idx = rng.choice(Dtr["OKW"], size=a.batch)
        B = gather(Dtr, idx, dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=bool(a.amp) and dev.type == "cuda"):
            loss, parts, _, _ = forward_all(B, stage1)
        if not torch.isfinite(loss):
            if not globals().get("_warned", False):
                globals()["_warned"] = True
                print(f"  [warn] non-finite loss at it={it}; skipping update. "
                      f"|fine|={float(B['fine'].abs().max()):.3g} "
                      f"|walg|={float(B['w_alg'].abs().max()):.3g} "
                      f"|wtrue|={float(B['w_true'].abs().max()):.3g}")
            opt.zero_grad(set_to_none=True); continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 20.0)
        scaler.step(opt); scaler.update(); sched.step()
        if it % 200 == 0 or it == a.iters - 1:
            with torch.no_grad():
                iv = rng.choice(Dva["OKW"], size=min(a.batch, 256))
                lv, pv, _, _ = forward_all(gather(Dva, iv, dev), False)
            print(f"  it{it:5d} {'S1' if stage1 else 'S2'} L={float(loss):8.3f} "
                  f"| rate {float(parts['rate']):7.3f} att {float(parts['att']):6.3f} "
                  f"phys {float(parts['phys']):6.3f} anch {float(parts['anchor']):6.3f} "
                  f"spec {float(parts['spec']):5.3f} bias {float(parts['bias']):6.3f} prior {float(parts['prior']):6.3f} "
                  f"| val L={float(lv):8.3f} rate {float(pv['rate']):7.3f} att {float(pv['att']):6.3f} "
                  f"({time.time()-t0:4.0f}s)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(state=net.state_dict(), xf_m=xf_m, xf_s=xf_s, xp_m=xp_m, xp_s=xp_s), a.out)
    print(f"saved {a.out}")

    with torch.no_grad():
        iv = Dva["OKW"][::max(1, len(Dva["OKW"]) // 3000)]
        B = gather(Dva, iv, dev)
        _, _, w_hat, _ = forward_all(B, False)
        satf = B["sat"].any(-1)
        e_net = (w_hat - B["w_true"]).abs().sum(-1)[satf].mean()
        e_alg = (B["w_alg"] - B["w_true"]).abs().sum(-1)[satf].mean()
        e_clip = (B["fine"][..., GYRO_SLICE] - B["w_true"]).abs().sum(-1)[satf].mean()
        hs = a.tilt_frames
        t_net = att_errors_deg(ins_rollout_dt(w_hat[:, :hs], B["R0"], B["dt"]),
                               B["R_true"][:, :hs])[0].mean()
        t_alg = att_errors_deg(ins_rollout_dt(B["w_alg"][:, :hs], B["R0"], B["dt"]),
                               B["R_true"][:, :hs])[0].mean()
        t_cl = att_errors_deg(ins_rollout_dt(B["fine"][:, :hs, GYRO_SLICE], B["R0"], B["dt"]),
                              B["R_true"][:, :hs])[0].mean()
        anch = (w_hat - B["fine"][..., GYRO_SLICE]).abs().sum(-1)[B["inr"]].mean() if B["inr"].any() else torch.tensor(float("nan"))
    def decompose(pred):
        e = (pred - B["w_true"])[satf]
        return float(e.mean(0).norm()), float((e - e.mean(0)).norm(dim=-1).mean())
    b_net, j_net = decompose(w_hat); b_cl, j_cl = decompose(B["fine"][..., GYRO_SLICE])
    b_al, j_al = decompose(B["w_alg"])
    def per_flight_bias(pred):
        """Mean over episodes of ||mean_{t in flight} error||: a global mean lets opposite per-flight
        lags cancel, which is exactly what the loss does NOT do (it penalises each window's mean)."""
        e = (pred - B["w_true"]).cpu().numpy(); sm = B["sat"].any(-1).cpu().numpy()
        epv = np.rint(B["ep"].cpu().numpy()).astype(np.int64)
        out = []
        for eid in np.unique(epv):
            j = np.where(epv == eid)[0]; m = sm[j]
            if m.any():
                out.append(float(np.linalg.norm(e[j][m].mean(0))))
        return float(np.mean(out)) if out else float("nan")
    print("\n=== held-out (episode split), saturated frames ===")
    print(f"  per-flight lag ||mean_t err|| : clipped {per_flight_bias(B['fine'][..., GYRO_SLICE]):6.2f} "
          f"| algebraic {per_flight_bias(B['w_alg']):6.2f} | network {per_flight_bias(w_hat):6.2f} rad/s")
    print(f"  stable bias |mean_t err| : clipped {b_cl:6.2f} | algebraic {b_al:6.2f} | network {b_net:6.2f} rad/s")
    print(f"  jitter      (std)        : clipped {j_cl:6.2f} | algebraic {j_al:6.2f} | network {j_net:6.2f} rad/s")
    print(f"  rate err : clipped {float(e_clip):7.3f} | algebraic {float(e_alg):7.3f} | "
          f"network {float(e_net):7.3f} rad/s")
    print(f"  INS tilt @{a.tilt_frames*5} ms : clipped {float(t_cl):7.2f} | algebraic {float(t_alg):7.2f} | "
          f"network {float(t_net):7.2f} deg")
    print(f"  in-range anchor |net - gyro| = {float(anch):.3f} rad/s")

    # ---- stratified by gyro range.  Two different reductions are needed and must not be mixed:
    #      * bias / jitter  -> frame level, over the saturated frames of the full window
    #      * tilt           -> window level, because the short-horizon rollout only has `hs` frames
    #        (mixing a (B,hs) tensor with a (B,WINDOW) mask was the bug that killed the run)
    try:
        dpsf = np.repeat(np.asarray(B["meta"][:, 1], float), WINDOW)
        satf_np = B["sat"].any(-1).reshape(-1).cpu().numpy()
        e_n = (w_hat - B["w_true"]).reshape(-1, 3).cpu().numpy()
        e_a = (B["w_alg"] - B["w_true"]).reshape(-1, 3).cpu().numpy()
        e_c = (B["fine"][..., GYRO_SLICE] - B["w_true"]).reshape(-1, 3).cpu().numpy()
        hs = a.tilt_frames
        dpsw = np.asarray(B["meta"][:, 1], float)
        anyw = B["sat"].any(-1).any(-1).cpu().numpy()
        t_n = att_errors_deg(ins_rollout_dt(w_hat[:, :hs], B["R0"], B["dt"]),
                             B["R_true"][:, :hs])[0][:, -1].cpu().numpy()
        t_a = att_errors_deg(ins_rollout_dt(B["w_alg"][:, :hs], B["R0"], B["dt"]),
                             B["R_true"][:, :hs])[0][:, -1].cpu().numpy()
        t_c = att_errors_deg(ins_rollout_dt(B["fine"][:, :hs, GYRO_SLICE], B["R0"], B["dt"]),
                             B["R_true"][:, :hs])[0][:, -1].cpu().numpy()
        print(f"\n  by gyro range: frame-level bias/jitter (saturated frames) and"
              f" window-level tilt@{hs*5}ms")
        print(f"    {'dps':>6} {'n_frame':>8} | {'clipped':>18} | {'algebraic':>18} | {'network':>18}")
        for dps in sorted(set(dpsf.tolist())):
            m = satf_np & (dpsf == dps)
            if int(m.sum()) < 5:
                continue
            def bjt(e, t, mw):
                ee = e[m]
                return (f"{np.linalg.norm(ee.mean(0)):5.1f}/"
                        f"{np.linalg.norm(ee - ee.mean(0), axis=1).mean():4.1f}/"
                        f"{t[mw].mean():4.1f}")
            mw = anyw & (dpsw == dps)
            print(f"    {dps:>6.0f} {int(m.sum()):>8} | {bjt(e_c, t_c, mw):>18} | "
                  f"{bjt(e_a, t_a, mw):>18} | {bjt(e_n, t_n, mw):>18}")
    except Exception as _e:
        print(f"  [warn] stratified evaluation failed: {type(_e).__name__}: {_e}")


if __name__ == "__main__":
    main()