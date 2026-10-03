# Consolidated code review — end-to-end IMU out-of-range estimator

Three independent reviews of `/home/liiil/Downloads/me` (ours; the vendored `gym_pybullet_drones` is the
supervisor's and was out of scope):

| reviewer | how it was run |
|---|---|
| **grok-4.7-high** | Cursor Agent CLI (`~/.local/bin/agent --print --trust --mode ask --model grok-4.7-high`), prompt in `prompt_grok.md` |
| **PhysMath** | omp `reviewer` subagent, physics/maths focus |
| **PipelineValidity** | omp `reviewer` subagent, validity/robustness focus |

All three were read-only.  Where two or three reviewers independently found the same defect it is
marked **consensus**.

---

## BLOCKER 1 — train-time physics hardcoded to 200 Hz on multi-rate data (consensus: all 3)

`scripts/train_e2e.py:211,214,188-189,284-286` integrate the attitude INS, the finite-difference
`wdot`, the spectral frequency axis and the evaluation rollouts with a hardcoded `dt=0.005`
(`*200.0`), but `scripts/collect_dr.py:38` collects at 100/200/250/400 Hz and stores `dt`.
`PipelineValidity` verified by reading the shards: dt ∈ {0.01, 0.005, 0.004, 0.0025} with only ~21 %
of episodes at 200 Hz.  Consequently the attitude loss (weight 0.5) is minimised by
`w_hat ≈ 0.5*w_true` on 400 Hz episodes and `2*w_true` on 100 Hz episodes — a frequency-dependent bias
that fights the rate loss — and every per-dps table averages over confounded sampling rates.

**Verified by me** (my v2 rewrite dropped `DT` from the dataset dict; the constant is in the code).
**FIXED**: `DT`/`EP` are carried per frame, `rodrigues_dt`/`ins_rollout_dt` added, `wdot` divides by
the per-window `dt`, the spectral ">10 Hz" mask is per-window.

## BLOCKER 2 — the reported "stable bias" is one *global* mean, so opposite per-flight lags cancel (grok)

`train_e2e.py decompose()` computes `||mean_t(e_t)||` over *all* saturated frames of the whole val
set, while the loss penalises each **window's** own mean (`errv.mean(dim=1)`).  Per-flight lags of
opposite sign therefore cancel in the reported number and reappear as "jitter", so a 7–8x drop in it
does not mean the error that *integrates inside one flight* fell 7–8x.  The log line "train bias
7.65 → 0.58" is a third statistic again (a Huber of the minibatch window mean).

**Verified by me** (reading `decompose` vs `forward_all`).
**FIXED**: added a `per-flight lag ||mean_t err||` metric (mean over episodes of the episode's mean
error norm) — the quantity that actually tracks attitude drift.

## BLOCKER 3 — the lever-arm residual is handed the *true* specific thrust (grok)

`collect_dr.py:167,197` log `tom = sum(thrust)/M_true`, the accelerometer was generated from that same
quantity, and the estimator builds `s = a_m − tom·e_z` and inverts it.  A 20 % mass error is ~2 m/s² —
the same order as the lever-arm signal — so the published rate numbers belong to an estimator whose
nuisance term was already removed.  `g_T` is only a feature/soft target, never used to form `s`.
Note: in the *first* 10^7 set (`dr_norand`) the airframe randomisation was inert, so the effective
mass was 1.0 while `tom` divided by the requested random mass — there the residual was not just
optimistic but **wrong**, which is why the algebraic observer scored *worse* than the clipped gyro.
**VERIFIED by me** (consistent with the observed anomaly).  **PENDING (P1)**: form `tom` from the
command through the identified `g_T` and the actuator lag; re-train and re-score.

## BLOCKER 4 — `k` is `||r||`, not `|r_perp|`, and the scan's scale is discarded (grok)

`train_e2e.py:63` sets `k = ||prior[:3]||` while the invariant is `|s| = |w|²·|r_perp|` with
`r_perp = r − (r·ŵ)ŵ`.  The nominal excitation is pure yaw, so `r_z` sits in the nullspace and the
ridge shrinks it; after the fault the spin is ~25° off body z so the *relevant* `|r_perp|` differs.
When `||r||` is too large, `n2 − known < 0` and `algebraic_estimate` **floors the saturated axis at 0,
which is worse than the clip** — consistent with the very poor 100 dps row.
**PENDING (P1)**: solve with the vector `r`, or evaluate `|r_perp|` against the current `ω̂`.

## MAJOR — the INS loss was compared one control period early (PhysMath)

`R_true = D["R"][i−W:i]` against `R0 = D["R"][idx−W]`, so rollout frame 0 is one step ahead of the
truth's frame 0 (an 11.5° offset at 40 rad/s).  Minimising the window-mean tilt then drives
`w_hat ≈ 0.969·w_true` — a systematic underestimate the bias loss cannot remove — and inflates every
reported tilt.  **Verified by me.**  **FIXED** (`R_true = D["R"][i−W+1:i+1]`).

## MAJOR — windows sampled across episode boundaries (consensus: grok-adjacent, PhysMath, PipelineValidity)

`idx` was drawn uniformly over the concatenated frame array, so a 48-frame window (and its attitude
chain) could straddle two episodes with different lever arm, dps, M and sample rate — corrupting the
INS targets in training and inflating the tilt for **all three** baselines in the held-out table.
**FIXED**: an episode id and a boundary-safe window list are computed in `build_dataset`; all sampling
(training + eval) now uses it.

## MAJOR — the lever-arm prior is lost on most 100–300 dps episodes (PhysMath)

`priors.py:107` requires in-range samples above a 2 rad/s floor, but the identification manoeuvre is
tuned to 0.3·range = 0.5–1.6 rad/s at those ranges, and for 1/2-rotor faults the spin saturates almost
immediately; the scan fallback only inspects the first 400 samples while the fault onset is uniform in
[0.12, 0.6]·steps, so it sees the fault only ~17 % of the time → `r = 0`, `k = NaN → 0`.
**PENDING (P1)**: lower/remove the floor, tune the manoeuvre to ≥2 rad/s, cover the fault onset with
the scan window, and treat `k = 0` as "unknown" rather than "zero".

## MAJOR — the tangential term is disabled in the preferred identification path (PhysMath)

`priors.py:110` calls `identify_lever_arm(..., use_tangential=False)`, yet the manoeuvre makes
`|wdot × r| / (|w|²|r_perp|) ≈ 2πf/(0.3·range)` reach ~2.4 — i.e. the neglected term **dominates** in
exactly the samples used to fit the centripetal-only model, biasing `r` (and `k`) by up to 2–3x.
**PENDING (P1)**: use the tangential LS (it is linear in `r` and needs no ground truth), or prefer the
scan as the observer's own docstring advises.

## MINOR — the `l_anchor` loss is structurally zero in the default mode (consensus: PhysMath, PipelineValidity)

`w_hat = w_alg + sat·δ` and `w_alg` equals the gyro on in-range frames, so `l_anchor ≡ 0` there: the
`--ablate anchor` row is flat by construction and the reported "in-range anchor = 0.000" is
**tautological, not evidence**.  **PENDING (P2)**: delete it in `e2e` mode or redefine it as a
consistency loss on the unsaturated axes of *partially* saturated frames.

## MINOR — a failed yaw identification silently zeroes `G` and `T` (PhysMath)

`id_yaw_channel` returns NaN below 50 in-range pairs and `np.nan_to_num` turns that into 0, making
`l_torque` become `huber(w_z/10)` for that episode — actively wrong supervision pulling `w_hat_z → 0`
— and a noisy ARX can return a negative `T` (inverted derivative sign).  **PENDING (P2)**: skip the
torque term when `G`/`T` are not identified; clamp `T > 0`.

## MINOR — the `"indi"` excitation mode is a silent no-op (PipelineValidity)

In `collect_dr.py` the `mode == "indi"` branch falls through to `_excite_random`; the INDI controller
is imported but never driven.  **PENDING (P2)**: implement it or remove it from the mode list.

## MINOR — IMU bias/noise draws are seed-degenerate across episodes (PipelineValidity)

`IMUConfig.seed` is never set per episode, so every episode replays the same normal-draw stream
(only the std changes).  **PENDING (P2)**: pass the episode seed.

## MINOR — the v1 estimator's "cross-episode" split claim is false (PipelineValidity)

`scripts/train_estimator.py` splits by *window*, not by episode, so the README's "cross-episode"
attribution for its 1.07 rad/s number is wrong.  **README must be corrected.**

---

## What this does to the headline numbers

The numbers quoted before this review (bias 0.81 rad/s, 7–8x better than the clipped gyro, tilt
11.5° vs 12.6°) were produced with BLOCKER 1, MAJOR-shift and the boundary bug **in the loop**, and
were scored with BLOCKER 2's global-mean metric.  A 400-iteration smoke run with the fixes in place
gives, on the saturated frames of a 150-episode subset:

| estimator | per-flight lag | global bias | jitter | tilt@60 ms |
|---|---|---|---|---|
| clipped gyro | 8.79 | 8.47 | 16.61 | 10.06 |
| algebraic | 7.74 | 4.85 | 15.48 | 8.81 |
| **network** | **4.76** | **2.13** | 13.12 | 8.17 |

So the ordering survives, the *magnitude* shrinks a lot, and the tilt advantage becomes small.  A
proper retrain (10^7 steps, all fixes) is required before any number is quoted again.

## Fix status

FIXED: BLOCKER 1 (dt), MAJOR shift, MAJOR boundaries, BLOCKER 2 (metric added; both metrics now
printed).
PENDING and ranked: BLOCKER 3 (`tom` realism), BLOCKER 4 (`k` vs `|r_perp|`), the prior floor /
manoeuvre / scan window, the tangential LS, the dead anchor term, the `G/T` guard, the `indi` no-op,
the IMU seed, and the README corrections.