# Post-fix adversarial critique (omp `reviewer` subagent, 23 min, read-only)

Question set: [`prompt_grok_v2.md`](prompt_grok_v2.md) (the same three load-bearing claims, asked of
grok in parallel; the Cursor API was unreachable — see [`grok_dialogue_v2.md`](grok_dialogue_v2.md)).

## 1. Scale degeneracy — **holds** (with two caveats)

Independently reproduced the criterion on 71 windows of `results/dr_1e7_v2` (window = `t_fault+0.3 s`
.. +400 samples, exactly `id_lever_arm`'s):

* `cost(2k)/cost(k)` = **1.00–1.46** whenever the veto fires, **1.51–13.4** when it does not
  (a clean gap at the 1.5 threshold);
* when the veto passes, the scan's `k` is within **±20 %** of the true `|r_perp|`
  (ratios 0.93–1.03 on five windows); when it fires, the argmin is essentially a random draw
  spanning **0.003×–8.4×** the truth, often pinned to the search floor (one window: 0.10 mm vs 31.5 mm).

Five falsification attempts, **all negative** — i.e. `k` really is not recoverable from those windows:

1. adding the neglected tangential term to the scan's inner LS (smoothed derivative on the sample
   grid): no interior minimum at the true `k`, the argmin still ran to the grid floor in ~80 % of the
   vetoed windows;
2. the "relative shape of `|s|`" route: within a window the spin direction is fixed, so
   `|s| = k(A_i + z_i²)` is monotone in `k` — `k` and the direction are not separable;
3. the yaw channel: the identified ARX tracks the true `ω_z` only to **7.6–27 % RMS** (median 19 %)
   after the fault, and a joint `(k, z0)` fit lands within [0.67, 1.5]×`k_true` in only **2/14** windows;
4. multi-axis clipped ratios carry **zero** information (every pinned axis reads exactly ±lim);
5. there is no second accelerometer on the platform.

**Caveat 1 (wording)**: "the residual is invariant to `k`" is too strong — with one pinned axis an
off-axis term survives, which is why the ratio is 1.00–1.44 rather than exactly 1.00. Say
**"near-degenerate under a 2× scale change"**.

**Caveat 2 (actionable)**: the *fallback* is now the dominant `k` error. Measured on 150 shards with
the current `verify_priors.py`: `k` identified-only median **−16.7 %** (5–95 %: −78 %/+151 %),
−48 % at 100–300 dps, 100 % unknown above 1500 dps. On 14 clean windows the front-end L1 is
clip **14.45** / shipped-k **8.57** / **true-k 3.46** rad/s — an exact scale is worth another 2.5× —
**and on 3/14 windows the shipped-k front end is worse than the plain clip** (e.g. 19.2 vs 0.9 rad/s
with k̂ = 3.2 mm against k_true = 28.1 mm). ⇒ *"algebraic beats the clip in every band"* is a median
statement that hides those windows; a corroboration veto is needed on the in-range-LS `k` too.

## 2. E4 — **holds-with-caveat**

The boundary/chaos reading is defensible for the adjacent-pair regime, but the `net` row is not the
trained configuration and the perturbation evidence the author quotes is not in the repo. Keep E4
strictly as the boundary stress test; do not use its `net` row as a result.

## 3. Which open-loop claim breaks first — **(a) the metric name**, plus a worse defect

* **"per-flight" is per-*window*** (`train_e2e.py:362` samples window ends; `per_flight_bias`
  groups by the window's last frame's episode and averages the *48-frame* window mean). The headline
  6.63/16.92/14.51 is the 240 ms window-mean — exactly the training objective — not the flight-level
  quantity that integrates into attitude drift. The same comparison reads 1.17× here and 2.2× under a
  different reduction ⇒ **the reduction must be stated with the number**.
* Off-by-one in the window/episode safety test (`train_e2e.py:168` uses `ep[i-W+1]==ep[i]`, which
  should be `ep[i-W]==ep[i]`): one window per boundary still mixes the previous episode into `R0`.
* **Worse than (b)–(d): the domain randomisation's nominal phase is contaminated.** `collect_dr.py:105`
  calls `env.shut_down_rotors(flag)` — which for flag 3 *immediately injects* `[U(−3,3), U(−3,3),
  −U(24,26)]` rad/s — before the "nominal" phase, and line 106 restores only the mask. Measured: the
  fraction of nominal-phase frames with ≥1 saturated axis for flag 3 is **0.98 / 0.72 / 0.38 / 0.13**
  at 100/300/1000/1500 dps; `k` is unknown in 15/17 flag-3 shards at 100 dps. ⇒ the story
  "priors from a healthy nominal flight" is **not what the data contain** at low/mid range, and the
  low-range `k` error is partly a data-generation artifact, **not** a physical limit.
* `collect_dr.py`'s docstring advertises "abrupt vs gradual" fault transients; no gradual path exists.

## 4. Single next step (one week)

Re-collect the shards with a **clean nominal phase** (zero the injected body rate before the nominal
segment, inject the spin only at `t_fault`; keep the identification manoeuvre inside the range at
100–300 dps), then re-run `verify_priors` and one retrain, reporting `k` **per route** (scan /
in-range-LS / unknown) with a paired, correctly-named statistic. Either the low/mid-range `k` error
drops materially (the central claim gets stronger with the same code) or it stays and the paper can
honestly say the limit is physical and the clip fallback is the right deployment choice.

**Cut from the story**: the name "per-flight lag" and the 2.55×/2.19× headline as the lead number;
the sentence "algebraic beats the clip in all three bands"; and any use of E4's `net` row as a result.

## Actions taken in response (same day)

* `scripts/collect_dr.py`: records the env's own fault spin, **clears it for the nominal phase**, and
  re-applies it at `t_fault` — re-collection into `results/dr_1e7_v3` is running.
* the metric's name and scope are being corrected in the docs (`per-window lag`, with the reduction
  stated next to every number).
* planned: the `k` corroboration veto (keep the clip when the identified scale implies an unphysical
  spin magnitude) and the `train_e2e.py:168` off-by-one fix.