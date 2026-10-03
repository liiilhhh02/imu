# Dialogue request — post-fix state of the out-of-range rate estimator (adversarial critique wanted)

**READ-ONLY: do not edit, create, or delete any file. Do not run training or collection — long jobs
are currently running on this machine and must not be disturbed.** Reading files and reasoning is all
I want. Answer in English.

## Where we are

You reviewed this codebase once before (the findings are merged in `review/CONSOLIDATED.md`). Every
BLOCKER/MAJOR you and two other reviewers raised has since been fixed, plus eight defects you did not
catch. The pipeline was re-collected (10^7 steps) and retrained. Three things now exist that I want
you to attack, because each is load-bearing for the paper this is turning into:

1. **A scale-degeneracy finding that contradicts the earlier review advice.** The review said "do not
   use `k = ‖r‖`; use the scalar the 1-D scan minimised". I implemented that, and it broke the
   estimator. The argument I ended up with (`docs/DESIGN.md` §2.1, code in `gpd_me/observer.py`
   `calibrate_scan` + `gpd_me/priors.py` `id_lever_arm`) is: with the quasi-steady model
   `s = (ww^T − |w|^2 I) r` and the saturated axes reconstructed from `|w|^2 = |s|/k`, the fitted `r`
   scales as `k` and the residual is *invariant* to `k` — the scan's cost is flat in the scale
   (measured `cost(2k)/cost(k) = 1.00–1.33` for the collapses vs 2.0–2.1 when the neglected tangential
   term pins it). Therefore `k` is only identifiable from the in-range LS vector, and where there is
   no in-range information (or the scan is flat) `k` is declared *unknown* and the estimator falls
   back to the plain clipped gyro.

2. **A negative closed-loop result** (`docs/STATUS.md` §5, `scripts/e4_closed_loop.py`). In the
   adjacent-pair dual-failure regime, the supervisor's policy with the **true** body rate holds in only
   17/20 independent fault draws; with the clipped gyro 1/20; with the retrained network ~0/20. A
   0.002 rad/s perturbation of the initial condition flips the outcome, and feeding the true attitude
   instead of the INS does not rescue it. My reading: this operating point sits on the stability
   boundary, so the closed loop cannot be used to demonstrate an estimator benefit, and the estimator's
   value must be shown open-loop (where the retrained network is 2.55x better than the clip and 2.19x
   better than the analytic front end on "per-flight lag", the norm of the per-flight time-mean error).

3. **The v6 numbers** (`docs/STATUS.md` §4, `README.md` §5, produced by `scripts/train_e2e.py`).
   Held-out (split by episode), saturated frames only:
   clipped 16.92 / algebraic 14.51 / network 6.63 rad/s per-flight lag; stable bias 9.87 / 3.21 / 1.33;
   jitter 28.32 / 27.65 / 13.64; per-range table shows the network's bias beating the clip in all 12
   gyro-range bands.

## What I want from you

Attack the load-bearing claims, in this order, and be concrete — cite the file/line or the number that
would have to change for your objection to be answered:

1. **Is the degeneracy argument actually right?** Try to falsify it: is there an identification route
   to `k` I am throwing away (tangential term with proper smoothing? variation of the spin-axis
   direction across the window? the *relative* shape of `|s|` over the window? multi-axis clipping
   ratios? the yaw-torque channel `T·wdot_z + w_z = G·Σ±u` as an extra constraint? the accelerometer
   along a second axis?). If some route works, the "unknown ⇒ clip" fallback is leaving information on
   the table and I want to know exactly which samples carry it.
2. **Is my reading of E4 defensible, or am I rationalising a failure?** If you think a closed-loop
   demonstration is still possible without retraining the supervisor's policy, say exactly what to
   change (initial condition family, fault timing, target trajectory, torque limits, which failure
   mask, what metric) and what would count as evidence. If you think it is not possible, say what the
   honest scope of the closed-loop claim is.
3. **Which v6 claim breaks first under review?** I am specifically worried about: (a) "per-flight" in
   the eval is actually per-*window* (grouped by the window's last frame's episode within a 48-frame
   window); (b) the algebraic baseline is now strong because I fixed its front end too, so claiming a
   2.19x win over it is a *harder* claim than it looks; (c) the losses `L_rate/L_bias/L_att` are
   supervised with simulator truth, so the open-loop numbers are an upper bound that a real system
   would not reach; (d) the domain randomisation draws the fault as an instantaneous mask change plus
   an injected spin, which may not cover realistic fault transients.
4. **One next step.** If you had this repo and one week, what single change would you make next, and
   what would you *cut* from the story? Prioritise things that increase the credibility of the central
   claim (priors identifiable from flight data; the accelerometer carries the out-of-range rate; the
   learned temporal part supplies what the algebra cannot) over things that increase the size of the
   numbers.

Pointers to read (do not bother with anything else unless your argument needs it):
`docs/DESIGN.md`, `docs/STATUS.md`, `gpd_me/observer.py`, `gpd_me/priors.py`, `gpd_me/e2e.py`,
`scripts/train_e2e.py`, `scripts/verify_priors.py`, `scripts/e4_closed_loop.py`,
`review/CONSOLIDATED.md`.