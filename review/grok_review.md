The headline table is not a measurement of per-flight lag, and several quantities the network is scored on are oracle signals. Details below, in the order they break the claim.

## BLOCKER

**1. Reported "stable bias" is the norm of one global mean, so opposite per-flight lags cancel.**
`scripts/train_e2e.py:289-291` and `:325-327`. `decompose` does `||mean_t(e_t)||` over every saturated frame in the val set (and, in the stratified table, over every frame at that dps). The loss being optimised is different: `train_e2e.py:224` penalises each window's own mean. Yaw spin is injected with a fixed sign (`scripts/collect_dr.py:110`) but the transverse rates are symmetric, so their per-episode biases average toward 0 and are counted as jitter. A 7–8× drop in this number does not mean the error that integrates inside one flight got 7–8× smaller. The log line "train bias 7.65 → 0.58" is not even that statistic: it is a Huber of the window mean on one minibatch (`train_e2e.py:265`), while the table's 3.02 is the global mean.
Fix: report the mean over episodes of `||mean_{t in episode}(w_hat - w)||`, which is what `scripts/e4_closed_loop.py:195-196` already computes, and stop quoting the Huber scalar as rad/s.

**2. The residual `s` is built with true specific thrust, so the lever-arm inversion never sees the mass error the priors are supposed to have.**
`scripts/collect_dr.py:167` and `:197` log `tom = sum(thrust) / M` with the true mass. The accelerometer was generated from that same quantity (`gpd_me/env_faulty.py:91-93`, `MetaShutDown7.py:160`). `gpd_me/e2e.py:112` and `scripts/train_e2e.py:64-66` then set `s = a_m - tom * e_z` and invert `|w|^2 = |s| / k`. Identified `g_T` is only a feature and a slow-head target (`train_e2e.py:206`, `:232-233`); the physics residual also subtracts logged `tom`, not `g_T * sum(u)` (`train_e2e.py:218-220`). A 20 % mass error is ~2 m/s² on the specific-force term, the same order as the lever-arm signal. The published rate numbers are for an estimator that was handed the nuisance term already removed.
Fix: form `tom` from the command (after the same actuator lag the plant uses) times identified `g_T` only. Re-train and re-score. If the gap over the clipped gyro disappears, the claim is false.

**3. `k` is `||r||`, not `|r_perp|`, and the scan's `k` is thrown away.**
`scripts/train_e2e.py:63` sets `k = ||prior[:3]||`. `gpd_me/e2e.py:94` then uses `|w|^2 = |s| / k`. The identity in `gpd_me/observer.py:9-11` is `|s| = |w|^2 |r_perp|` with `r_perp = r - (r·ŵ)ŵ`. `gpd_me/priors.py:108-111` returns the full least-squares `r` (and `use_tangential=False`). The nominal excitation is pure yaw (`scripts/collect_dr.py:162-163`: equal thrust plus a differential; roll/pitch moments cancel), so `r_z` is in the nullspace of `ωωᵀ - |ω|²I` and the ridge in `observer.py:198` shrinks it. After the fault the spin is tens of degrees off body z, so the `|r_perp|` that matters is a different number. When `||r||` is too big, `n2 - known < 0` and `e2e.py:100` floors the saturated axis at 0, which is worse than the clip. That matches "algebraic worse than clipped" without needing the low-dps story.
Fix: keep the scalar the scan actually minimised, or drop the scalar and solve the saturated axis from `s = (ωωᵀ - |ω|² I) r` with the vector `r`. Evaluate `|r_perp|` at the current `ŵ`, not `||r||`.

**4. Attitude, `ωdot`, and the spectral loss are integrated at 200 Hz on data that is not 200 Hz.**
Collection draws the control rate from `{100, 200, 250, 400}` (`scripts/collect_dr.py:38`, `:82-83`) and stores `dt`. The trainer ignores it: `scripts/train_e2e.py:211` (`dt=0.005`), `:214` (`* 200`), `:188-189` (DFT bin > 10 Hz assuming 5 ms). Three quarters of episodes therefore train the INS and physics losses at the wrong timestep (2× at 100 Hz, 0.5× at 400 Hz). `WINDOW = 48` is 240 ms only at 200 Hz (`gpd_me/e2e.py:30`). Rate error in rad/s does not use `dt`, but it is the output of a network whose attitude and physics gradients were wrong on most of the data.
Fix: pass per-episode `dt` into `ins_rollout`, the finite difference, and `make_dft`. Score tilt only on the 200 Hz slice until that is in.

**5. INS labels are one frame ahead of the integrated attitude.**
Rates and quaternions are logged at the same instant, before `env.step` (`scripts/collect_dr.py:164-171`). `gpd_me/e2e.py:56-60` sets `R_hat[:, k] = R0 @ exp(w_0 dt)…exp(w_k dt)`, and `scripts/train_e2e.py:133-134` sets `R_true[:, 0] = R0`. With a perfect rate, `R_hat[:, k]` matches `R_true[:, k+1]`, and the loss compares it to `R_true[:, k]` (`train_e2e.py:212`). At 40 rad/s that is ~11° of rotation per step about an axis that is not body z, i.e. several degrees of tilt even with zero rate error. Clipped and estimated rates move the thrust axis by different amounts, so the bias is not common-mode.
Fix: supervise `R_hat[:, :-1]` against `R_true[:, 1:]` (or integrate `w[k]` from `R[k]` to predict `R[k+1]` and drop the first label).

**6. The stratified tilt column cannot be produced; the exception is swallowed.**
`scripts/train_e2e.py:307-327`. `e_*` and `satf_np` have length `B * 48`. `t_n` is the first `tilt_frames` (default 12, `:154-156`) and has length `B * 12`. `t[m]` with a boolean mask of the wrong length raises, and `:330-331` prints a warning and continues. The aggregate tilt at `:284-287` is a `.mean()` over every frame of every window, not over saturated frames, while README's caption says the tilt sits on saturated frames. The checkpoint is written at `:272` before this block, so a "successful" run can save weights and never emit the table.
Fix: compute tilt on the same index set as the rate error (per saturated frame, or per window, one horizon). Delete the `try/except`. Fail the run if the shapes differ.

**7. The slow head never enters `w_hat`, so it cannot be what "compensates for weak identification".**
`gpd_me/e2e.py:147-149`: `w_hat = w_alg + sat * delta`. `delta` does not depend on `corr`. `scripts/train_e2e.py:204-208` uses `corr` only inside `r_id`, `g_T`, `G`, `T` for the physics, torque, and prior losses. Stage 2 then regresses `r_id` and `g_T` onto the ground-truth lever and `1/M` (`:94-96`, `:231-233`; `G` and `T` in that target are hardcoded 0 and unused). From the stage-2 switch onward, physics gradients on `delta` are taken at the true `r`. At test time `w_hat` still does not read `r_id`. The rate table is a fast residual on a frozen `w_alg`, co-adapted to oracle geometry.
Fix: either feed `r_id` back into a differentiable `w_alg` (and stop supervising it with `true`), or remove the slow head from the claim. Ablate `--no_slow` and expect a zero rate change with the current graph; a nonzero change means the coupling was added.

**8. The in-range anchor is identically zero and has zero gradient. It is not evidence the net tracks the gyro.**
On unsaturation `gpd_me/e2e.py:96-107` leaves `w_alg = gyro`, and `:148` multiplies `delta` by `sat`. `scripts/train_e2e.py:225-226` applies the anchor only where `INR` is true, i.e. where `sat` is already false. The residual is 0 for any weights, and `∂w_hat/∂delta = 0` there. "Anchor 0.000 throughout" will print for a randomly initialised net.
Fix: delete it, or apply it to something the parameters can change (unsaturated-axis bias/scale, or a path that is not multiplied by `sat`). An ablation that drops `anchor` must move the val rate; if it does not, do not cite the term.

**9. Windows are cut out of a concatenated stream, so train and val both integrate across episode boundaries.**
`scripts/train_e2e.py:97` concatenates episodes. `:128-133` slices `[i-WINDOW:i]` with no boundary check. `R0`, `ω`, and the GRU state at those indices mix two flights. With 48-frame windows that is about one window in forty. A single discontinuity of tens of rad/s moves a global mean (`:291`) by ~1 rad/s, which is the size of the reported network bias.
Fix: store `episode_id` and sample `i` only when `episode_id[i-WINDOW] == episode_id[i-1]`.

**10. The 1.07 rad/s "cross-episode" result is not what `train_estimator.py` does.**
`scripts/train_estimator.py:161-178` builds overlapping windows (stride 4, length 12) and splits them 80/20 at random. Neighbouring windows share 8 frames, so val frames are in the train set. There is no episode hold-out, and the default is 24 episodes, not "trained on 8 of 40". The same run also does not randomise the rigid-body mass: `:68-69` writes `env.M` after `shut_down_rotors` → `reset()`, which has already called `changeDynamics(..., mass=1)` (`MetaShutDown7.py:367-372`). `env.M` only rescales `last_acc`. `KM` and `delay` do take effect; mass and inertia do not.
Fix: split by episode id before windowing. Call `changeDynamics` for mass and inertia after `reset()`, as `collect_dr.py:106-108` does. Re-quote the number only from that run.

## MAJOR

**11. `G` and `T` are fit to ground-truth yaw rate, and a failed fit becomes a "yaw should be 0" loss.**
`gpd_me/priors.py:146` passes `omega_true[:, 2]` into the ARX. The in-range gyro is what a vehicle actually has; truth removes bias, scale, and noise. `scripts/collect_dr.py:153-159` also sizes the yaw manoeuvre from `omega_true`. `id_thrust_gain` (`priors.py:46-48`) picks the hover samples from ground-truth velocity. `np.nan_to_num` (`priors.py:149`) maps a failed ARX to `G = T = 0`. The torque loss (`train_e2e.py:222`) is then `huber(w_z / 10)` on every frame, a pull toward zero yaw rate. The order-2 mask (`priors.py:67-74`) mutates `in_range` in place and demands a long run of Trues, so short in-range stretches return NaN and hit this path. It also does not guarantee the lags `k-1`, `k-2` are themselves in range.
Fix: regress the in-range gyro, not `omega`. Gate hover on specific force, and restrict it to the pre-fault segment (the collector knows `t_fault`; `id_thrust_gain` only drops the first 0.3 s, `priors.py:44-45`). On failure, drop the torque term for that episode instead of substituting 0.

**12. `scripts/fix_priors.py` rewrites shards with a worse mass identifier.**
`:29-32` calls `identify_priors` without `vel`, so `id_thrust_gain` takes every post-skip sample (`priors.py:51-52`), including post-failure thrust, which is not `Mg`. It still passes logged `omega` (truth). It overwrites the npz in place (`:34`).
Fix: pass a pre-fault mask and the in-range gyro. Write a new directory. Do not treat repaired shards as comparable to the ones the table was trained on.

**13. "INDI excitation" is a random walk. Sample-and-hold never runs.**
`scripts/collect_dr.py:115` draws `"indi"`, `:121-126` sets `indi_ok` and never reads it, and `:191-192` falls through to `_excite_random`. A quarter of episodes are labelled INDI and flown open-loop. `IMUConfig.sample_hz` is left at `inf`, and `gpd_me/env_faulty.py:92-93` calls `measure(..., force=True)`, which skips the hold in `gpd_me/imu.py:120`. The README's 100/200/250/400 Hz "sample rates" are the physics step, not the IMU.
Fix: actually step `INDIController`, or stop listing it. If you want sample-and-hold, set `sample_hz` and pass `force=False` on ordinary steps.

**14. Multi-axis "ratio preservation" preserves the clip, not the rate.**
`gpd_me/e2e.py:101-106`. Once two axes sit on `±w_max`, `gyro[m]` has equal absolute value, so the reconstruction splits the leftover magnitude in that ratio. The true ratio is not in the measurement. Combined with item 3, `w_alg` on the 100–400 dps rows is not the lever-arm inverse.
Fix: for a known vector `r`, solve the quadratic in the saturated components from `s = ω×(ω×r) + ω̇×r`. Do not use `|s|/||r||`.

**15. Closed-loop deployment does not match the training features, and the onset is exactly where the context is fake.**
`scripts/e4_closed_loop.py:149-166` creates `NetRate` after the fault, so the GRU never sees the identification phase it was trained on. `:73-78` pads by repeating the first post-fault sample. `:80-84` pads the 2 s summary the same way, whereas training uses the real causal window (`train_e2e.py:73-85`). `:160` feeds `last_action`; training logs the command applied on that step (`collect_dr.py:165`). `:77` again uses `k = ||prior[:3]||`. `identify_priors` is called on `d["w"]` (`e4_closed_loop.py:119`, `:140`), i.e. truth. `shut_down_rotors` (`e4_closed_loop.py:143`, `MetaShutDown7.py:536-538`) calls `reset()` before the eval flight; with this script's fixed plant the mass happens not to change, but the IMU biases and `last_action` do.
Fix: keep one continuous episode, carry the nominal-phase buffer across the fault, and feed the same `u` and `tom` definition the trainer used. Do not call `reset()` between identification and eval.

**16. Collection is flown on the ground-truth rate. The val metric never closes the loop.**
`scripts/collect_dr.py:92` and `:185-186`: the policy observation is `omega_true` and the true attitude. The published errors are open-loop residuals on that distribution. E4 exists and is not in the README results. Injecting a better rate into a loop whose attitude is still integrated from the clipped gyro was already shown to be worse; nothing in `train_e2e.py` tests the joint injection.
Fix: the falsification flight below. Treat open-loop saturated-frame L1 as a diagnostic, not as evidence the vehicle stays up.

**17. Last iterate is the only checkpoint, and it is the overfit one.**
`scripts/train_e2e.py:272` saves after `iters`, with no best-val copy. The README's own curve says val rate was best at iteration 5000 (13.1) and worse at the end (17.9). The table is the worse point. The saved dict also omits `mode` and `ablate`, while `e4_closed_loop.py:87-90` always adds `w_alg` and masks by `sat`. A `--mode noalg` checkpoint silently gets a second copy of `w_alg` at deployment.
Fix: save the minimum-val state and store `mode`, `dt` handling, and the feature version in the checkpoint. Refuse to load a mismatch.

**18. The regularity numbers that justify the network are fit to truth, on one trajectory.**
`scripts/analyze_regularity.py:91-93` and `:109` fit and initialise the yaw model on ground-truth `w_z`. `:126-128` builds `k` from true `|w|^2`, and the printed "true `|r_perp|`" is `||r||` (the arm is in the xy-plane, so they coincide only for spin along z). `:145-163` initialises the KF at true `w_z` and updates it on that same flight. `Q = 1e-6`. The 0.19 rad/s figure is an oracle-initialised filter on the training trajectory, not a gyro-only held-out error.
Fix: fit `a, b` and `k` on the in-range gyro only, start the free-run from the last in-range gyro sample, and average over seeds and masks. Publish that number next to the old one.

## MINOR

**19. Aggregate rate error is mean L1, and the printed "jitter (std)" is not a standard deviation.**
`scripts/train_e2e.py:280` is `mean(|e_x|+|e_y|+|e_z|)`. `:296` labels `mean(||e - mean(e)||)` as std. The old estimator reports mean L2 (`scripts/train_estimator.py:226`). The 12.12 rad/s and 1.07 rad/s figures are not the same functional.
Fix: report RMSE of the 3-vector, and call the second moment what it is.

**20. `META[:, 0]` is not the fault mask.**
`scripts/train_e2e.py:93` stores `mask[1]`. That is 1 for both `[0,1,1,1]` and `[0,1,0,1]`, and 0 for `[0,0,1,1]`. Any mask breakdown from this column merges single-rotor failure with the diagonal pair.
Fix: store `flag` (already written into the npz `meta`).

**21. Spectral penalty is yaw-only, and the high-frequency bin is shared across the batch.**
`scripts/train_e2e.py:227-229`. The DFT ratio's denominator is a single batch scalar, so one loud window sets the scale for every other window. The 7.1 Hz figure came from one in-range yaw trace (`analyze_regularity.py:79-86`), including the hole-filled FFT of the in-range subset.
Fix: per-window ratio on the full `ω` vector, with the cutoff in Hz using that episode's `dt`.

**22. `AttitudeINS.yaw_error` is the full rotation angle.**
`gpd_me/ins.py:56-61` computes `z` and never uses it. Nothing calls it. `estimator.py:83` still does `clamp(-1, 1)` before `arccos`; `e2e.py:64-67` was fixed because that derivative is infinite. Retraining the old net hits the NaN you already diagnosed.
Fix: yaw = angle of `R_trueᵀ R_est` about `R_true e_z`. Use the strict interior clamp in `estimator.py`.

**23. Stage-2 prior target stores `G = T = 0`.**
`scripts/train_e2e.py:95`. Harmless only because `:232-233` ignores those two entries. `dG` and `dT` (`:207-208`) are otherwise an unbounded residual trained only through the torque loss.
Fix: either supervise them against an ARX fit to the gyro, or stop emitting them.

## NIT

**24. `id_yaw_channel` includes an affine term `c` and then drops it when converting to `G`.**
`gpd_me/priors.py:76` and `:90`. The torque loss has no matching bias, so a thrust-trim offset is absorbed into `G`.
Fix: match the loss to the regressor (include `c`, or fit the homogeneous model).

**25. Dead code on the E4 path.**
`scripts/e4_closed_loop.py:176-178` (`weight`, `w_prev`) and `scripts/collect_dr.py:138` (`yid_t0`) are unused. `coarse_summary` in `gpd_me/e2e.py:121-126` is not what training uses; E4 does use it. They currently match at 200 Hz and will drift the moment one is edited.
Fix: one function, called from both.

## What to run first

Falsify the central claim with one flight, not another open-loop L1:

Adjacent-pair failure, 1000 and 2000 dps, INS and the policy rate input both driven by the same signal, multi-seed. Three sources: truth, clipped gyro, network. The network must build `s` and `w_alg` from the in-range gyro and from `g_T_hat * u_command` only — no `sum(thrust)/M`, no `omega_true` in the ARX, no `||r||` standing in for `|r_perp|`. Score per-episode `||mean error||` over the saturated segment, tilt of the INS thrust axis against truth at 60 ms with the real `dt` and the one-step alignment fixed, and whether altitude stays up. The claim is false if the network does not beat clipped on per-episode bias and tilt under that interface, even if the current table still shows a gap.

## Missing experiments

The flags `--ablate` and `--no_slow` exist (`train_e2e.py:157-163`) and no result from them is in the README. A reviewer expects these, on an episode-safe split, at fixed 200 Hz, with the oracle `tom` removed:

- command-derived `tom` versus true `sum(thrust)/M` (this is the one that can zero out the result)
- scalar `||r||` versus vector lever-arm inversion
- per-episode bias versus the global mean, same checkpoint
- best-val checkpoint versus last iterate
- `--no_slow` and anchor-removed (both should be null with the current graph; if the paper needs them not to be, the graph is wrong)
- train on truth-rate flights versus train with the estimate already in the INS and the policy (the covariate shift item 16)
- 3-rotor failure as a held-out mask (it is excluded from `collect_dr.py:32-36` and it is the case the earlier sections say is fatal)
- gyro-only ARX versus truth-`ω` ARX, reported as a prior-error distribution, not a single median
