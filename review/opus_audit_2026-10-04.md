Read-only audit complete. Nothing modified.

## Bottom line

The plan is **already implemented** in `scripts/train_deploy.py` and **already running** (`results/rl/deploy_flag3/train.log`), so the question is not whether to do it but why the existing run is uninformative. It is uninformative for three independent reasons: it performs **4 gradient steps per 4800 transitions** (`scripts/train_deploy.py:281-282`), it stores **mis-paired (state, action) tuples** (`:144-150`), and it trains on a vehicle whose **lever arm differs from the one the priors were identified on** (`:66` vs `scripts/e4_closed_loop.py:141`). Separately, the plan is *not* leakage-clean: observation dim 6 is the simulator's true thrust, and prior `G`/`T` are identified from `omega_true`.

---

## 1. Legitimacy w.r.t. the no-leakage rule

Verdict: the *concept* is legitimate (training-time simulation may use anything; the policy at deployment consumes only `w_hat`, the INS attitude, the accelerometer, its own commands and the mask). But the chain as written has three real violations, all of which are also present in `e4_closed_loop.py`, i.e. the *acceptance* harness is leaking too.

**(i) Observation dim 6 is privileged, in training and at deployment.** `tom = env.thrust_over_mass` (`scripts/e4_closed_loop.py:258`, `scripts/train_deploy.py:124`) is `float(self.last_acc[0])` (`gpd_me/env_faulty.py:162`), and `last_acc = sum(self.thrust[0])/self.M` (`MetaShutDown7.py:160`) — the simulator's exact total thrust over mass, not a measurement. The docstring claims the opposite: "`tom` is the measured thrust-over-mass (the accelerometer's x sample), never a commanded or simulated quantity" (`gpd_me/e2e.py:298-299`). The legitimate substitute already exists in the repo: `tom_from_command(...)` (`gpd_me/priors.py:99`), which `NetRate` already uses internally (`gpd_me/e2e.py:267`).

**(ii) The prior vector is identified from the true body rate.** `identify_priors` takes `omega_true` and feeds it straight into the yaw-channel ARX: `G, T = id_yaw_channel(u_cmd, omega_true[:, 2], inr, dt, mask=mask_t)` (`gpd_me/priors.py:442`). The docstring only disclaims `tom` ("accepted for **diagnostics only**", `:420-421`) and says nothing about `omega_true`. `e4_closed_loop.py:213` passes `d["w"]`, which is `env.omega_true` (`:127`, `:171`). `G` and `T` then enter the network twice: as `prior[4]`/`prior[5]`, and as the 29th feature `w_z_model = G * sum(±u)` (`gpd_me/e2e.py:170-173`). Restricting to in-range samples (`priors.py:433`) makes the numerical effect small but does not make it legitimate, and it hides a real-vehicle errors-in-variables bias (gyro noise 0.05 rad/s would bias `G` low).

**(iii) The INS is initialised from the true attitude at fault time.** `AttitudeINS(quat_to_matrix(env.quat[0]))` (`scripts/e4_closed_loop.py:252`). Defensible as pre-fault alignment, but it should be stated, and `train_deploy.py:116` uses `np.eye(3)` instead — so the two chains differ here as well.

Not violations, for the record: the reward using true state, the anchor using `omega_true` as its rate source (`train_deploy.py:127`), and the outer PID using true position/velocity (`:131`) are all declared and are simulation-side only.

---

## 2. Concrete bugs that would make estimator-in-the-loop fine-tuning fail or silently cheat

**B1 — The fine-tune barely trains (fatal).** `for _ in range(4 if buffer.buffer_size >= a.batch else 0): policy.train(buffer, iterations=1)` (`scripts/train_deploy.py:281-282`). `ACRL.train` does one gradient step per `iterations` (`ACRL.py:538-566` pattern). One episode produces `8 envs × 600 steps = 4800` transitions, so the update-to-data ratio is 4/4800. `train_robust.py:237-238` does one update per control step, i.e. 600 per episode — **150× higher**. The 43 episodes in `results/rl/deploy_flag3/train.log` correspond to ~172 gradient steps total. No conclusion can be drawn from that run.

**B2 — The replay buffer stores `(s_{t-1}, a_t, r_t, s_t)`.** `act` is selected from `obs_batch` (= `s_t`) at `train_deploy.py:144`, copied to `prev_act` at `:145`, then stored against `prev_obs[i]` (= `s_{t-1}`) at `:149`; `prev_obs` is only advanced at `:150`. The critic therefore learns `Q(s_{t-1}, a_t)`. The docstring at `:104-107` claims exactly this class of bug is what the file exists to avoid.

**B3 — Lever-arm mismatch between the identification env and the training env.** Priors come from `identification()` in the acceptance env, lever arm `(-0.012, -0.0055, 0.0)` (`scripts/e4_closed_loop.py:141`); the training envs have lever arm `(0.013, 0.004, 0.002)` (`scripts/train_deploy.py:66`, same value in `train_robust.py:77`). Different magnitude *and* sign. The estimator's `k = prior[8]` and the whole algebraic front end (`gpd_me/e2e.py:267-269`) are therefore calibrated for a geometry the training vehicle does not have. Visible in the log: `k_med 0.005m` (`train.log:2`) against `‖r‖ = 0.0137 m` for the training lever arm.

**B4 — `NetRate`'s rolling window is never reset between episodes.** `self.buf` is created in `__init__` (`gpd_me/e2e.py:237`) and only ever appended to (`:260`); there is no `reset`. `train_deploy.py:215` builds `nets` once and only rewrites the priors each episode (`:277-278`). So from episode 2 onward the 96-frame window and the 400-frame coarse summary (`gpd_me/e2e.py:271-276`) are filled with the *previous* episode's post-divergence tail, and `buf` grows without bound. `evaluate()` builds a fresh `NetRate` each time (`:230`), so training and evaluation see structurally different estimators.

**B5 — `preseed` is used at deployment but not in training.** `e4_closed_loop.py:245-248` seeds the window with the real pre-fault history by default (`--no_preseed` to disable); `train_deploy.py` never calls `NetRate.preseed`. Combined with B4 this is a hard train/deploy mismatch in the estimator's input distribution.

**B6 — Eval draws are the training draws.** `evaluate()` is documented as "In-loop check on fresh draws" (`train_deploy.py:224`) but calls `draw_fault(env, ..., pool[k % a.pool])` (`:228`) with `eval_draws=4`, `pool=8` — eval draws 0-3 are training pool slots 0-3, same priors, same `fault_rng`, same injected spin. Worse, the pool is identified with seeds `a.seed + 100*k` = 0, 100, … (`:199`) while acceptance runs seeds `0..N-1` (`e4_closed_loop.py:371`), so **acceptance draw 0 is a training draw**. The policy also only ever sees 8 fixed initial conditions for the entire run.

**B7 — `--net_lp` is a no-op, and a published negative result rests on it.** `NET_LP`/`NET_LP_STATE` are declared (`e4_closed_loop.py:88-89`) and assigned in `main` (`:358`) but never read inside `run()`; `rate = net.step(...)` at `:268` is used unfiltered. The three "20/10/5 Hz → 0/8" runs in `docs/STATUS.md:556` were therefore three identical *unfiltered* runs. The low-pass hypothesis is untested, not refuted.

**B8 — The reported `k_hat` is not the `k` in use.** `run()` returns `k=float(np.linalg.norm(prior[:3]))` (`e4_closed_loop.py:302`), i.e. `‖r̂‖`, while the front end uses `prior[8] = |r_perp|` (`gpd_me/e2e.py:202`). `--k_override` writes `prior[8]` (`e4_closed_loop.py:376`) and so does not change the printed column at all. `prior[8]` is logged nowhere.

**B9 — Reward logging is divided by `len_episode` twice.** `episode()` returns `rew=rew_sum / a.len_episode` (`train_deploy.py:154`) and the log line prints `st['rew'] / a.len_episode` (`:285`). The reward trace, which is the primary forgetting signal, is off by 600×.

**B10 — Dead accelerometer-bias path.** `priors.py:342-344` calls `identify_lever_arm(..., fit_bias=False)` and then reads `ob.accel_bias`, which that branch unconditionally sets to `np.zeros(3)` (`observer.py:278`). `b_acc` is always zero, `accel_c = accel - b_acc` (`priors.py:349`) is a no-op, and the diagnostic string always prints `|b|=0.00`.

**B11 — No sensor or airframe randomisation in acceptance.** `e4_closed_loop.py:139-142` passes no `seed` to `IMUConfig`, so `IMU` uses `cfg.seed = 0` (`imu.py:45`, `:53`) for every one of the 12 "independent" draws — identical noise realisation throughout — and the lever arm is fixed, so `k`'s identifiability is only ever exercised at one geometry. All the airframe randomisation in `MetaShutDown7.reset` is commented out (`MetaShutDown7.py:356-380`).

**B12 — `train_robust.py` selects actions from a stale observation.** The action comes from `obs_batch` built at `:218` *before* `target_a`/`target_z_body` are updated at `:230`; `obs_batch[i]` is then recomputed at `:231` and discarded. The eval path does it the other way round (`:107-109`), as does deployment (`e4_closed_loop.py:285`). So `train_robust`'s obs dims 0, 1 and 5 are one step behind both its own eval and deployment.

---

## 3. The four specific checks

### (a) Is the training observation identical to `e4_closed_loop.py`'s, field by field?

**For `train_deploy.py`: yes for the 15 fields, no for what fills them.** `deploy_obs` (`gpd_me/e2e.py:301-305`) is used on both ends, and `PositionPID` is a faithful port of `RLShutDownControl` — same gains (`gpd_me/policy.py:60-62` vs `RLAttitudeControl.py:54-56`), same `self.G = 9.8` (`BaseControl.py:37`), same integral clamp, same `clip_angle` default. `e.TIMESTEP = DT`. The inputs differ, though: different lever arm (B3), window never reset and never preseeded (B4, B5), INS initialised to identity vs the true attitude (B6/§1-iii).

Note that `scripts/test_deploy_obs.py` does **not** verify this. It compares `deploy_obs` against a re-typed copy of the same formula (`:25-29`); it never touches `env._computeObs()`. The guard is weaker than its docstring claims.

**For `train_robust.py`: no, and the mismatch is in the leaking direction.** It feeds the env's own `_computeObs()` (`:218`, `:231`). Dims 0-1 and dim 5 are derived from `tz`/`ta` produced by `RLShutDownControl(cur_quat=raw[3:7])` (`:101-102`, `:222-224`) — the **true** quaternion — whereas deployment derives them from the INS attitude (`e4_closed_loop.py:272-273`). Dim 6 additionally carries `+N(0, 0.02)` that deployment does not (`MetaShutDown7.py:177`, `:186` vs `e4_closed_loop.py:281`). Dims 0, 1, 5, 6 are all different.

### (b) Does in-episode online identification give the same `k`? Which `k` is used at eval?

**There is no in-episode identification.** The priors are identified once per pool slot before training (`train_deploy.py:199`) by `identification()`, which needs a 3.0 s nominal phase plus a 1.5 s post-fault open-loop transient (`e4_closed_loop.py:197`, `:91`, `:206-208`) — several seconds of dedicated open-loop excitation that a 600-step training episode never performs. Each slot is then reused for `setup_episodes=40` episodes (`train_deploy.py:275`).

**The same prior is used at eval** (`:228` passes `pool[k % a.pool]`), which is why B6 is both a mismatch and a leak.

Two further points. The identification flight is *discarded*: `shut_down_rotors(3)` begins with `self.reset()` (`MetaShutDown7.py:537`), which reloads the URDF at the initial pose, so the vehicle that produced the priors and the vehicle that flies share only an RNG seed — the "a real vehicle has the pre-fault history" justification for `preseed` (`gpd_me/e2e.py:240-246`) does not hold in this harness. And the identified value is bad: `k_med 0.005 m` / range `[0.001, 0.012]` (`train.log:2`) against a true `‖r‖` of 0.0132 m (e4 geometry) or 0.0137 m (training geometry), consistent with the `k̂ 0.66-0.77 cm vs 1.32 cm` already recorded in `docs/STATUS.md:210`.

### (c) Is the fine-tuning reward the one the policy was trained with, and does it still make sense?

**In `train_robust.py`: no, and it is a live reward-hacking channel.** `_computeReward` uses `self.att_rad_error` (`MetaShutDown7.py:250`). With `att_source="ins"` (`train_robust.py:76`), `env_faulty._computeObs` **overwrites** `att_rad_error` with the INS-based angle (`gpd_me/env_faulty.py:204`), and `step()` calls `_computeObs()` before `_computeReward()` (`MetaBaseAviary4.py:289-290`). So the attitude reward is computed entirely in estimate space: the policy is paid for making the *estimate* look level while the real vehicle flips. `train_robust.py:116` and `:236` log that same corrupted `att_rad_error` as "att", so the log cannot detect the divergence either. This alone can explain recipes (b) and (c) "keeping nominal but 0/4 robust".

**In `train_deploy.py`: the form is right but the objective is misaligned with acceptance.** `att_source="truth"` (`:79`), so `att_rad_error` is the true-attitude angle. But the reward is `2·att + 0.5·alt + 1.0·action` with `alt_reward = -|target_a - last_acc|` and the velocity term weighted 0 (`MetaShutDown7.py:250-269`) — **there is no altitude term at all**, while the acceptance metric is `z > 0.3 and std(z) < 3` (`e4_closed_loop.py:312`). Both `target_z_body` and `target_a` are now set from the *estimated* attitude (`train_deploy.py:143`), so when the INS drifts the policy is rewarded for tracking a wrong setpoint. The log shows exactly this: `z` climbs monotonically from ~0 to 15.09 m over episodes 27-43 while `rew` stays flat at ≈-0.15 (`train.log:30-43`). The reward is being optimised; the acceptance criterion is not.

Also a task-distribution shift in both scripts: the original policy was trained as an attitude/thrust tracker with randomly drawn setpoints (`target_z_body = random_vector_deg()` within 15°, `target_a ~ U(7,13)`, `MetaShutDown7.py:539-540`). Under an outer PID on a drifting attitude, `scalar_acc = dot(target_thrust, R_est[:,2])` clipped at 0 (`RLAttitudeControl.py:258-259`) can collapse toward 0, driving obs dim 5 to ≈-3.3 — far outside `(7-9.8)/3 = -0.93 … (13-9.8)/3 = 1.07`. Out-of-distribution inputs on a warm-started actor are a plausible cause of the observed forgetting independent of the corruption model.

### (d) Where does the student see privileged state?

| Quantity | Seen? | Where |
|---|---|---|
| True thrust / mass | **Yes, in the observation, both paths** | `e4_closed_loop.py:258` + `env_faulty.py:162` + `MetaShutDown7.py:160`; `train_deploy.py:124` |
| True yaw rate | **Yes, via `prior[4]`/`prior[5]` and feature 29** | `priors.py:442`; `e2e.py:170-173` |
| True attitude | **Yes in `train_robust`** (dims 0,1,5 via `raw[3:7]`) | `train_robust.py:101-102`, `:222-224` |
| True attitude at t=0 | Yes, as INS initialisation | `e4_closed_loop.py:252` |
| True rate | No in the observation; yes as the anchor's rate source (declared) | `train_deploy.py:127` |
| True lever arm | No | `k` comes from `priors.id_lever_arm` only |
| True position / velocity | Yes, in the outer PID (declared) | `train_deploy.py:131`; `train_robust.py:25` |

---

## 4. Minimal-risk recipe

Fix before running anything; each item is cheap and each invalidates the run if skipped.

1. **B1**: `policy.train(buffer, iterations=K)` with `K = len_episode // 4` ≈ 150 per episode (UTD ≈ 1/32). Without this the experiment measures nothing.
2. **B2**: store `(obs_batch[i], act[i], rew, next_obs_batch[i], done)`. Buffer the current observation, step, rebuild the observation, then add — do not reuse `prev_act`.
3. **B3**: one lever arm everywhere. Pass the same `IMUConfig` object to `train_deploy.make_env` and `e4_closed_loop.make_env`; better, randomise it per episode and re-identify, since a single fixed geometry cannot show generalisation.
4. **B4/B5**: add `NetRate.reset()`, call it per episode, and `preseed` from `pool[slot][4]` exactly as `e4_closed_loop.py:245-248` does.
5. **B6**: eval pool disjoint from the training pool, and disjoint from acceptance seeds `0..11`. Use pool seeds `10000 + 100k` for training.
6. **Reward**: add an explicit altitude-hold term matching acceptance, e.g. `-|z - 1.0|` clipped, weight ~0.5. Without it the policy will keep finding the climb solution.

Then: **anchor 0.3** (keep it — the measured forgetting at anchor 0 is unambiguous, `train_robust.py:202-206`), **no corruption curriculum** (the estimator *is* the curriculum; there is no λ to ramp), **budget ≥ 2000 episodes** (`len_episode=600`, 8 envs ⇒ ~10 M transitions; at 0.06 ep/s that is ~9 h, so run it overnight rather than `--hours 3`), **pool ≥ 32 identification draws** re-identified every 20 episodes.

Log every episode: (i) **true** mean tilt error `AttitudeINS.tilt_error(ins.R, quat_to_matrix(env.quat[0]))`, never `env.att_rad_error`; (ii) per-step reward, correctly normalised (fix B9); (iii) `prior[8]` per env, not `‖r̂‖` (fix B8); (iv) `hold_steps` (first step with `z < 0.3`) as the continuous proxy for the binary hold.

Every 50 episodes evaluate on held-out draws at **both** rate sources and log `truth N/M` and `net N/M` side by side.

**Stopping rule.** Stop and accept when `net ≥ 0.9 × truth` on ≥ 24 held-out draws. Stop for forgetting when `truth` drops below `0.8 × its episode-0 value` on two consecutive evaluations — restore the last checkpoint above that line and halve the learning rate. Stop for futility at 2000 episodes.

**What falsifies the hypothesis.** The hypothesis is "the four failures were caused by corruption-model mismatch." It is falsified if, after fixes 1-6, with ≥ 1000 episodes and a healthy `truth` row, `net` stays at or below the current `2/4` baseline (`train.log:3`) while `hold_steps` shows no upward trend. It is also falsified more cheaply, and I would run this first: re-run `train_robust.py` with **only** the `att_source="ins"` reward leak fixed (B12/§3c) and the lever arm aligned. If the synthetic-corruption recipes then start working, the original diagnosis was a reward bug, not a corruption-model mismatch, and estimator-in-the-loop training is not needed.

**One cheap pre-registration.** Before spending 9 h: run `e4_closed_loop.py --src net` with `--k_override` set to the true `|r_perp|` for the acceptance geometry (and fix B8 so the column reports `prior[8]`). If the net row stays at 0/12 even with a perfect `k`, the estimator is not `k`-limited and no amount of policy fine-tuning against it will help either.

---

## 5. Is the hypothesis wrong, and what else

**I think it is wrong as stated, but for a different reason than the repo's own conclusion.** The repo says the gap is "qualitative, not incremental" (`docs/STATUS.md:542-543`). I do not think that is established, because two of the three experiments supporting it do not measure what they claim: the filtering refutation ran an inert flag (B7), and the fine-tuning refutation ran 172 gradient steps on mis-paired transitions on a mismatched lever arm (B1-B3). What *is* established is that the truth-rate baseline holds only 7/12 in pybullet and 1/12 in Gazebo. **That is the real blocker.** "≥ 90 % of truth" with truth at 7/12 means you must hit 7/12, and a 12-draw binary on a chaotic boundary where a 0.002 rad/s perturbation flips the outcome (`e4_closed_loop.py:27-28`) has no power to resolve a 90 % ratio. The acceptance harness is currently measuring chaos, and no amount of fine-tuning changes that.

So the highest-value work, in order:

**(1) Fix the measurement before fixing the controller.** Switch the primary metric from the binary hold to the paired continuous one already computed, `hold_steps` (`e4_closed_loop.py:301`), and report the paired per-draw difference `hold_steps(net) - hold_steps(truth)` with a Wilcoxon signed-rank over 50+ draws. This is a defensible read of "≥ 90 % of truth-rate performance," it has real statistical power, and it costs only compute. Also randomise the IMU seed and lever arm per draw (B11) so the 12 draws are actually independent.

**(2) The estimator-side fix that breaks the dead end — a tilt observer from inertial acceleration.** `gpd_me/ins.py:1-15` argues correctly that the *body-frame* accelerometer carries no attitude information, because gravity cancels in specific force and the only non-gravitational force is thrust along body z. But that argument shows the opposite in the *inertial* frame: if the body-frame specific force is `(T/M) e_z` and the inertial acceleration is `a_i = R (T/M) e_z - g e_z^world`, then

```
R[:, 2] = (a_i + g e_z^world) / |a_i + g e_z^world|
```

i.e. the thrust axis — exactly the reduced attitude the controller consumes (`RLAttitudeControl.py:258-261`) — is **directly observable** from differentiated velocity plus the accelerometer's own magnitude. No new sensor: the outer PID already consumes position and velocity (`train_deploy.py:131`, declared in `train_robust.py:25`), so the velocity channel is already assumed present. Fusing this absolute tilt measurement with the gyro in a complementary filter bounds the drift that `AttitudeINS` currently accumulates without limit (`gpd_me/ins.py:36-41`). This attacks the mechanism the repo itself identifies as the killer — bias integrating into attitude error, `docs/STATUS.md:495-496` — rather than attacking the rate RMS, which five estimator versions have already shown does not transfer.

**(3) The control-side fix that needs no new sensor.** The failure is attitude-loop divergence, and the policy's only attitude input is dims 0-1, `R_est.T @ z_body`. Under a drifting INS those two numbers are the corrupted ones. Replacing them with the tilt from (2) is a drop-in change to `deploy_obs` that does not touch the policy weights, and it can be A/B-tested against the current chain in one afternoon on the existing harness. If it moves `hold_steps`, *then* fine-tune. Fine-tuning a policy to tolerate an unboundedly drifting attitude estimate is asking it to solve an unobservable problem; fixing the observability first is strictly cheaper.

I would also raise the operating-point question with the professor directly. `docs/STATUS.md:207-208` already recommends it, and it is the honest framing: flag 3 with `shutdown_real_7_4` is a configuration where the *ideal* controller fails 25-42 % of the time, so a relative-to-truth criterion on it is not a measurement of the estimator. Flags 0 and 1 hold 3/3 in Gazebo (`docs/STATUS.md:596-597`) and would let the saturated-IMU contribution be demonstrated cleanly, which is what the thesis actually needs to show.
