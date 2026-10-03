# Adversarial code review request — end-to-end IMU out-of-range rate estimator

You are reviewing a research codebase. **READ-ONLY: do not edit, create, or delete any file; do not
run training or collection (long jobs are currently running and must not be disturbed).**
Reading files and reasoning is all I want.

## Context (what the code is trying to do)

A quadrotor loses rotors. It must then spin fast (e.g. 40 rad/s = 2300 deg/s) to stay controllable, so
a 1000–4000 dps rate gyro **saturates**. The idea under test: the accelerometer mounted at a lever arm
`r` from the CoM measures centripetal acceleration, carrying `|s| = |omega|^2 * |r_perp|` **beyond the
gyro range**; identify the scale from the *in-range* phase, invert it when saturated, and let a learned
temporal network supply what the algebra cannot (transients, multi-axis saturation, identification
error). The network's estimate must then drive **both** the controller's rate input and the attitude
INS (an earlier experiment showed injecting it in only one place is harmful).

Everything lives in `/home/liiil/Downloads/me`:

- `gpd_me/imu.py` — gyro/accelerometer model: saturation, bias, scale error, noise, lever arm,
  sample-and-hold.
- `gpd_me/ins.py` — attitude dead-reckoned from the (clipped) gyro.
- `gpd_me/env_faulty.py` — the PyBullet env: the controller only sees measurable signals; ground truth
  is bypassed for supervision.
- `gpd_me/observer.py` — analytic lever-arm observer: in-range r identification, 1-D scale scan,
  closed-form reconstruction.
- `gpd_me/priors.py` — identification of `r`, `g_T = 1/M`, `G` (yaw command -> yaw rate DC gain),
  `T` (yaw time constant) **from flight data only** (nominal phase + post-failure transient).
- `gpd_me/e2e.py` — the end-to-end network: features from raw measurables only, differentiable INS,
  losses (rate / attitude / physics / torque / anchor / spectral / bias), `w_hat = w_alg + sat*delta`.
- `gpd_me/estimator.py, policy.py, indi.py` — earlier residual estimator, policy/PID helpers, and a
  port of the supervisor's traditional INDI baseline.
- `scripts/collect_dr.py` — domain-randomised data collection (episode = nominal identification phase
  -> fault injection -> post-failure regime).
- `scripts/train_e2e.py` — training + evaluation (declared metrics: stable bias, jitter, rate error,
  short-horizon INS tilt, stratified by gyro range).
- `scripts/*.py` — verification and experiment drivers (`verify_imu.py`, `verify_observer.py`,
  `verify_ins_attitude.py`, `main_experiment.py`, `e4_closed_loop.py`, `fix_priors.py`,
  `analyze_regularity.py`).
- `README.md` — the claim set, including measured numbers.

## What I want from you

A **prioritised, adversarial** list of problems. Be specific and evidence-based (`file:line`), and for
each item give a concrete fix. Rank by severity. Concentrate on, in this order:

1. **Correctness bugs** — physics/algebra: signs, frames (body vs world), units (dps vs rad/s, N vs
   commanded thrust), indexing/slicing off-by-one, feature-order mismatches, `arccos`/`sqrt` gradient
   and NaN hazards, anything that makes a reported number an artifact.
2. **Data leakage / invalid metrics** — is any evaluation using information the deployed system could
   not have? Are normalisation constants, episode splits, or the "truth" used in a way that inflates
   results? Are the declared metrics (bias vs jitter vs tolerance error) actually measuring what the
   README claims? Is the "in-range anchor is structurally 0" claim sound?
3. **Training-setup pitfalls** — silent no-ops (parameters set then overwritten), domain randomisation
   that does not actually take effect, loss terms that are dead or dominate, detach/zero-init
   mistakes, overfitting disguised as generalisation.
4. **Robustness / reproducibility** — anything that makes the pipeline fail silently, rely on a stale
   artifact, or produce a plausible-looking but wrong checkpoint.
5. **Whether the README's conclusions are supported by the code** — flag any claim that the code
   cannot actually produce.

Also state explicitly: what you would test first to falsify the central claim, and any *missing*
experiment or ablation that a reviewer would expect.

Output: markdown, severity-tagged (BLOCKER / MAJOR / MINOR / NIT), each with file:line, why it matters,
and the fix. No praise, no summary of what the code does.