# `reference/` — provenance, and why the supervisor's code is not in this repository

The work builds directly on the supervisor's fork of **`utiasDSL/gym-pybullet-drones`** (checked out
at `/home/liiil/Downloads/gym-pybullet-drones`, commit `90769e3`).  Everything this repository
`import`s from that package (`gym_pybullet_drones.envs.MetaShutDown7`, `.algo.ACRL`,
`.control.RLAttitudeControl`, `.utils.MetaBuffer`, …) lives there; we treat it as a read-only
dependency.

`reference/senior/` holds a local snapshot of the supervisor's relevant files
(`envs/MetaShutDown*.py`, `envs/MetaBaseAviary4.py`, `algo/{ACRL,MetaSAC_ShutDown,MetaRL,Net}.py`,
`utils/{MetaBuffer,VaeBuffer,enums}.py`, `control/RLAttitudeControl.py`,
`examples/{ShutDownControlV7.ipynb,baseline_INDI.py,extract_params.ipynb}`, `assets/cf2x.urdf`).

**That directory is deliberately git-ignored**: it is the supervisor's unpublished work and we have no
licence to publish it.  To reproduce anything here you need his repository present; the absolute path
is configurable through the `REPO` constant at the top of every script (they default to
`/home/liiil/Downloads/gym-pybullet-drones`).

What we *do* publish is our own contribution: the IMU over-range model, the analytic lever-arm
observer and identification layer, the environment wrapper that hides ground truth from the
controller, the end-to-end estimator, the data collector, the training/evaluation drivers, the
verification scripts, and the review of all of it.