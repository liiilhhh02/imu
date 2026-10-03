"""`MetaShutDown7` + a physically-modelled, range-limited IMU.

The observation layout is left unchanged (15-D) so the senior's checkpoints still load:

    [ des_rad(2) | rate(3) | norm_target_a(1) | norm_last_acc(1) | last_action(4) | mask(4) ]

Only the semantics of dims 2-4 changes: instead of the simulator's ground-truth body rate they
now carry whatever the "controller" is allowed to see, selected by ``RATE_SOURCE``:

    "measured"  saturated, noisy gyro                                  (the failure case)
    "truth"     ground-truth body rate                                 (upper bound / reference)
    "override"  whatever was written into ``env.rate_override``        (the estimated rate)

Ground truth is never destroyed: ``omega_true``, ``omega_dot``, ``accel_meas``, ``lever_arm``,
``theta_true`` stay available for supervision and evaluation.
"""
from __future__ import annotations

import numpy as np

from gym_pybullet_drones.envs.MetaShutDown7 import MetaAviary as _ShutDown7

from .imu import IMU, IMUConfig
from .ins import AttitudeINS


class RateCorruptor:
    """Surrogate for the *estimator's* error, used to train a policy that tolerates it.

    Why this exists: the deployed controller never sees the true body rate -- it sees the network's
    estimate, and the attitude it uses is the INS integrated from that same estimate.  Training the
    policy on the true rate and then deploying it with an estimate is exactly the kind of hidden
    mismatch that produces a controller which "works in simulation" only.  So the policy is fine-tuned
    against a corruption model whose statistics are *measured from the real estimator*
    (`docs/STATUS.md` §4: per-flight bias, jitter, and the direction error that dominates under
    multi-axis saturation), with the same corrupt rate driving both the controller input and the INS.

    This is a surrogate only for *training*; every reported closed-loop number is produced with the
    real network in the loop, never with this model.
    """

    def __init__(self, rng: np.random.Generator, bias: float = 1.5, scale: float = 0.06,
                 dir_deg: float = 8.0, noise: float = 1.0, lag_s: float = 0.015):
        self.rng = rng
        self.bias = float(bias)            # rad/s, per-axis constant offset
        self.scale = float(scale)          # relative magnitude error (the identified `k` error)
        self.dir_deg = float(dir_deg)      # spin-axis direction error (dominant at 3-axis saturation)
        self.noise = float(noise)          # rad/s, per-sample
        self.lag_s = float(lag_s)
        self.b = rng.normal(0.0, self.bias / 3.0, size=3)
        self.s = 1.0 + rng.normal(0.0, self.scale / 3.0)
        ax = rng.normal(size=3)
        self.axis = ax / max(np.linalg.norm(ax), 1e-9)
        self.ang = np.deg2rad(rng.normal(0.0, self.dir_deg / 3.0))
        self._state = None

    def reset(self, rng: np.random.Generator):
        self.__init__(rng, self.bias, self.scale, self.dir_deg, self.noise, self.lag_s)

    def __call__(self, omega_true: np.ndarray, dt: float) -> np.ndarray:
        w = np.asarray(omega_true, float)
        if self._state is None:
            self._state = w.copy()
        a = float(np.exp(-dt / max(self.lag_s, 1e-6)))
        self._state = a * self._state + (1.0 - a) * w              # estimator lag
        v = self.s * self._state + self.b + self.rng.normal(0.0, self.noise / 3.0, size=3)
        K = _rodrigues(self.axis * self.ang)                        # direction error
        return K @ v


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(rotvec))
    if th < 1e-12:
        return np.eye(3)
    k = rotvec / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


class MetaAviaryFaulty(_ShutDown7):
    """Rotor-failure aviary whose controller only sees a saturating, lever-arm-corrupted IMU."""

    IMU_CFG = IMUConfig()

    def __init__(self, *args, imu_cfg: IMUConfig | None = None, rate_source: str = "measured",
                 att_source: str = "truth", corrupt_cfg: dict | None = None, **kwargs):
        self.imu = IMU(imu_cfg or self.IMU_CFG)
        self.RATE_SOURCE = rate_source
        self.ATT_SOURCE = att_source          # "truth" | "ins" (deployment-faithful attitude)
        self.rate_override = None
        self.ins = AttitudeINS()
        self._rng = np.random.default_rng(0)
        self.corruptor = RateCorruptor(self._rng, **(corrupt_cfg or {})) if corrupt_cfg else None
        self.corrupt_rate = np.zeros(3)
        # measurement / ground-truth channels
        self.omega_true = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.gyro_meas = np.zeros(3)
        self.accel_meas = np.zeros(3)
        self.thrust_over_mass = 0.0
        self._prev_omega = None
        self._omega_at_sample = None
        self._imu_counter = -1
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)   # applies M/KM/delay/inertia when eval=False
        self.imu.reset()                       # new lever arm + bias for this episode
        # the INS is initialised from the attitude at episode start (a real one is initialised from
        # its accelerometer/attitude alignment before the failure), then dead-reckons from the rate
        # the controller is given -- that is the no-leakage attitude channel
        from scipy.spatial.transform import Rotation as _Rot
        self.ins.reset(_Rot.from_quat(np.asarray(self.quat[0], float)).as_matrix())
        if self.corruptor is not None:
            self.corruptor.reset(self._rng)
        self.corrupt_rate = np.zeros(3)
        self._prev_omega = None
        self._omega_at_sample = None
        self._imu_counter = -1
        self.omega_dot = np.zeros(3)
        self.rate_override = None
        return self._computeObs()

    # ------------------------------------------------------------------ IMU plumbing
    @property
    def lever_arm(self) -> np.ndarray:
        return self.imu.lever_arm

    @property
    def theta_true(self) -> dict:
        """Physical parameters a real controller would have to identify."""
        return dict(lever_arm=self.imu.lever_arm.copy(), M=float(self.M), KM=float(self.KM),
                    delay=float(self.delay), gyro_limit=self.imu.gyro_limit)

    def _imu_step(self) -> np.ndarray:
        """Advances the IMU at most once per control step, but always on a genuine state change.

        ``_imu_counter`` suppresses duplicate sampling when ``_computeObs`` is called twice within
        one control step; comparing the body rate catches the simulator mutating the state without
        a physics step (the senior's ``shut_down_rotors`` writes an initial body rate after
        ``reset()``), which must produce a fresh measurement instead of a stale zero-rate one.
        """
        omega = np.asarray(self.ang_vel, dtype=float)
        fresh = self._imu_counter != self.step_counter
        mutated = not np.array_equal(omega, self._omega_at_sample)
        if not fresh and not mutated:
            return self.gyro_meas
        dt = self.TIMESTEP * self.AGGR_PHY_STEPS if fresh else 0.0
        if fresh and self._prev_omega is not None:
            self.omega_dot = (omega - self._prev_omega) / dt
        else:
            self.omega_dot = np.zeros(3)          # no physics time elapsed
        self._prev_omega = omega.copy()
        self._omega_at_sample = omega.copy()
        self._imu_counter = self.step_counter
        self.omega_true = omega
        self.thrust_over_mass = float(np.atleast_1d(self.last_acc)[0])
        self.gyro_meas, self.accel_meas = self.imu.measure(omega, self.omega_dot,
                                                          self.thrust_over_mass, dt, force=True)
        return self.gyro_meas

    def _controller_rate(self) -> np.ndarray:
        src = self.RATE_SOURCE
        if src == "truth":
            return self.omega_true
        if src == "override":
            if self.rate_override is None:
                raise RuntimeError("RATE_SOURCE='override' but env.rate_override is None")
            return np.asarray(self.rate_override, dtype=float)
        if src == "corrupt":
            if self.corruptor is None:
                raise RuntimeError("RATE_SOURCE='corrupt' but no corrupt_cfg was given")
            self.corrupt_rate = self.corruptor(self.omega_true, self.TIMESTEP * self.AGGR_PHY_STEPS)
            return self.corrupt_rate
        return self.gyro_meas

    # ------------------------------------------------------------------ observation
    def _computeObs(self):
        obs = super()._computeObs()        # sets ang_vel/last_acc/att_rad_error from ground truth
        cnt_before = self._imu_counter
        self._imu_step()
        fresh = self._imu_counter != cnt_before     # a genuine new sample (see _imu_step's guard)
        rate = self._controller_rate()
        obs[2:5] = np.array([rate[0] / 10.0, rate[1] / 10.0, rate[2] / 50.0], dtype=np.float32)
        if self.ATT_SOURCE == "ins":
            # The attitude a real vehicle has: the INS integrated from the *same* rate the controller
            # is given.  Dims 0:2 are the target body-z direction expressed with that attitude, i.e.
            # the env's own formula (`rel_z_body = R.T @ target_z_body`) but with the estimated R.
            #
            # Integrate **once per control step**, not once per `_computeObs` call: the supervisor's
            # loops call `_computeObs()` at the top of the step *and* `step()` calls it again, and
            # integrating twice makes the INS turn at twice the true rate (measured: the closed loop
            # diverges even with the true rate as input).  Same reason the IMU has `_imu_counter`.
            dt = self.TIMESTEP * self.AGGR_PHY_STEPS
            if fresh:
                self.ins.update(rate, dt)
            rel = self.ins.R.T @ np.asarray(self.target_z_body, float)
            self.des_rad = np.array([rel[0], rel[1]])
            self.att_rad_error = float(np.arccos(np.clip(rel[2], -1.0, 1.0)))
            obs[0:2] = self.des_rad.astype(np.float32)
        return obs

    # ------------------------------------------------------------------ helpers for estimators
    def imu_vector(self) -> np.ndarray:
        """Raw 6-D IMU frame ``[gyro(3), accel(3)]`` as a controller would receive it."""
        return np.concatenate([self.gyro_meas, self.accel_meas])

    def measured_rate_is_saturated(self) -> np.ndarray:
        """Per-axis boolean: is the gyro pinned at the range limit right now?"""
        return self.imu.saturated_axes(self.gyro_meas)


def make_faulty_env(env_mod=None, freq: int = 200, gui: bool = False,
                    gyro_range_dps: float = 1000.0, lever_arm=None,
                    lever_arm_radius: float = 0.02, seed: int = 0,
                    rate_source: str = "measured", **imu_kwargs):
    """Convenience factory mirroring the senior's env constructor."""
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    cfg = IMUConfig(gyro_range_dps=gyro_range_dps, lever_arm=lever_arm,
                    lever_arm_radius=lever_arm_radius, seed=seed, **imu_kwargs)
    env = MetaAviaryFaulty(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        physics=Physics("pyb"), aggregate_phy_steps=1,
        freq=freq, gui=gui, record=False, obstacles=False,
        imu_cfg=cfg, rate_source=rate_source)
    env.eval = False
    env.reset()
    return env