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


class MetaAviaryFaulty(_ShutDown7):
    """Rotor-failure aviary whose controller only sees a saturating, lever-arm-corrupted IMU."""

    IMU_CFG = IMUConfig()

    def __init__(self, *args, imu_cfg: IMUConfig | None = None, rate_source: str = "measured",
                 **kwargs):
        self.imu = IMU(imu_cfg or self.IMU_CFG)
        self.RATE_SOURCE = rate_source
        self.rate_override = None
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
        return self.gyro_meas

    # ------------------------------------------------------------------ observation
    def _computeObs(self):
        obs = super()._computeObs()        # sets ang_vel/last_acc/att_rad_error from ground truth
        self._imu_step()
        rate = self._controller_rate()
        obs[2:5] = np.array([rate[0] / 10.0, rate[1] / 10.0, rate[2] / 50.0], dtype=np.float32)
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