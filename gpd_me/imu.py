"""IMU model for the rotor-failure regime: range saturation + lever-arm (centripetal) effect.

Physics
-------
A rate gyro + accelerometer are rigidly mounted at a body-frame offset ``r`` from the CoM and
measure, in the *body* frame:

    gyro  :  w_m = clip( w + b_g + n_g , +-w_max )                       [rad/s]
    accel :  a_m = (T/M) * e_z  +  wdot x r  +  w x (w x r)  +  b_a + n_a  [m/s^2]

where
    * ``(T/M) * e_z`` is the non-gravitational specific force at the CoM produced by the rotor
      thrust (this is exactly the senior's ``last_acc = sum(thrust)/M``, always along body z),
    * ``wdot x r`` is the tangential lever-arm term (only significant during spin-up),
    * ``w x (w x r)`` is the centripetal term, which grows like ``|w|^2`` and therefore still
      carries information about ``w`` **after the gyro has saturated**,
    * gravity does *not* appear: an accelerometer measures specific force, so at the CoM the
      reading is purely ``(T/M) e_z`` (no attitude information in this thrust-along-z setup).

``truth_specific_force`` and ``measure`` are two separate code paths on purpose:
:mod:`scripts.verify_imu` checks ``measure`` against a pybullet finite-difference ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEG2RAD = np.pi / 180.0


@dataclass
class IMUConfig:
    """Sensor + mounting configuration. Distances in m, rates in rad/s, accel in m/s^2."""
    gyro_range_dps: float = 1000.0      # saturation limit of the rate gyro; np.inf = unlimited
    accel_range: float = np.inf         # saturation limit of the accelerometer
    gyro_noise_std: float = 0.05        # senior's MetaShutDown7.gyro_noise_std
    accel_noise_std: float = 0.02       # senior's MetaShutDown7.imu_noise_std
    gyro_bias_std: float = 0.0          # constant per episode
    gyro_scale_std: float = 0.0         # multiplicative per-axis scale error (per episode)
    accel_bias_std: float = 0.0         # constant per episode
    sample_hz: float = np.inf           # sample-and-hold rate; np.inf = sample every step
    lever_arm: tuple | None = None      # fixed body-frame offset; None = randomize on reset()
    lever_arm_radius: float = 0.02      # |r| upper bound when lever_arm is None
    seed: int = 0


class IMU:
    """Saturating gyro + lever-arm accelerometer attached to the drone body."""

    def __init__(self, cfg: IMUConfig | None = None):
        self.cfg = cfg or IMUConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.gyro_limit = self.cfg.gyro_range_dps * DEG2RAD
        self._lever_arm = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.gyro_scale = np.ones(3)
        self.accel_bias = np.zeros(3)
        self._hold = np.zeros(3 + 3)         # last (gyro, accel) when sample-and-hold is active
        self._t_since_sample = np.inf
        self.reset()

    # ------------------------------------------------------------------ mounting / calibration
    def reset(self, lever_arm: np.ndarray | None = None) -> None:
        """Samples a new lever arm / bias (per episode) and clears the sample-and-hold state."""
        cfg = self.cfg
        if lever_arm is not None:
            self._lever_arm = np.asarray(lever_arm, dtype=float).copy()
        elif cfg.lever_arm is not None:
            self._lever_arm = np.asarray(cfg.lever_arm, dtype=float).copy()
        else:
            # uniform inside a ball of radius lever_arm_radius (direction uniform on the sphere)
            u = self.rng.normal(size=3)
            u /= np.linalg.norm(u) + 1e-12
            self._lever_arm = u * cfg.lever_arm_radius * self.rng.uniform(0.3, 1.0) ** (1 / 3)
        self.gyro_bias = self.rng.normal(0.0, cfg.gyro_bias_std, 3) if cfg.gyro_bias_std > 0 else np.zeros(3)
        self.gyro_scale = 1.0 + (self.rng.normal(0.0, cfg.gyro_scale_std, 3)
                                 if cfg.gyro_scale_std > 0 else np.zeros(3))
        self.accel_bias = self.rng.normal(0.0, cfg.accel_bias_std, 3) if cfg.accel_bias_std > 0 else np.zeros(3)
        self._hold = np.zeros(6)
        self._t_since_sample = np.inf

    @property
    def lever_arm(self) -> np.ndarray:
        return self._lever_arm

    # ------------------------------------------------------------------ physics
    def specific_force_com(self, thrust_over_mass: float) -> np.ndarray:
        """Non-gravitational specific force at the CoM, in the body frame."""
        return np.array([0.0, 0.0, float(thrust_over_mass)])

    def lever_arm_terms(self, omega: np.ndarray, omega_dot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(centripetal, tangential)`` lever-arm accelerations, body frame."""
        r = self._lever_arm
        centripetal = np.cross(omega, np.cross(omega, r))
        tangential = np.cross(omega_dot, r)
        return centripetal, tangential

    def specific_force_truth(self, omega: np.ndarray, omega_dot: np.ndarray,
                             thrust_over_mass: float) -> np.ndarray:
        """Ideal (noise-free, unsaturated) accelerometer output, body frame."""
        return self.specific_force_com(thrust_over_mass) + self.specific_force_imu(omega, omega_dot)

    def specific_force_imu(self, omega: np.ndarray, omega_dot: np.ndarray) -> np.ndarray:
        """Lever-arm contribution only (what the CoM accelerometer would miss)."""
        centripetal, tangential = self.lever_arm_terms(omega, omega_dot)
        return centripetal + tangential

    # ------------------------------------------------------------------ measurements
    def measure(self, omega: np.ndarray, omega_dot: np.ndarray, thrust_over_mass: float,
                dt: float, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Saturated, noisy ``(gyro, accel)`` pair sampled at ``dt`` (sample-and-hold aware).

        ``force=True`` samples unconditionally (used when the simulator state is mutated without
        advancing a physics step, e.g. ``shut_down_rotors`` writing an initial body rate).
        """
        cfg = self.cfg
        self._t_since_sample += dt
        period = np.inf if not np.isfinite(cfg.sample_hz) else 1.0 / cfg.sample_hz
        if force or (not np.isfinite(period)) or (self._t_since_sample >= period):
            self._t_since_sample = 0.0
            gyro = (omega + self.gyro_bias) * self.gyro_scale
            accel = self.specific_force_truth(omega, omega_dot, thrust_over_mass) + self.accel_bias
            if cfg.gyro_noise_std > 0:
                gyro = gyro + self.rng.normal(0.0, cfg.gyro_noise_std, 3)
            if cfg.accel_noise_std > 0:
                accel = accel + self.rng.normal(0.0, cfg.accel_noise_std, 3)
            self._hold = np.concatenate([gyro, accel])
        gyro, accel = self._hold[:3], self._hold[3:]
        gyro = np.clip(gyro, -self.gyro_limit, self.gyro_limit)
        if np.isfinite(cfg.accel_range):
            accel = np.clip(accel, -cfg.accel_range, cfg.accel_range)
        return gyro.copy(), accel.copy()

    # ------------------------------------------------------------------ analysis helpers
    def saturated_axes(self, gyro: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Boolean mask of the axes pinned at the range limit (as the driver would report them)."""
        return np.abs(np.abs(gyro) - self.gyro_limit) <= tol

    @staticmethod
    def accel_residual(accel: np.ndarray, thrust_over_mass: float,
                       omega: np.ndarray, lever_arm: np.ndarray) -> np.ndarray:
        """``a_m - (T/M)e_z - (w x (w x r))``; zero only if ``omega``/``r`` are correct."""
        return accel - np.array([0.0, 0.0, float(thrust_over_mass)]) - np.cross(omega, np.cross(omega, lever_arm))