"""Attitude from integrating the *measured* (range-limited) gyro — the missing half of the IMU model.

Why this matters here.  On this vehicle the accelerometer carries **no attitude information**: the
accelerometer measures specific force, and in free flight the only non-gravitational force is the
rotor thrust acting along body z, so

    f_body = R^T (a_inertial - g) = (T/M) e_z        (verified to 0.0026 m/s^2 against pybullet)

i.e. the gravity term cancels exactly and the reading is attitude-independent.  A conventional AHRS
tilt correction from the accelerometer therefore does *not* exist for this platform, so the gyro is
the only attitude sensor: if a gyro axis saturates, the attitude estimate integrates a too-small
rate and drifts without bound.

The INU is propagated with the same clipping that the controller sees: every consumer of a
saturated axis gets the range value, never the true one.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class AttitudeINS:
    """Dead-reckoning attitude: ``R <- R @ exp([omega_meas]_x dt)``.

    ``omega_meas`` is whatever the gyro output is (already clipped at the range and noisy), so the
    saturation error accumulates in the attitude exactly as it would on a real platform.
    """

    def __init__(self, R0: np.ndarray | None = None):
        self.R = np.eye(3) if R0 is None else np.asarray(R0, float).copy()

    def reset(self, R0: np.ndarray):
        self.R = np.asarray(R0, float).copy()

    def update(self, omega_meas: np.ndarray, dt: float) -> np.ndarray:
        omega_meas = np.asarray(omega_meas, float)
        norm = float(np.linalg.norm(omega_meas))
        if norm > 1e-12:
            self.R = self.R @ Rotation.from_rotvec(omega_meas * dt).as_matrix()
        return self.R

    @staticmethod
    def angle_error(R_est: np.ndarray, R_true: np.ndarray) -> float:
        """Rotation angle between two attitude estimates, in radians."""
        c = (np.trace(R_est.T @ R_true) - 1.0) / 2.0
        return float(np.arccos(np.clip(c, -1.0, 1.0)))

    @staticmethod
    def tilt_error(R_est: np.ndarray, R_true: np.ndarray) -> float:
        """Angle between the two thrust axes (what the reduced-attitude controller actually uses)."""
        z_e, z_t = R_est[:, 2], R_true[:, 2]
        return float(np.arccos(np.clip(float(z_e @ z_t), -1.0, 1.0)))

    @staticmethod
    def yaw_error(R_est: np.ndarray, R_true: np.ndarray) -> float:
        """Rotation about the *true* thrust axis (the component a reduced-attitude controller ignores)."""
        z = R_true[:, 2]
        a = R_true.T @ R_est
        c = (np.trace(a) - 1.0) / 2.0
        return float(np.arccos(np.clip(c, -1.0, 1.0)))