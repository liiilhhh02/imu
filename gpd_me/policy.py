"""Policy + outer position PID, importable without pybullet (so Gazebo can reuse them).

`PositionPID` is a line-by-line port of `RLControl.RLShutDownControl` (only the pybullet
`getMatrixFromQuaternion` call is replaced by an explicit quaternion -> matrix conversion), so the
Gazebo controller and the pybullet controller share the exact same outer loop.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from gym_pybullet_drones.algo.ACRL import ACRL

DEFAULT_MODEL_DIR = "/home/liiil/Downloads/gym-pybullet-drones/gym_pybullet_drones/model"

# prop positions from the senior's cf2x.urdf: prop0..prop3 at (+-0.1273, +-0.1273)
PROP_XY = np.array([[0.1273, 0.1273], [-0.1273, 0.1273], [-0.1273, -0.1273], [0.1273, -0.1273]])


class ACRLArgs:
    """Notebook hyper-parameters (net shapes must match the saved checkpoints)."""
    state_dim = 15
    num_critics = 2
    hidden_dim_actor = 64
    hidden_dim_critic = 128
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    gamma = 0.99
    tau = 0.005
    eta = 0


def load_policy(ckpt: str, model_dir: str = DEFAULT_MODEL_DIR, device: str = "cpu") -> ACRL:
    dev = torch.device(device)
    _orig = torch.load
    torch.load = lambda *a, **k: _orig(*a, **{**k, "map_location": k.get("map_location", dev)})
    policy = ACRL(state_dim=15, action_dim=4, max_action=1, device=dev, args=ACRLArgs())
    policy.load(ckpt, model_dir)
    return policy


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """World<-body rotation matrix for a ``[x, y, z, w]`` quaternion (pybullet/Gazebo convention)."""
    x, y, z, w = np.asarray(q, float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class PositionPID:
    """Port of ``RLControl.RLShutDownControl`` (outer loop -> (target specific thrust, body-z dir))."""

    def __init__(self, g: float = 9.8):
        self.g = g
        self.P_COEFF_FOR = np.array([1.2, 1.2, 1.4])
        self.I_COEFF_FOR = np.array([0.0, 0.0, 0.5])
        self.D_COEFF_FOR = np.array([1.0, 1.0, 1.4])
        self.reset()

    def reset(self):
        self.integral_pos_e = np.zeros(3)

    def step(self, dt, cur_pos, cur_quat, cur_vel, target_pos,
             target_vel=None, clip_angle=np.inf):
        R = quat_to_matrix(cur_quat)
        pos_e = np.asarray(target_pos, float) - np.asarray(cur_pos, float)
        vel_e = (np.zeros(3) if target_vel is None else np.asarray(target_vel, float)) \
            - np.asarray(cur_vel, float)
        self.integral_pos_e = np.clip(self.integral_pos_e + pos_e * dt, -2.0, 2.0)
        target_thrust = (self.P_COEFF_FOR * pos_e + self.I_COEFF_FOR * self.integral_pos_e
                         + self.D_COEFF_FOR * vel_e + np.array([0.0, 0.0, self.g]))
        target_thrust[0] = np.clip(target_thrust[0], -clip_angle, clip_angle)
        target_thrust[1] = np.clip(target_thrust[1], -clip_angle, clip_angle)
        scalar_acc = float(np.clip(np.dot(target_thrust, R[:, 2]), 0.0, np.inf))
        z_body = target_thrust / np.linalg.norm(target_thrust)
        return scalar_acc, z_body


class ActuatorLag:
    """First-order rotor-thrust lag of `MetaBaseAviary4._physics` (delay [s])."""

    def __init__(self, delay: float = 0.026, n_rotors: int = 4):
        self.delay = delay
        self.thrust = np.zeros(n_rotors)

    def reset(self):
        self.thrust = np.zeros_like(self.thrust)

    def step(self, forces: np.ndarray, dt: float) -> np.ndarray:
        forces = np.asarray(forces, float)
        for i in range(4):
            if forces[i] == 0.0:
                self.thrust[i] = self.thrust[i] * np.exp(-2 * dt / self.delay)
            else:
                n0_ratio = np.sqrt(max(self.thrust[i], 0.0) / forces[i])
                n_ratio = 1.0 + (n0_ratio - 1.0) * np.exp(-dt / self.delay)
                self.thrust[i] = forces[i] * (n_ratio ** 2)
        return self.thrust.copy()


def mixer_body_wrench(thrust: np.ndarray, km: float, damping_torque: np.ndarray | None = None):
    """Equivalent wrench at the CoM for ``thrust`` (N, 4) applied along body z at the prop offsets."""
    t = np.asarray(thrust, float)
    force = np.array([0.0, 0.0, float(t.sum())])
    tau_x = float(np.sum(PROP_XY[:, 1] * t))
    tau_y = float(-np.sum(PROP_XY[:, 0] * t))
    tau_z = float(km * (t[0] - t[1] + t[2] - t[3]))
    torque = np.array([tau_x, tau_y, tau_z])
    if damping_torque is not None:
        torque = torque + np.asarray(damping_torque, float)
    return force, torque