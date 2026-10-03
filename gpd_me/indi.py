"""Port of the senior's `examples/baseline_INDI.ipynb` — the *traditional-control* baseline.

Structure kept 1:1 with the notebook (same class names, gains and ordering):

    PositionController          P on position -> velocity -> desired acceleration
                                -> desired thrust direction n_des + its filtered derivative
    PrimaryAxisAttitudeController   reduced attitude (body z -> n_des, i.e. yaw surrendered)
                                -> desired p, q and their derivatives + desired specific force
    INDIController              incremental NDI:
                                    u = G^-1 ( v_in - ydot_f ) + u_f
                                with G = diag(1/Ix, 1/Iy, 1/m) @ A_failed,
                                ydot_f = [d(p)/dt, d(q)/dt, f_z] and v_in the desired
                                [p_dot_des + k1(p_des-p), q_dot_des + k2(q_des-q), f_z_des + k3*I]

The allocation matrix excludes the failed rotor (`FAILED_MOTOR = 0`, valid = [1,2,3]) which is
exactly the single-rotor-failure case.

Why this baseline is the right one for the over-range question: `p, q` enter through a *derivative*
(`LPF.calc_with_derivative`), so if the gyro clips, the differentiated rate is wrong; and `r` enters
the attitude controller's gyroscopic compensation term directly.  These are the two places where a
traditional controller depends on the rate measurement much more strongly than the senior's RL policy
does (whose reward never contained the angular rate).
"""
from __future__ import annotations

import numpy as np

BW = 50
BODY_LENGTH = 0.125
FAILED_MOTOR = 0
VALID_MOTORS = [i for i in range(4) if i != FAILED_MOTOR]

ALLOCATION_FULL = np.array([
    [BODY_LENGTH, BODY_LENGTH, -BODY_LENGTH, -BODY_LENGTH],    # tau_x (roll)
    [-BODY_LENGTH, BODY_LENGTH, BODY_LENGTH, -BODY_LENGTH],    # tau_y (pitch)
    [1, 1, 1, 1],                                              # f_z
])
ALLOCATION_FAILED = ALLOCATION_FULL[:, VALID_MOTORS]


class LPF(object):
    """First-order low pass (notebook implementation)."""

    def __init__(self, ts, cutoff_freq, data):
        self.ts = ts
        self.cutoff_freq = cutoff_freq
        self.last_output = np.zeros_like(data, dtype=float) if isinstance(data, np.ndarray) else 0.0

    def calc(self, input):
        output = (self.cutoff_freq * self.ts * input + self.last_output) / (self.cutoff_freq * self.ts + 1)
        self.last_output = output
        return output

    def calc_with_derivative(self, input):
        output = (self.cutoff_freq * self.ts * input + self.last_output) / (self.cutoff_freq * self.ts + 1)
        derivative = (output - self.last_output) / self.ts
        self.last_output = output
        return output, derivative


class PositionController(object):
    def __init__(self, ts):
        self.ts = ts
        self.kp_pos = np.array([[1], [1], [1]])
        self.kp_vel = np.array([[2], [2], [6]])
        self.ki_vel = np.array([[0], [0], [0]])
        self.int_lim = 5.0
        self.max_vel = 10.0
        self.max_angle = 10.0 / 57.3
        self.max_lateral = abs(9.81 * np.tan(self.max_angle))
        self.integrals = np.zeros((3, 1))
        self.acc_I_des = np.zeros((3, 1))
        self.g = np.array([[0], [0], [-9.81]])
        self.n_des_I = np.zeros((3, 1))
        self.n_des_I_lpf = LPF(self.ts, BW, self.n_des_I)

    def calc(self, pos_target, pos_real, vel_real):
        pos_err = pos_target - pos_real
        vel_target = np.clip(self.kp_pos * pos_err, -self.max_vel, self.max_vel)
        vel_err = vel_target - vel_real
        self.integrals = np.clip(self.integrals + vel_err * self.ts, -self.int_lim, self.int_lim)
        self.acc_I_des = self.kp_vel * vel_err + self.ki_vel * self.integrals
        lat_ratio = np.linalg.norm(self.acc_I_des[:2, 0]) / self.max_lateral
        if lat_ratio > 1:
            self.acc_I_des[:2, 0] /= lat_ratio
        self.acc_I_des[2, 0] = np.clip(self.acc_I_des[2, 0], -5, 5)
        self.n_des_I = (self.acc_I_des - self.g) / np.linalg.norm(self.acc_I_des - self.g)
        _, n_des_I_dot = self.n_des_I_lpf.calc_with_derivative(self.n_des_I)
        n_des_I_dot = np.clip(n_des_I_dot, -0.5, 0.5)
        return self.acc_I_des, self.n_des_I, n_des_I_dot


class PrimaryAxisAttitudeController(object):
    def __init__(self, ts):
        self.ts = ts
        self.n_B = np.array([[0], [0], [1]])
        self.kx = 5
        self.ky = 5
        self.p_des_lpf = LPF(self.ts, BW, 0.0)
        self.q_des_lpf = LPF(self.ts, BW, 0.0)
        self.f_z_des_lpf = LPF(self.ts, BW, 0.0)
        self.g = np.array([[0], [0], [-9.81]])

    def calc(self, R, r, acc_I_des, n_des_I, n_des_I_dot):
        n_des_B = R.T @ n_des_I
        h1, h2, h3 = n_des_B[0, 0], n_des_B[1, 0], n_des_B[2, 0]
        vout = np.array([[self.kx * (0 - h1)], [self.ky * (0 - h2)]])
        temp = np.array([[0, 1 / h3], [-1 / h3, 0]])
        n_des_I_hat_dot = (R.T @ n_des_I_dot)[:2, 0]
        temp1 = temp @ (vout - r * np.array([[h2], [-h1]]) - n_des_I_hat_dot.reshape(2, 1))
        p_des, p_des_dot = self.p_des_lpf.calc_with_derivative(temp1[0, 0])
        q_des, q_des_dot = self.q_des_lpf.calc_with_derivative(temp1[1, 0])
        f_z_des = np.linalg.norm(acc_I_des - self.g) / self.n_B[2, 0]
        f_z_des = self.f_z_des_lpf.calc(f_z_des)
        return p_des, q_des, f_z_des, p_des_dot, q_des_dot


class INDIController(object):
    def __init__(self, ts, allocation_failed=ALLOCATION_FAILED, Ix=0.0045, Iy=0.0045, mass=0.72,
                 valid_motors=None):
        self.ts = ts
        self.Ix, self.Iy, self.mass = Ix, Iy, mass
        self.valid = list(VALID_MOTORS if valid_motors is None else valid_motors)
        if allocation_failed is None or self.valid != VALID_MOTORS:
            allocation_failed = ALLOCATION_FULL[:, self.valid]
        self.G = np.diagflat([1 / Ix, 1 / Iy, 1 / mass]) @ allocation_failed
        # 3 objectives ([tau_x/Ix, tau_y/Iy, f_z/m]) from n_valid rotors.
        # n_valid == 3 -> exact inversion; n_valid < 3 -> rank deficient, least-squares (yaw row is
        # already absent from v_in, i.e. yaw was surrendered by the senior's design).
        self.Ginv = (np.linalg.inv(self.G) if self.G.shape[0] == self.G.shape[1]
                     else np.linalg.pinv(self.G))
        self.f_z_lpf = LPF(self.ts, BW, 0.0)
        self.p_lpf = LPF(self.ts, BW, 0.0)
        self.q_lpf = LPF(self.ts, BW, 0.0)
        self.u_lpf = LPF(self.ts, BW, np.zeros((allocation_failed.shape[1], 1)))
        self.k1, self.k2, self.k3 = 30, 30, 10
        self.integrals = 0.0

    def calc(self, p, p_des, p_des_dot, q, q_des, q_des_dot, f_z, f_z_des, u):
        p, p_dot = self.p_lpf.calc_with_derivative(p)
        q, q_dot = self.q_lpf.calc_with_derivative(q)
        f_z = self.f_z_lpf.calc(f_z)
        self.integrals = np.clip(self.integrals + (f_z_des - f_z) * self.ts, -1.0, 1.0)
        v_in = np.array([[p_des_dot + self.k1 * (p_des - p)],
                         [q_des_dot + self.k2 * (q_des - q)],
                         [f_z_des + self.k3 * self.integrals]])
        y_f_dot = np.array([[p_dot], [q_dot], [f_z]])
        u_f = self.u_lpf.calc(u)
        return self.Ginv @ (v_in - y_f_dot) + u_f