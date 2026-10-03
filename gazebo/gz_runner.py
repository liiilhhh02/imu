#!/usr/bin/env python3
"""The pybullet rotor-failure experiment, ported to Gazebo Classic (ROS2 Humble).

Same control stack as `me/scripts/verify_observer.py`:
  outer position PID (`gpd_me.policy.PositionPID`, a port of RLShutDownControl)
  -> 15-D observation -> ACRL inner policy -> per-rotor thrust [0, 15] N
  -> first-order thrust lag -> equivalent wrench at the CoM -> `/apply_link_wrench`

The IMU is the same model as in pybullet (`gpd_me.imu`): saturating gyro + lever-arm accelerometer
built from Gazebo's *true* base-link state.  `--src` selects what the controller is allowed to see:

    truth     ground-truth body rate            (upper bound)
    measured  saturated gyro                    (the failure case)
    override  lever-arm observer reconstruction (route A)

Gazebo wrench semantics learned the hard way (see README): a finite `duration` combined with an
unset `start_time` makes the wrench expire immediately, so this runner always uses duration = -1
and relies on re-application replacing the previous wrench.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose, Twist, Wrench
from rosgraph_msgs.msg import Clock

REPO = "/home/liiil/Downloads/gym-pybullet-drones"
ME = "/home/liiil/Downloads/me"
sys.path[:0] = [REPO, ME]

from gpd_me.imu import IMU, IMUConfig              # noqa: E402
from gpd_me.observer import LeverArmObserver       # noqa: E402
from gpd_me.policy import (ActuatorLag, PositionPID, load_policy, mixer_body_wrench,  # noqa: E402
                           quat_to_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "cf2x_gazebo.urdf")
MODEL, LINK = "cf2", "cf2::base_link"
SHUT_DOWN = {0: [0, 1, 1, 1], 1: [0, 1, 0, 1], 2: [0, 0, 0, 1], 3: [0, 0, 1, 1]}
CKPT = {0: "shutdown_real_7", 2: "shutdown_real_7_4"}
LEVER_ARM = (-0.012, -0.0055, 0.0)


class Runner(Node):
    def __init__(self, a):
        super().__init__("gz_shutdown_ctrl")
        self.a = a
        self.state = None
        self.sim_t = None
        self.create_subscription(ModelStates, "/model_states", self._on_states, 10)
        self.create_subscription(Clock, "/clock", self._on_clock,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.del_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.set_cli = self.create_client(SetEntityState, "/set_entity_state")
        self.wr_pub = self.create_publisher(Wrench, "/cf2/cmd_wrench", 1)

    def _on_states(self, m):
        if MODEL in m.name:
            i = m.name.index(MODEL)
            p, t = m.pose[i], m.twist[i]
            self.state = (np.array([p.position.x, p.position.y, p.position.z]),
                          np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
                          np.array([t.linear.x, t.linear.y, t.linear.z]),
                          np.array([t.angular.x, t.angular.y, t.angular.z]))

    def _on_clock(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def _call(self, cli, req, t=3.0):
        f = cli.call_async(req)
        t0 = time.time()
        while not f.done() and time.time() - t0 < t:
            time.sleep(0.0002)
        return f.result() if f.done() else None

    def setup(self):
        self.spawn_cli.wait_for_service(timeout_sec=10.0)
        d = DeleteEntity.Request(); d.name = MODEL
        self._call(self.del_cli, d)
        time.sleep(0.4)
        r = SpawnEntity.Request(); r.name = MODEL; r.xml = open(URDF).read()
        r.initial_pose = Pose(); r.initial_pose.position.z = 1.0
        out = self._call(self.spawn_cli, r)
        print(f"[gz] spawn success={out.success if out else None}")
        t0 = time.time()
        while self.state is None and time.time() - t0 < 8:
            time.sleep(0.05)
        if self.a.spin_init > 0:
            st = EntityState(); st.name = MODEL
            st.pose = Pose(); st.pose.position.z = 1.0; st.pose.orientation.w = 1.0
            st.twist = Twist(); st.twist.angular.z = -float(self.a.spin_init)
            out = self._call(self.set_cli, SetEntityState.Request(state=st))
            print(f"[gz] initial spin {-self.a.spin_init:.1f} rad/s  set={out.success if out else None}")
            time.sleep(0.2)

    def apply(self, force, torque):
        """Publishes the body-frame wrench; the gazebo_ros_force plugin latches and applies it."""
        w = Wrench()
        w.force.x, w.force.y, w.force.z = [float(v) for v in force]
        w.torque.x, w.torque.y, w.torque.z = [float(v) for v in torque]
        self.wr_pub.publish(w)
        return True


def build_obs(rel_xy, rate, target_a, thrust_over_mass, last_action, mask):
    return np.array([rel_xy[0], rel_xy[1],
                     rate[0] / 10.0, rate[1] / 10.0, rate[2] / 50.0,
                     (target_a - 9.8) / 3.0, (thrust_over_mass - 9.8) / 3.0,
                     *last_action, *(mask * 2 - 1)], dtype=np.float32)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--flag", type=int, default=0, choices=[0, 1, 2, 3])
    pa.add_argument("--dps", type=float, default=1000.0)
    pa.add_argument("--src", default="measured", choices=["truth", "measured", "override"])
    pa.add_argument("--steps", type=int, default=2000)
    pa.add_argument("--dt", type=float, default=0.005)
    pa.add_argument("--mass", type=float, default=1.0)
    pa.add_argument("--km", type=float, default=0.01)
    pa.add_argument("--delay", type=float, default=0.026)
    pa.add_argument("--kappa", type=float, default=0.0025, help="body-rate damping torque coeff")
    pa.add_argument("--spin_init", type=float, default=0.0)
    pa.add_argument("--target_pos", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    pa.add_argument("--calib_steps", type=int, default=0)
    pa.add_argument("--tag", default="")
    a = pa.parse_args()

    rclpy.init()
    n = Runner(a)
    ex = MultiThreadedExecutor(); ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    n.setup()

    policy = load_policy(CKPT[a.flag]) if a.flag in CKPT else None
    pid = PositionPID(); lag = ActuatorLag(a.delay)
    imu = IMU(IMUConfig(gyro_range_dps=a.dps, lever_arm=LEVER_ARM,
                        gyro_noise_std=0.05, accel_noise_std=0.02))
    obs_est = LeverArmObserver(gyro_limit_dps=a.dps) if a.src == "override" else None
    mask = np.array(SHUT_DOWN[a.flag], dtype=float)
    target_pos = np.array(a.target_pos, dtype=float)
    last_action = np.zeros(4)
    prev_omega, calib = None, {k: [] for k in ("g", "a", "t")}

    log = {k: [] for k in ("z", "xy", "wt", "wu", "rate")}
    t_wall = time.time()
    for i in range(a.steps):
        pos, quat, vel, omega_w = n.state
        R = quat_to_matrix(quat)
        omega = R.T @ omega_w
        wdot = np.zeros(3) if prev_omega is None else (omega - prev_omega) / a.dt
        prev_omega = omega.copy()
        tom = float(lag.thrust.sum()) / a.mass
        gyro, accel = imu.measure(omega, wdot, tom, a.dt, force=True)
        if a.src == "truth":
            rate = omega
        elif a.src == "measured":
            rate = gyro
        else:
            if i < a.calib_steps:
                calib["g"].append(gyro); calib["a"].append(accel); calib["t"].append(tom)
                rate = gyro
            else:
                if obs_est.k is None:
                    try:
                        obs_est.calibrate_inrange(calib["g"], calib["a"], calib["t"])
                    except ValueError:
                        obs_est.calibrate_joint(calib["g"], calib["a"], calib["t"])
                rate = obs_est.step(gyro, accel, tom)

        target_a, z_body = pid.step(a.dt, pos, quat, vel, target_pos)
        r_xy = np.hypot(z_body[0], z_body[1])
        if r_xy > 0.26:
            s = 0.26 / r_xy
            z_body = np.array([z_body[0] * s, z_body[1] * s,
                               np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
        rel = R.T @ z_body
        obs = build_obs(rel[:2], rate, target_a, tom, last_action * mask, mask)
        action = (policy.select_action(obs, deterministic=True) if policy is not None
                  else np.zeros(4))
        forces = np.clip((action + 1.0) * 7.5, 0.0, 15.0) * mask
        thrust = lag.step(forces, a.dt)
        force, torque = mixer_body_wrench(thrust, a.km, damping_torque=-a.kappa * omega)
        if not n.apply(force, torque):
            print("[gz] wrench apply failed")
        last_action = action.copy()

        log["z"].append(float(pos[2])); log["xy"].append(float(np.hypot(pos[0], pos[1])))
        log["wt"].append(omega.copy()); log["wu"].append(np.asarray(rate).copy())
        sleep = a.dt - (time.time() - t_wall - i * a.dt)
        if sleep > 0:
            time.sleep(sleep)
    elapsed = time.time() - t_wall

    z = np.array(log["z"]); xy = np.array(log["xy"])
    wt = np.array(log["wt"]); wu = np.array(log["wu"])
    w = max(10, a.steps // 5)
    lim = np.deg2rad(a.dps)
    sat = (np.abs(wt) > lim).any(axis=1)
    err = np.linalg.norm(wu - wt, axis=1)
    tag = a.tag or (f"gz flag={a.flag} dps={a.dps:.0f} src={a.src}")
    print(f"RESULT {tag}: z_last={z[-w:].mean():7.2f} (std {z[-w:].std():5.2f}) "
          f"xy_last={xy[-w:].mean():6.3f} |w|={np.linalg.norm(wt[-w:], axis=1).mean():6.1f} rad/s "
          f"wz={wt[-w:, 2].mean():7.2f} sat={100*sat.mean():4.0f}% "
          f"rate_err={err[sat].mean() if sat.any() else 0:6.2f} "
          f"loop={a.steps/elapsed:5.1f}Hz")
    ex.shutdown(); rclpy.shutdown()


if __name__ == "__main__":
    main()