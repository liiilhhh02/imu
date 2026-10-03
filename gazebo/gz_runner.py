#!/usr/bin/env python3
"""The pybullet rotor-failure experiment, ported to Gazebo Classic (ROS2 Humble).

Same control stack as `scripts/study_truth_rate.py` (the reference loop):
  outer position PID (`gpd_me.policy.PositionPID`, a port of RLShutDownControl)
  -> 15-D observation (same construction as `MetaShutDown7._computeObs`) -> ACRL policy
  -> per-rotor thrust [0, 15] N -> first-order thrust lag -> wrench at the CoM.

Wrench path: the body-frame wrench is published on `/cf2/cmd_wrench`
(`libgazebo_ros_force` with `<force_frame>link</force_frame>`), which applies
`AddRelativeForce` + `AddRelativeTorque` at the CoM every physics step -- the same wrench
`gpd_me.policy.mixer_body_wrench` builds for the pybullet plant (whose `_physics` is
`MetaBaseAviary4._physics`; its yaw reaction is `KM*(t0-t1+t2-t3)`, matching the mixer).

Fault initial condition: `MetaShutDown7.shut_down_rotors(flag)` re-poses the drone at
[0,0,1] with a preset spin for flags 2/3 (`angularVelocity=[U(-3,3),U(-3,3),-U(24,26)]`).
Without that spin the adjacent-pair failure is unrecoverable even in pybullet (measured),
so the runner draws it too (`--seed` to reproduce a draw).

Loop synchronisation: one control step per `--substeps` physics steps, paced by the
`/model_states` publisher (whose `update_rate` in `me.world` therefore equals the physics
rate). `/clock` is published by `gazebo_ros_init` at only 10 Hz by default and its rate
cannot be changed at runtime (the Throttler is built once in Load()), so it must not be
used to pace a 200 Hz loop. The substep ratio is auto-detected from the measured state rate,
so the control loop stays at its 200 Hz design point.

`--src` selects what the controller is allowed to see:
    truth     ground-truth body rate            (upper bound)
    measured  saturated gyro                    (the failure case)
    override  lever-arm observer reconstruction (route A)
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
CKPT = {0: "shutdown_real_7", 1: "shutdown_real_7",
        2: "shutdown_real_7_4", 3: "shutdown_real_7_4"}
LEVER_ARM = (-0.012, -0.0055, 0.0)


class Runner(Node):
    def __init__(self, a):
        super().__init__("gz_shutdown_ctrl")
        self.a = a
        self.state = None
        self.states_received = 0
        self.state_hz = 200.0
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
            self.states_received += 1

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
        # drop any state cached from the previous model: only messages that arrive *after*
        # the spawn count as fresh, otherwise the control loop starts on a stale (already
        # fallen) pose and wastes the whole run chasing it.
        self.state = None
        self.states_received = 0
        r = SpawnEntity.Request(); r.name = MODEL; r.xml = open(URDF).read()
        r.initial_pose = Pose(); r.initial_pose.position.z = 1.0
        out = self._call(self.spawn_cli, r)
        print(f"[gz] spawn success={out.success if out else None}")
        t0 = time.time()
        while self.states_received == 0 and time.time() - t0 < 8:
            time.sleep(0.01)
        # measure the physics/state rate while the drone is still free-falling from the spawn
        # (the fault IC below re-poses it), so the control loop can be kept at its 200 Hz design
        # point whatever update_rate the world uses
        m0, w0 = self.states_received, time.time()
        time.sleep(0.4)
        self.state_hz = (self.states_received - m0) / max(time.time() - w0, 1e-9)
        self.state = None
        self.states_received = 0
        if self.a.spin_init > 0:
            omega0 = np.array([0.0, 0.0, -float(self.a.spin_init)])
        else:
            omega0 = fault_ic_spin(self.a.flag, np.random.default_rng(self.a.seed))
        st = EntityState(); st.name = MODEL
        st.pose = Pose(); st.pose.position.z = 1.0; st.pose.orientation.w = 1.0
        st.twist = Twist()
        st.twist.angular.x, st.twist.angular.y, st.twist.angular.z = [float(v) for v in omega0]
        st.reference_frame = "world"
        out = self._call(self.set_cli, SetEntityState.Request(state=st))
        print(f"[gz] fault IC spin={np.round(omega0, 3)} rad/s  set={out.success if out else None}")
        # start the loop on the first state message *after* the IC: any extra delay is free-fall
        # the controller did not cause (the reference loop applies the IC and the first action in
        # the same step), and >~2 steps of startup lag is enough to lose this chaotic run.
        self.state = None
        self.states_received = 0
        t0 = time.time()
        while self.states_received == 0 and time.time() - t0 < 2.0:
            time.sleep(0.001)

    def apply(self, force, torque):
        """Publishes the body-frame wrench; the gazebo_ros_force plugin latches and applies it."""
        w = Wrench()
        w.force.x, w.force.y, w.force.z = [float(v) for v in force]
        w.torque.x, w.torque.y, w.torque.z = [float(v) for v in torque]
        self.wr_pub.publish(w)
        return True


def fault_ic_spin(flag, rng):
    """Initial angular velocity (world frame, = body frame at identity attitude) that
    ``MetaShutDown7.shut_down_rotors(flag)`` sets with ``p.resetBaseVelocity``.

    Flags 0/1 have no preset spin in the reference environment; flags 2/3 start with the
    fault spin that makes the failure recoverable -- it is part of the scenario, not an
    optional extra.
    """
    if flag == 2:
        return np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), -rng.uniform(20, 25)])
    if flag == 3:
        return np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), -rng.uniform(24, 26)])
    return np.zeros(3)


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
    pa.add_argument("--kappa", type=float, default=0.0025,
                    help="body-rate damping torque coeff (torque = -kappa*omega)")
    pa.add_argument("--spin_init", type=float, default=0.0)
    pa.add_argument("--target_pos", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    pa.add_argument("--calib_steps", type=int, default=0)
    pa.add_argument("--tag", default="")
    pa.add_argument("--dump", type=int, default=0, help="print first N control steps")
    pa.add_argument("--sync", default="state", choices=["state", "wall"],
                    help="state: one control step per physics step (paced by /model_states, whose "
                         "update_rate must equal the physics rate); wall: wall-clock dt pacing")
    pa.add_argument("--seed", type=int, default=None,
                    help="RNG seed for the random fault initial spin (default: fresh entropy)")
    pa.add_argument("--substeps", type=int, default=0,
                    help="physics steps per control step: act on every Nth /model_states message; "
                         "0 (default) auto-detects it from the measured state rate so the control "
                         "loop stays at 200 Hz whatever the world's physics rate is")
    a = pa.parse_args()

    rclpy.init()
    n = Runner(a)
    ex = MultiThreadedExecutor(); ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    # Build the controller *before* spawning: loading a torch checkpoint takes ~1 s, and doing
    # it after the spawn lets the drone free-fall for that whole second before the loop starts.
    policy = load_policy(CKPT[a.flag]) if a.flag in CKPT else None
    pid = PositionPID(); lag = ActuatorLag(a.delay)
    imu = IMU(IMUConfig(gyro_range_dps=a.dps, lever_arm=LEVER_ARM,
                        gyro_noise_std=0.05, accel_noise_std=0.02))
    obs_est = LeverArmObserver(gyro_limit_dps=a.dps) if a.src == "override" else None
    n.setup()
    if a.substeps <= 0:
        a.substeps = max(1, int(round(n.state_hz / 200.0)))
    print(f"[gz] state rate {n.state_hz:.0f} Hz -> substeps={a.substeps} "
          f"(control at {n.state_hz / a.substeps:.0f} Hz)")
    mask = np.array(SHUT_DOWN[a.flag], dtype=float)
    target_pos = np.array(a.target_pos, dtype=float)
    last_action = np.zeros(4)
    prev_omega, calib = None, {k: [] for k in ("g", "a", "t")}

    log = {k: [] for k in ("z", "xy", "wt", "wu", "rate")}
    t_wall = time.time()
    for i in range(a.steps):
        if a.sync == "state":
            # sim-locked: act on every Nth physics step (N=1 -> one control step per physics step,
            # paced by the state publisher)
            t0 = time.time()
            for _ in range(a.substeps):
                seen = n.states_received
                while n.states_received == seen and time.time() - t0 < 2.0:
                    time.sleep(0.0001)
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
        if a.dump and (i < a.dump or i % 50 == 0):
            print(f"  i={i:4d} sim={n.sim_t:.4f} wall={time.time()-t_wall:6.3f} "
                  f"z={pos[2]:9.4f} om={np.round(omega, 3)} "
                  f"ta={target_a:6.3f} act={np.round(action, 3)} th={np.round(thrust, 3)} "
                  f"F={np.round(force, 3)} T={np.round(torque, 4)}")

        log["z"].append(float(pos[2])); log["xy"].append(float(np.hypot(pos[0], pos[1])))
        log["wt"].append(omega.copy()); log["wu"].append(np.asarray(rate).copy())
        sleep = a.dt - (time.time() - t_wall - i * a.dt)
        if a.sync == "wall" and sleep > 0:
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