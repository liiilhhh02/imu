#!/usr/bin/env python3
"""Controlled, headless wrench experiments against the running Gazebo world.

These tests deliberately do NOT use the policy: they apply a known wrench and compare the
resulting rigid-body acceleration against hand-computed theory, so the physics/plugin
conventions (frame, application point, sign, latch, control/sim synchronisation) can be
checked independently of the controller.

Tests (``--test`` selects one, ``all`` runs every one):

  freefall   no wrench -> z(t) = z0 - 1/2 g t^2                (gravity + integrator sanity)
  hover      body force [0,0,m g] -> vz == 0, z == z0          (force frame/magnitude)
  tilt       body force [0,0,m g] with roll=90deg -> vx growth (link-frame vs world-frame)
  rotor      single-rotor wrench from ``mixer_body_wrench`` ->
             alpha = tau / I                                    (mixer <-> Gazebo torque)
  torquex    pure body-x torque -> wz-free alpha_x = tau_x / Ixx
  torquey    pure body-y torque -> alpha_y = tau_y / Iyy
  torquez    pure body-z torque -> alpha_z = tau_z / Izz (yaw)
  latch      ONE wrench message -> is it applied over many physics steps?
  service    same hover via /apply_link_wrench (reference_frame=LINK) vs the topic

All forces/torques are expressed in the body (link) frame, exactly like ``gz_runner.py``.

Usage (always one wrapped shell, see README):
  bash -c 'source /opt/ros/humble/setup.bash; export PYTHONPATH=...; $PY gazebo/gz_wrench_test.py'
"""
import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from gazebo_msgs.srv import (ApplyLinkWrench, DeleteEntity, GetEntityState, LinkRequest,
                             SetEntityState, SpawnEntity)
from gazebo_msgs.msg import EntityState, ModelStates
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, Wrench
from rosgraph_msgs.msg import Clock

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "cf2x_gazebo.urdf")
MODEL = "cf2"
LINK = "cf2::base_link"
PROP_XY = np.array([[0.1273, 0.1273], [-0.1273, 0.1273],
                    [-0.1273, -0.1273], [0.1273, -0.1273]])
MASS = 1.0
J = np.array([0.004, 0.004, 0.007])
G = 9.81
KM = 0.01
CTRL_DT = 0.005


def quat_to_matrix(q):
    x, y, z, w = np.asarray(q, float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mixer_body_wrench(thrust):
    """Local copy of gpd_me.policy.mixer_body_wrench (read-only reference; kept in sync)."""
    t = np.asarray(thrust, float)
    return (np.array([0.0, 0.0, float(t.sum())]),
            np.array([float(np.sum(PROP_XY[:, 1] * t)),
                      float(-np.sum(PROP_XY[:, 0] * t)),
                      float(KM * (t[0] - t[1] + t[2] - t[3]))]))


class H(Node):
    def __init__(self):
        super().__init__("gz_wrench_test")
        self.sim_t = None
        self.steps = 0
        self.mstate = None
        self.create_subscription(ModelStates, "/model_states", self._on_states, 10)
        self.create_subscription(
            Clock, "/clock", self._on_clock,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.del_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.set_cli = self.create_client(SetEntityState, "/set_entity_state")
        self.get_cli = self.create_client(GetEntityState, "/get_entity_state")
        self.wr_cli = self.create_client(ApplyLinkWrench, "/apply_link_wrench")
        self.clr_cli = self.create_client(LinkRequest, "/clear_link_wrenches")
        self.pub = self.create_publisher(Wrench, "/cf2/cmd_wrench", 1)

    def _on_states(self, m):
        if MODEL in m.name:
            i = m.name.index(MODEL)
            p, t = m.pose[i], m.twist[i]
            self.mstate = (
                np.array([p.position.x, p.position.y, p.position.z]),
                np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
                np.array([t.linear.x, t.linear.y, t.linear.z]),
                np.array([t.angular.x, t.angular.y, t.angular.z]))

    def _on_clock(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9
        self.steps += 1

    def call(self, cli, req, timeout=5.0):
        f = cli.call_async(req)
        t0 = time.time()
        while not f.done() and time.time() - t0 < timeout:
            time.sleep(0.0002)
        return f.result() if f.done() else None

    def spawn(self):
        for c in (self.spawn_cli, self.set_cli, self.get_cli):
            c.wait_for_service(timeout_sec=10.0)
        d = DeleteEntity.Request(); d.name = MODEL
        self.call(self.del_cli, d)
        time.sleep(0.4)
        r = SpawnEntity.Request(); r.name = MODEL; r.xml = open(URDF).read()
        r.initial_pose = Pose(); r.initial_pose.position.z = 1.0
        out = self.call(self.spawn_cli, r)
        print(f"[gz] spawn success={out.success if out else None}")
        t0 = time.time()
        while self.sim_t is None and time.time() - t0 < 8:
            time.sleep(0.05)

    def zero_wrench(self):
        self.pub.publish(Wrench())

    def reset(self, z=1.0, quat=(0.0, 0.0, 0.0, 1.0), ang=(0.0, 0.0, 0.0)):
        """Teleport to a clean state with zero linear velocity."""
        self.zero_wrench()
        st = EntityState(); st.name = MODEL
        st.pose = Pose()
        st.pose.position = Point(x=0.0, y=0.0, z=float(z))
        st.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        st.twist = Twist()
        st.twist.angular = Vector3(x=ang[0], y=ang[1], z=ang[2])
        st.reference_frame = "world"
        out = self.call(self.set_cli, SetEntityState.Request(state=st))
        # let the teleport settle and the sim advance a couple of steps
        self.wait_sim(0.02)
        return out.success if out else None

    def get(self):
        """Returns (pos, quat, lin, ang, sim_stamp) in the world frame via /get_entity_state."""
        req = GetEntityState.Request(); req.name = LINK; req.reference_frame = "world"
        out = self.call(self.get_cli, req)
        if out is None or not out.success:
            raise RuntimeError("get_entity_state failed")
        s = out.state
        stamp = out.header.stamp.sec + out.header.stamp.nanosec * 1e-9
        return (np.array([s.pose.position.x, s.pose.position.y, s.pose.position.z]),
                np.array([s.pose.orientation.x, s.pose.orientation.y, s.pose.orientation.z,
                          s.pose.orientation.w]),
                np.array([s.twist.linear.x, s.twist.linear.y, s.twist.linear.z]),
                np.array([s.twist.angular.x, s.twist.angular.y, s.twist.angular.z]),
                stamp)

    def wait_sim(self, dur, wall_timeout=30.0):
        t0 = self.sim_t
        if t0 is None:
            return
        w0 = time.time()
        while self.sim_t - t0 < dur and time.time() - w0 < wall_timeout:
            time.sleep(0.0005)

    def apply_topic(self, force, torque, dur, sync="clock"):
        """Publish a constant body-frame wrench for ``dur`` seconds of *sim* time.

        sync='clock': publish one message per physics step (clock-gated) -- the correct,
                      sim-locked policy.
        sync='wall' : publish at 200 Hz of *wall* time (what gz_runner.py does today).
        Returns (n_published, n_physics_steps, actual_elapsed_sim_time).
        """
        w = Wrench()
        w.force = Vector3(x=float(force[0]), y=float(force[1]), z=float(force[2]))
        w.torque = Vector3(x=float(torque[0]), y=float(torque[1]), z=float(torque[2]))
        t0 = self.sim_t
        steps0 = self.steps
        n = 0
        w0 = time.time()
        if sync == "clock":
            next_t = t0
            while self.sim_t - t0 < dur and time.time() - w0 < 30.0:
                if self.sim_t >= next_t:
                    self.pub.publish(w); n += 1; next_t += CTRL_DT
                time.sleep(0.0002)
        else:
            while time.time() - w0 < dur:
                self.pub.publish(w); n += 1
                time.sleep(CTRL_DT)
        t_stop = self.sim_t
        self.zero_wrench()
        return n, self.steps - steps0, t_stop - t0

    def apply_service(self, force, torque, dur):
        req = ApplyLinkWrench.Request()
        req.link_name = LINK
        req.reference_frame = LINK
        req.wrench = Wrench()
        req.wrench.force = Vector3(x=float(force[0]), y=float(force[1]), z=float(force[2]))
        req.wrench.torque = Vector3(x=float(torque[0]), y=float(torque[1]), z=float(torque[2]))
        req.duration.sec = -1
        out = self.call(self.wr_cli, req)
        ok = out.success if out else None
        self.wait_sim(dur)
        clr = LinkRequest.Request(); clr.link_name = LINK
        self.call(self.clr_cli, clr)
        return ok


def body_rate(quat, ang_world):
    return quat_to_matrix(quat).T @ np.asarray(ang_world, float)


def test_freefall(n, args):
    n.reset(z=1.0)
    p0 = n.get()
    n.wait_sim(args.dur)
    p1 = n.get()
    dt = p1[4] - p0[4]
    z, vz = p1[0][2], p1[2][2]
    z_th = p0[0][2] - 0.5 * G * dt ** 2
    print(f"TEST freefall   dt={dt:.4f}s: z={z:9.4f} (theory {z_th:9.4f})  "
          f"vz={vz:9.4f} (theory {-G*dt:9.4f})  |w|={np.linalg.norm(p1[3]):.4f}")


def test_hover(n, args, service=False, sync="clock"):
    n.reset(z=1.0)
    p0 = n.get()
    if service:
        ok = n.apply_service([0.0, 0.0, MASS * G], [0.0, 0.0, 0.0], args.dur)
        p1 = n.get()
        dt = p1[4] - p0[4]
    else:
        _, _, dt = n.apply_topic([0.0, 0.0, MASS * G], [0.0, 0.0, 0.0], args.dur, sync=sync)
        p1 = n.get()
    dz, vz = p1[0][2] - p0[0][2], p1[2][2]
    tag = "service" if service else f"topic/{sync}"
    print(f"TEST hover[{tag:11s}] dt={dt:.4f}s: dz={dz:+9.4f} (theory +0.0000)  "
          f"vz={vz:+9.4f} (theory +0.0000)  |w|={np.linalg.norm(p1[3]):.4f}")


def test_tilt(n, args):
    """Roll the body 90 deg and push along its own z: link-frame -> sideways motion."""
    th = math.pi / 2
    q = (math.sin(th / 2), 0.0, 0.0, math.cos(th / 2))
    n.reset(z=1.0, quat=q)
    p0 = n.get()
    _, _, dt = n.apply_topic([0.0, 0.0, MASS * G], [0.0, 0.0, 0.0], args.dur)
    p1 = n.get()
    R = quat_to_matrix(q)
    v_body = R.T @ p1[2]
    print(f"TEST tilt(roll90) dt={dt:.4f}s: dv_world={np.round(p1[2]-p0[2], 4)} "
          f"(theory link-frame {np.round(R @ np.array([0.0, 0.0, G*dt]), 4)}) "
          f"v_body={np.round(v_body, 4)}")


def test_rotor(n, args):
    """Single-rotor wrench (mixer mapping) -> measured alpha vs tau/I."""
    t = args.force_n
    dur = min(args.dur, 0.05)
    force, torque = mixer_body_wrench([t, 0.0, 0.0, 0.0])
    alpha_th = torque / J
    n.reset(z=1.0)
    n.wait_sim(0.2)                     # settle: no wrench yet
    p0 = n.get()
    w0 = body_rate(p0[1], p0[3])
    _, _, dt = n.apply_topic(force, torque, dur)
    p1 = n.get()
    w1 = body_rate(p1[1], p1[3])
    alpha = (w1 - w0) / dt
    print(f"TEST rotor0 t={t:.2f}N dt={dt:.4f}s: force={np.round(force, 4)} "
          f"torque={np.round(torque, 4)}")
    print(f"     alpha_meas={np.round(alpha, 3)} rad/s^2   alpha_theory(tau/I)="
          f"{np.round(alpha_th, 3)}   ratio={np.round(alpha / alpha_th, 3)}")


def test_torque(n, args, axis):
    tau = np.zeros(3); tau[axis] = args.force_n
    n.reset(z=1.0)
    n.wait_sim(0.2)
    p0 = n.get()
    w0 = body_rate(p0[1], p0[3])
    _, _, dt = n.apply_topic([0.0, 0.0, 0.0], tau, args.dur)
    p1 = n.get()
    w1 = body_rate(p1[1], p1[3])
    alpha = (w1 - w0) / dt
    th = tau / J
    print(f"TEST torque{['x','y','z'][axis]} tau={tau[axis]:.4f} N.m dt={dt:.4f}s: "
          f"alpha_meas={np.round(alpha, 4)}  alpha_theory={np.round(th, 4)}  "
          f"ratio={np.round(alpha / th, 4)}")


def test_latch(n, args):
    """Publish ONE wrench; a latched plugin keeps applying it every physics step."""
    n.reset(z=1.0)
    n.zero_wrench()
    w = Wrench(); w.force = Vector3(x=0.0, y=0.0, z=MASS * G)
    n.pub.publish(w)
    time.sleep(0.05)                    # let the message arrive
    p0 = n.get()
    n.wait_sim(1.0)
    p1 = n.get()
    dt = p1[4] - p0[4]
    print(f"TEST latch(1 msg, dt={dt:.3f}s): dz={p1[0][2]-p0[0][2]:+9.4f} "
          f"(hover theory +0.0000)  vz={p1[2][2]:+9.4f}  "
          f"-> latched={'YES' if abs(p1[2][2]) < 0.5 else 'NO'}")
    n.zero_wrench()


def test_trace(n, args):
    """Time series of a constant hover wrench: reveals force duty-cycle / sync issues."""
    n.reset(z=1.0)
    n.wait_sim(0.05)
    force = [0.0, 0.0, MASS * G]
    w = Wrench(); w.force = Vector3(x=0.0, y=0.0, z=float(force[2]))
    t0 = n.sim_t
    next_t = t0
    next_sample = t0
    npub = 0
    rows = []
    print(f"TEST trace force={force} dur={args.dur}s")
    print(f"     {'sim_t':>9} {'stamp':>9} {'n_pub':>6} {'z':>10} {'vz':>10} {'|w|':>8}")
    while n.sim_t - t0 < args.dur:
        if n.sim_t >= next_t:
            n.pub.publish(w); npub += 1; next_t += CTRL_DT
        if n.sim_t >= next_sample:
            s = n.get()
            rows.append((n.sim_t, s[4], npub, s[0][2], s[2][2], np.linalg.norm(s[3])))
            next_sample += 0.05
        time.sleep(0.0005)
    n.zero_wrench()
    for r in rows[::max(1, len(rows) // 15)]:
        print(f"     {r[0]:9.3f} {r[1]:9.3f} {r[2]:6d} {r[3]:10.4f} {r[4]:10.4f} {r[5]:8.4f}")
    if rows:
        print(f"     last: z={rows[-1][3]:.4f} vz={rows[-1][4]:.4f} rows={len(rows)} "
              f"sim_dt={rows[-1][0]-rows[0][0]:.4f} stamp_dt={rows[-1][1]-rows[0][1]:.4f}")


def test_statecmp(n, args):
    """Compare /model_states (what gz_runner reads) with /get_entity_state (world frame)."""
    tau = np.array([0.05, -0.03, 0.02])
    n.reset(z=1.0)
    n.wait_sim(0.2)
    s_geo = n.get()
    s_mod = n.mstate
    print("TEST statecmp (at rest)")
    print(f"  get_entity_state: pos={np.round(s_geo[0], 5)} quat={np.round(s_geo[1], 5)} "
          f"lin={np.round(s_geo[2], 5)} ang={np.round(s_geo[3], 5)}")
    print(f"  /model_states   : pos={np.round(s_mod[0], 5)} quat={np.round(s_mod[1], 5)} "
          f"lin={np.round(s_mod[2], 5)} ang={np.round(s_mod[3], 5)}")
    _, _, dt = n.apply_topic([0.0, 0.0, 0.0], tau, 0.4)
    n.wait_sim(0.05)      # settle so both sources see the same (constant) omega
    s_geo = n.get()
    s_mod = n.mstate
    print(f"TEST statecmp (after tau={tau} for {dt:.3f}s, omega constant)")
    print(f"  get_entity_state: ang_world={np.round(s_geo[3], 5)} "
          f"ang_body={np.round(body_rate(s_geo[1], s_geo[3]), 5)}")
    print(f"  /model_states   : ang_world={np.round(s_mod[3], 5)} "
          f"ang_body={np.round(body_rate(s_mod[1], s_mod[3]), 5)}")
    print(f"  quat diff={np.round(s_geo[1]-s_mod[1], 6)}  pos diff="
          f"{np.round(s_geo[0]-s_mod[0], 6)}  lin diff={np.round(s_geo[2]-s_mod[2], 5)}")


TESTS = {
    "freefall": test_freefall,
    "hover": test_hover,
    "trace": test_trace,
    "statecmp": test_statecmp,
    "tilt": test_tilt,
    "rotor": test_rotor,
    "torquex": lambda n, a: test_torque(n, a, 0),
    "torquey": lambda n, a: test_torque(n, a, 1),
    "torquez": lambda n, a: test_torque(n, a, 2),
    "latch": test_latch,
    "service": lambda n, a: test_hover(n, a, service=True),
}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--test", default="all",
                    choices=["all"] + sorted(TESTS))
    pa.add_argument("--dur", type=float, default=1.0)
    pa.add_argument("--force-n", type=float, default=2.0,
                    help="rotor thrust [N] or pure torque [N.m] for the torque tests")
    pa.add_argument("--sync", default="clock", choices=["clock", "wall"])
    args = pa.parse_args()

    rclpy.init()
    n = H()
    ex = MultiThreadedExecutor(); ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    n.spawn()

    if args.test == "all":
        order = ["freefall", "hover", "tilt", "latch", "rotor",
                 "torquex", "torquey", "torquez", "service"]
    else:
        order = [args.test]
    print(f"# sim clock t0={n.sim_t}  sync={args.sync}  dur={args.dur}")
    for name in order:
        TESTS[name](n, args)
        n.zero_wrench()
        n.wait_sim(0.1)
    ex.shutdown(); rclpy.shutdown()


if __name__ == "__main__":
    main()