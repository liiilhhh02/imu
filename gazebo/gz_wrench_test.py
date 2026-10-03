#!/usr/bin/env python3
"""Decisive test of the Gazebo wrench semantics: ONE ApplyLinkWrench call, then watch z.

Variants (selected by -p variant:=infinite|finite|finite_start):
  infinite      duration = -1  (apply until cleared)          -> one call must produce lift
  finite        duration = 0.5, start_time = 0 (unset)
  finite_start  duration = 0.5, start_time = current sim time (from /clock)

A single call isolates the "wrench is replaced/accumulated/expired" question from the
per-step re-application in the controller.
"""
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import ApplyLinkWrench, DeleteEntity, LinkRequest, SpawnEntity
from geometry_msgs.msg import Pose, Wrench
from rosgraph_msgs.msg import Clock

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "cf2x_gazebo.urdf")
MODEL = "cf2"
LINK = "cf2::base_link"


class T(Node):
    def __init__(self):
        super().__init__("gz_wrench_test")
        self.declare_parameter("variant", "infinite")
        self.declare_parameter("force_n", 50.0)
        self.state = None
        self.sim_t = None
        self.create_subscription(ModelStates, "/model_states", self._on_states, 10)
        self.create_subscription(Clock, "/clock", self._on_clock,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.del_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.wr_cli = self.create_client(ApplyLinkWrench, "/apply_link_wrench")
        self.clr_cli = self.create_client(LinkRequest, "/clear_link_wrenches")

    def _on_states(self, m):
        if MODEL in m.name:
            i = m.name.index(MODEL)
            self.state = (np.array([m.pose[i].position.x, m.pose[i].position.y, m.pose[i].position.z]),
                          np.array([m.twist[i].linear.x, m.twist[i].linear.y, m.twist[i].linear.z]))

    def _on_clock(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def wait(self, fut, t=3.0):
        t0 = time.time()
        while not fut.done() and time.time() - t0 < t:
            time.sleep(0.0005)

    def call(self, cli, req):
        f = cli.call_async(req)
        self.wait(f)
        return f.result()

    def spawn(self):
        for c in (self.spawn_cli, self.wr_cli):
            c.wait_for_service(timeout_sec=10.0)
        d = DeleteEntity.Request(); d.name = MODEL
        self.call(self.del_cli, d)
        time.sleep(0.5)
        r = SpawnEntity.Request(); r.name = MODEL; r.xml = open(URDF).read()
        r.initial_pose = Pose(); r.initial_pose.position.z = 1.0
        out = self.call(self.spawn_cli, r)
        self.get_logger().info(f"spawn={out.success} {out.status_message}")


def main():
    rclpy.init()
    n = T()
    ex = MultiThreadedExecutor(); ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    n.spawn()

    t0 = time.time()
    while n.state is None and time.time() - t0 < 10:
        time.sleep(0.05)
    variant = n.get_parameter("variant").value
    force = float(n.get_parameter("force_n").value)
    print("variant=%s force=%s N sim_t=%s z0=%.4f" % (variant, force, n.sim_t, n.state[0][2]))

    req = ApplyLinkWrench.Request()
    req.link_name = LINK
    req.reference_frame = LINK
    w = Wrench(); w.force.z = force
    req.wrench = w
    if variant == "infinite":
        req.duration.sec = -1
    else:
        req.duration.sec = 0
        req.duration.nanosec = 500_000_000
        if variant == "finite_start":
            req.start_time.sec = int(n.sim_t or 0)
            req.start_time.nanosec = int(((n.sim_t or 0) % 1) * 1e9)
    out = n.call(n.wr_cli, req)
    print(f"apply_link_wrench success={out.success} ({out.status_message})")

    for k in range(8):
        time.sleep(0.1)
        z = n.state[0][2] if n.state else float("nan")
        vz = n.state[1][2] if n.state else float("nan")
        print(f"  t+{(k+1)*0.1:.1f}s  z={z:8.4f}  vz={vz:8.4f}  (free fall would be vz={-9.81*(k+1)*0.1:7.3f})")
    clr = LinkRequest.Request(); clr.link_name = LINK
    n.call(n.clr_cli, clr)
    ex.shutdown(); rclpy.shutdown()


if __name__ == "__main__":
    main()