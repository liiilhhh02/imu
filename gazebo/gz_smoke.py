#!/usr/bin/env python3
"""Toolchain smoke test: spawn the drone in Gazebo and hold it with a constant equivalent wrench.

Validates, before any controller work:
  * `/spawn_entity` accepts the senior's URDF,
  * `/model_states` gives world pose/twist,
  * `/apply_link_wrench` with reference_frame == the body link really produces body-frame forces,
  * the wrench is *replaced* (not accumulated) when re-applied every step.

Expected result: z stays close to 1.0 m with F_z = M g = 9.8 N.
"""
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import ApplyLinkWrench, DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose, Wrench

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "cf2x_gazebo.urdf")
MODEL = "cf2"
LINK = "cf2::base_link"


class Smoke(Node):
    def __init__(self):
        super().__init__("gz_smoke")
        self.declare_parameter("thrust_n", 9.8)
        self.declare_parameter("duration_s", 2.0)
        self.declare_parameter("duration_ns", 0.01)   # wrench hold time
        self.declare_parameter("ref", "body")         # body | world
        self.declare_parameter("prefix", True)        # cf2::base_link vs base_link
        self.state = None
        self.create_subscription(ModelStates, "/model_states", self._on_states, 10)
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.wrench_cli = self.create_client(ApplyLinkWrench, "/apply_link_wrench")

    def _wait_service(self, cli, name, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cli.service_is_ready():
                return True
            self.get_logger().info(f"waiting for {name} ...")
            time.sleep(1.0)
        return cli.service_is_ready()

    def _on_states(self, msg):
        if MODEL in msg.name:
            i = msg.name.index(MODEL)
            p, t = msg.pose[i], msg.twist[i]
            self.state = (np.array([p.position.x, p.position.y, p.position.z]),
                          np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
                          np.array([t.linear.x, t.linear.y, t.linear.z]),
                          np.array([t.angular.x, t.angular.y, t.angular.z]))

    def spawn(self):
        if not self._wait_service(self.spawn_cli, "/spawn_entity", 20.0):
            raise RuntimeError("/spawn_entity not available")
        if self.delete_cli.wait_for_service(timeout_sec=3.0):
            d = DeleteEntity.Request()
            d.name = MODEL
            self._wait(self.delete_cli.call_async(d))
            time.sleep(0.3)
        req = SpawnEntity.Request()
        req.name = MODEL
        req.xml = open(URDF).read()
        req.initial_pose = Pose()
        req.initial_pose.position.z = 1.0
        fut = self.spawn_cli.call_async(req)
        self._wait(fut)
        ok = fut.result().success
        self.get_logger().info(f"spawn success={ok} ({fut.result().status_message})")
        return ok

    def apply(self, force, torque, duration=None):
        req = ApplyLinkWrench.Request()
        req.link_name = LINK if bool(self.get_parameter("prefix").value) else "base_link"
        req.reference_frame = (LINK if bool(self.get_parameter("prefix").value) else "base_link") \
            if self.get_parameter("ref").value == "body" else ""
        if duration is None:
            duration = float(self.get_parameter("duration_ns").value)
        w = Wrench()
        w.force.x, w.force.y, w.force.z = [float(v) for v in force]
        w.torque.x, w.torque.y, w.torque.z = [float(v) for v in torque]
        req.wrench = w
        if duration < 0:
            req.duration.sec = -1
            req.duration.nanosec = 0
        else:
            req.duration.sec = int(duration)
            req.duration.nanosec = int(round((duration - int(duration)) * 1e9))
        fut = self.wrench_cli.call_async(req)
        self._wait(fut)
        return fut.result().success

    def _wait(self, fut, timeout=2.0):
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.0005)
        return fut.done()


def main():
    rclpy.init()
    node = Smoke()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    assert node.spawn(), "spawn failed"
    t0 = time.time()
    while node.state is None and time.time() - t0 < 5.0:
        time.sleep(0.05)
    assert node.state is not None, "no /model_states for the spawned model"
    print(f"spawned {MODEL}; initial z = {node.state[0][2]:.3f} m")

    if not node._wait_service(node.wrench_cli, "/apply_link_wrench", 20.0):
        raise RuntimeError("/apply_link_wrench not available")

    ft = float(node.get_parameter("thrust_n").value)
    dur = float(node.get_parameter("duration_s").value)
    print(f"params: thrust={ft} N  ref={node.get_parameter('ref').value}  "
          f"prefix={node.get_parameter('prefix').value}  hold={node.get_parameter('duration_ns').value} s")
    dt = 0.005
    n = int(dur / dt)
    zs, ok = [], 0
    for i in range(n):
        if node.apply([0.0, 0.0, ft], [0.0, 0.0, 0.0]):
            ok += 1
        time.sleep(dt)
        if node.state is not None and i % 40 == 0:
            zs.append((i * dt, node.state[0][2], node.state[2][2]))
            print(f"  t={i*dt:4.2f}s z={node.state[0][2]:7.4f} m  vz={node.state[2][2]:7.4f} m/s")
    print(f"wrench calls ok: {ok}/{n}")
    print(f"z drift: {zs[0][1]:.4f} -> {zs[-1][1]:.4f} m")
    ex.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()