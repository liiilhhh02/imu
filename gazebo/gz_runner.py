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
import torch

# A batch-of-one GRU at 200 Hz is *slower* with the default thread pool (measured 2.79 ms with 24
# threads): the sync overhead dominates.  One thread brings the whole estimator step to ~1.5-2 ms, which
# is what lets the closed loop run in a real-time world instead of a slowed one -- and the slowed world
# turned out to make the closed-loop outcome pacing-dependent.  The same argument the supervisor's own
# trainer makes for its tiny nets.
torch.set_num_threads(1)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation

from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose, Twist, Wrench
from rosgraph_msgs.msg import Clock

REPO = "/home/liiil/Downloads/gym-pybullet-drones"
ME = "/home/liiil/Downloads/me"
sys.path[:0] = [REPO, ME]

from gpd_me.e2e import WINDOW, E2ENet, algebraic_estimate, coarse_summary, fine_features  # noqa: E402
from gpd_me.imu import IMU, IMUConfig              # noqa: E402
from gpd_me.ins import AttitudeINS                 # noqa: E402
from gpd_me.tilt import TiltObserver               # noqa: E402
from gpd_me.observer import LeverArmObserver       # noqa: E402
from gpd_me.policy import (ActuatorLag, PositionPID, load_policy, mixer_body_wrench,  # noqa: E402
                           quat_to_matrix)
from gpd_me.priors import identify_priors, tom_from_command  # noqa: E402

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

    def apply_ic(self):
        """(Re-)pose the drone at the fault IC: [0,0,1], identity attitude, zero linear velocity
        and the preset spin of `MetaShutDown7.shut_down_rotors(flag)`."""
        self.apply(np.zeros(3), np.zeros(3))
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

    def step_paced(self):
        """Wait one control step (``--substeps`` physics steps) and return the fresh state."""
        t0 = time.time()
        for _ in range(self.a.substeps):
            seen = self.states_received
            while self.states_received == seen and time.time() - t0 < 2.0:
                time.sleep(0.0001)
        return self.state

    def send_u(self, u_cmd, mask, lag):
        """Open-loop: per-rotor thrust command [N] -> thrust lag -> mixer -> /cf2/cmd_wrench."""
        forces = np.clip(np.asarray(u_cmd, float), 0.0, 15.0) * np.asarray(mask, float)
        thrust = lag.step(forces, self.a.dt)
        omega = quat_to_matrix(self.state[1]).T @ self.state[3]
        force, torque = mixer_body_wrench(thrust, self.a.km,
                                          damping_torque=-self.a.kappa * omega)
        self.apply(force, torque)
        return thrust

    def identify(self, imu, lag):
        """The deployment procedure's first half, as `scripts/e4_closed_loop.identification`:
        nominal hover + the two-stage open-loop yaw manoeuvre, then the fault and its open-loop
        transient -> `identify_priors`.  Every logged quantity is measurable on the real aircraft
        (the true thrust is a diagnostic only)."""
        dt, mass = self.a.dt, self.a.mass
        seed = self.a.seed if self.a.seed is not None else int.from_bytes(os.urandom(4), "little")
        L = {k: [] for k in ("g", "a", "u", "w", "v", "t", "m")}
        prev = [None]

        def log_step(mask, om):
            tom = float(lag.thrust.sum()) / mass
            wdot = np.zeros(3) if prev[0] is None else (om - prev[0]) / dt
            prev[0] = om.copy()
            gyro, accel = imu.measure(om, wdot, tom, dt, force=True)
            L["g"].append(gyro); L["a"].append(accel); L["v"].append(self.state[2].copy())
            L["w"].append(om.copy()); L["t"].append(tom)
            L["m"].append(np.asarray(mask, float).copy())

        rng0 = np.random.default_rng(seed)
        yid_f = float(rng0.uniform(0.3, 2.0)); yid_target = 0.30 * np.deg2rad(self.a.dps)
        yid_d0 = float(rng0.uniform(0.3, 1.0)); yid_tA = int(0.2 / dt)
        hover_u = mass * 9.81 / 4.0
        yid_amp, yid_w0 = None, None
        ones = np.ones(4)
        for i in range(int(1.6 / dt)):
            self.step_paced()
            om = quat_to_matrix(self.state[1]).T @ self.state[3]
            log_step(ones, om)
            base = hover_u + 2.0 * (1.0 - float(self.state[0][2]))
            if i == yid_tA:
                yid_w0 = float(om[2])
            if i < yid_tA:
                exc = yid_d0
            else:
                if yid_amp is None:
                    dtA = max(i - yid_tA, 1) * dt
                    gain = abs(float(om[2]) - yid_w0) / max(abs(yid_d0) * dtA, 1e-6)
                    yid_amp = float(np.clip(yid_target / max(gain, 1e-3), 0.05, 3.0))
                exc = yid_amp * np.sin(2 * np.pi * yid_f * (i - yid_tA) * dt)
            u = np.clip(np.array([base - exc, base + exc, base - exc, base + exc]), 0.0, 15.0)
            L["u"].append(u)
            self.send_u(u, ones, lag)
        self.apply_ic()                       # the fault draw
        mask = np.array(SHUT_DOWN[self.a.flag], float)
        rng = np.random.default_rng(seed * 7919 + 13)
        base = mass * 9.81 / max(float(mask.sum()), 1.0)
        walk = rng.normal(0.0, 0.05, 4); freqs = rng.uniform(0.2, 8.0, 4)
        phases = rng.uniform(0, 2 * np.pi, 4); amps = rng.uniform(0.0, 3.0, 4)
        for i in range(int(float(self.a.id_post) / dt)):
            self.step_paced()
            om = quat_to_matrix(self.state[1]).T @ self.state[3]
            log_step(mask, om)
            walk = np.clip(walk * 0.995 + rng.normal(0.0, 0.05, 4), -1.5, 1.5)
            u = np.clip(base + walk + amps * np.sin(2 * np.pi * freqs * i * dt + phases), 0.0, 15.0)
            L["u"].append(u)
            self.send_u(u, mask, lag)
        d = {k: np.array(v) for k, v in L.items()}
        self.apply(np.zeros(3), np.zeros(3))
        prior, diag = identify_priors(d["g"], d["a"], d["u"], d["w"], d["m"], self.a.dps, dt,
                                      vel=d["v"], tom=d["t"])
        print(f"[gz] prior={np.round(np.asarray(prior, float), 5)}")
        return prior, diag

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


_WARNED = set()          # checkpoints already reported as incompatible (loud once, not per run)


class NetRate:
    """Causal wrapper around the trained end-to-end estimator (same as e4_closed_loop.NetRate:
    rolling 48-frame window, algebraic front end, actuator model from the identified priors; the
    simulator's true thrust is never used online)."""

    def __init__(self, ckpt, dev, prior, dps, dt, mode="net"):
        self.dev = dev
        self.mode = mode          # "net" = trained residual on top of w_alg; "alg" = w_alg only
        self.dt = dt
        self.prior = np.asarray(prior, float)
        self.g_T = float(self.prior[3]); self.tau = float(self.prior[7]); self.k = float(self.prior[8])
        self.lim = np.deg2rad(dps)
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        ck_hidden, ck_layers = int(ck.get("hidden", 128)), int(ck.get("layers", 1))
        self.net = E2ENet(hidden=ck_hidden, layers=ck_layers).to(dev).eval()
        try:
            self.net.load_state_dict(ck["state"])
            self.compat = True
            self.xf_m = torch.tensor(ck["xf_m"], device=dev, dtype=torch.float32)
            self.xf_s = torch.tensor(ck["xf_s"], device=dev, dtype=torch.float32)
            self.xp_m = torch.tensor(ck["xp_m"], device=dev, dtype=torch.float32)
            self.xp_s = torch.tensor(ck["xp_s"], device=dev, dtype=torch.float32)
        except RuntimeError as exc:
            self.compat = False
            msg = next((ln.strip() for ln in str(exc).splitlines() if "size mismatch" in ln),
                       str(exc).splitlines()[0].strip())
            if ckpt not in _WARNED:
                _WARNED.add(ckpt)
                print(f"[gz] WARNING: {ckpt} cannot be loaded ({msg}); the 'net' row falls back "
                      f"to the algebraic estimate.")
            F = fine_features(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), bool),
                              np.zeros((1, 4)), np.zeros(1), self.prior, np.zeros((1, 3))).shape[1]
            self.xf_m = torch.zeros(F, device=dev); self.xf_s = torch.ones(F, device=dev)
            self.xp_m = torch.zeros(len(self.prior), device=dev)
            self.xp_s = torch.ones(len(self.prior), device=dev)
        self.buf = {k: [] for k in ("a", "g", "u", "m")}
        # `netr` = None when the checkpoint cannot be loaded: the residual head is zero-initialised so a
        # fresh net would *numerically* return w_alg, but running it anyway is a trap if that assumption
        # ever changes -- make the bypass explicit instead of relying on an initialisation detail.
        self.bypass = not self.compat
        # The 2 s coarse summary costs a 400-sample actuator-model recursion plus a 400-frame stack; on
        # its own it dominated the per-step cost (measured 18-32 ms per control step, which forced the
        # slowed world and made the closed loop pacing-dependent).  It is a *causal 2 s* aggregate, so
        # recomputing it every `CS_EVERY` steps (250 ms at 200 Hz) changes it negligibly while keeping
        # the loop inside a 5 ms budget, i.e. able to run in a real-time world.
        self._cs = None
        self._cs_age = 10 ** 9

    def step(self, accel, gyro, u_cmd, mask):
        for key, val in (("a", accel), ("g", gyro), ("u", u_cmd), ("m", mask)):
            self.buf[key].append(np.asarray(val, float))
        H = min(WINDOW, len(self.buf["a"]))
        pad = WINDOW - H
        rep = lambda x: np.concatenate([np.repeat(x[:1], pad, 0), x], 0) if pad else x
        a_ = rep(np.stack(self.buf["a"][-H:])); g_ = rep(np.stack(self.buf["g"][-H:]))
        u_ = rep(np.stack(self.buf["u"][-H:])); m_ = rep(np.stack(self.buf["m"][-H:]))
        sat_ = np.abs(g_) >= self.lim - 1e-9
        tom_ = tom_from_command(u_, m_, self.dt, self.g_T, self.tau)
        w_alg = algebraic_estimate(g_, a_ - np.array([0.0, 0.0, 1.0]) * tom_[:, None],
                                   self.k, sat_, self.lim)
        if self.mode == "alg" or self.bypass:
            # the analytic front end alone (either asked for, or a checkpoint that cannot be loaded --
            # keeps the row honest instead of silently running an untrained network)
            # the analytic front end alone: separates "estimator pipeline" (priors, tom_from_command,
            # INS-driven attitude) from "network" when a closed-loop divergence is being attributed
            return w_alg[-1].astype(float)
        Na = len(self.buf["a"]); Hc = min(400, Na)
        cs = self._cs
        if self._cs is None or self._cs_age >= 10:
            repc = lambda x: np.concatenate([np.repeat(x[:1], 400 - Hc, 0), x], 0) if 400 - Hc else x
            cg = repc(np.stack(self.buf["g"][-Hc:])); cm = repc(np.stack(self.buf["m"][-Hc:]))
            cu = repc(np.stack(self.buf["u"][-Hc:]))
            ctom_ = tom_from_command(cu, cm, self.dt, self.g_T, self.tau)
            cs = coarse_summary(cg, np.abs(cg) >= self.lim - 1e-9, cu, ctom_, self.prior)
            self._cs = cs
            self._cs_age = 0
        self._cs_age += 1
        T_ = lambda x: torch.tensor(np.asarray(x)[None], dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            ft = ((T_(fine_features(a_, g_, sat_, u_, tom_, self.prior, w_alg)) - self.xf_m)
                  / self.xf_s).clamp(-50, 50)
            pt = (T_(self.prior) - self.xp_m) / self.xp_s
            w_hat, _ = self.net(ft, T_(cs), pt, T_(w_alg), torch.tensor(sat_[None], device=self.dev))
        return w_hat[0, -1].cpu().numpy().astype(float)


def build_obs(rel_xy, rate, target_a, thrust_over_mass, last_action, mask):
    return np.array([rel_xy[0], rel_xy[1],
                     rate[0] / 10.0, rate[1] / 10.0, rate[2] / 50.0,
                     (target_a - 9.8) / 3.0, (thrust_over_mass - 9.8) / 3.0,
                     *last_action, *(mask * 2 - 1)], dtype=np.float32)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--flag", type=int, default=0, choices=[0, 1, 2, 3])
    pa.add_argument("--dps", type=float, default=1000.0)
    pa.add_argument("--src", default="measured",
                    choices=["truth", "measured", "clipped", "override", "alg", "net"])
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
    pa.add_argument("--ckpt", default=None, help="trained e2e estimator checkpoint for --src net "
                                                  "(default results/e2e_v7.pt, else e2e_v8.pt)")
    pa.add_argument("--tilt_obs", action="store_true",
                    help="bound the INS drift with the acceleration-derived thrust axis")
    pa.add_argument("--tilt_tau", type=float, default=0.05)
    pa.add_argument("--rl_ckpt", default=None, help="override the inner-loop policy checkpoint name")
    pa.add_argument("--rl_dir", default=None, help="directory that holds it (see e4_closed_loop.py)")
    pa.add_argument("--id_post", type=float, default=1.5,
                    help="seconds of open-loop excitation logged after the fault for identification")
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
    # --rl_ckpt/--rl_dir mirror scripts/e4_closed_loop.py, so the *fine-tuned* policy can be flown
    # here without editing the table.  The acceptance criterion is a Gazebo number, so the
    # Gazebo port must be able to fly whatever E4 just measured.
    policy = None
    if a.rl_ckpt or a.rl_dir or a.flag in CKPT:
        policy = load_policy(a.rl_ckpt or CKPT.get(a.flag, "shutdown_real_7"), a.rl_dir)
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

    # --- priors + fault IC for EVERY src, exactly like scripts/e4_closed_loop.py ---
    # This used to run only for net/alg, which made the sweep an unfair comparison: truth/clipped started
    # from a free-fall spawned state (wall-time dependent!) while net/alg started from the captured
    # post-fault state, so the rows were not the same draw at all -- and it also explained why the
    # alg/net rows looked impossibly worse than a plain clip.  One `identification()` per run costs
    # ~3 s of simulated time and pins the IC for all four rows, which is what makes them pairable.
    ins = None
    prior, _diag = n.identify(imu, lag)              # applies the fault IC internally
    n.apply_ic()                                     # re-apply it: identical start for every src
    ins = AttitudeINS(quat_to_matrix(n.state[1]))
    # Optional: bound the INS drift with the thrust axis measured from inertial acceleration
    # (gpd_me/tilt.py).  Gazebo feeds the *reported* velocity, which is quantised and noisier
    # than the pybullet arm's, so this is also the deployability check for the observer.
    tob = TiltObserver(tau=a.tilt_tau) if a.tilt_obs else None
    if a.src in ("net", "alg"):
        ckpt = a.ckpt or os.path.join(ME, "results", "e2e_v9.pt")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(ME, "results", "e2e_v8.pt")
        netr = NetRate(ckpt, torch.device("cpu"), prior, a.dps, a.dt, mode=a.src)
        print(f"[gz] src={a.src} ckpt={os.path.basename(ckpt)} compatible={netr.compat}")

    log = {k: [] for k in ("z", "xy", "wt", "wu", "rate", "att", "tilt")}
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
        R_true = quat_to_matrix(quat)
        omega = R_true.T @ omega_w
        wdot = np.zeros(3) if prev_omega is None else (omega - prev_omega) / a.dt
        prev_omega = omega.copy()
        tom = float(lag.thrust.sum()) / a.mass
        gyro, accel = imu.measure(omega, wdot, tom, a.dt, force=True)
        u_cmd = np.clip((last_action + 1.0) * 7.5, 0.0, 15.0)   # raw command sent (N/rotor)
        if a.src == "truth":
            rate = omega
        elif a.src in ("measured", "clipped"):
            rate = gyro
        elif a.src in ("net", "alg"):
            rate = netr.step(accel, gyro, u_cmd, mask)
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

        # the INS integrates the *same* estimate that feeds the controller (no-leakage); dims 0:2
        # of the observation then come from the INS attitude, never the simulator's true attitude
        if ins is not None:
            R = ins.update(rate, a.dt)
            if tob is not None:
                R = tob.correct(R, vel, a.dt, omega=rate)
            q_att = Rotation.from_matrix(R).as_quat()
        else:
            R, q_att = R_true, quat
        target_a, z_body = pid.step(a.dt, pos, q_att, vel, target_pos)
        r_xy = np.hypot(z_body[0], z_body[1])
        if r_xy > 0.26:
            s = 0.26 / r_xy
            z_body = np.array([z_body[0] * s, z_body[1] * s,
                               np.sqrt(1 - (z_body[0] * s) ** 2 - (z_body[1] * s) ** 2)])
        rel = R.T @ z_body
        att = float(np.arccos(np.clip(float(np.dot(rel, np.array([0.0, 0.0, 1.0]))), -1.0, 1.0)))
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
        log["att"].append(att)
        log["tilt"].append(AttitudeINS.tilt_error(R, R_true))
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
          f"att_last={np.rad2deg(np.median(np.array(log['att'])[-w:])):5.1f}deg "
          f"tilt_med={np.rad2deg(np.mean(log['tilt'])):5.1f}deg "
          f"att_med={np.rad2deg(np.median(log['att'])):5.1f}deg "
          f"loop={a.steps/elapsed:5.1f}Hz")
    ex.shutdown(); rclpy.shutdown()


if __name__ == "__main__":
    main()