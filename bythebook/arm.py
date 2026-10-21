"""Arm I/O (real follower or a simulated stand-in) and the motion primitives built on it.

Safety model
  - Nothing here moves the real arm unless a RealArm was created with torque=True, which
    the command-line tools only do when --execute is given.
  - Free-space moves are joint interpolations whose whole path is checked (by FK) to keep
    the tip above a floor height and inside the workspace box before the first command.
  - Descents are straight lines in small steps, slow, with contact detection from servo load.
  - The arm is parked in the pose it was found in before torque is released, because the
    servos go limp on disconnect.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import rig
from .geometry import apply_T, inv_T
from .kinematics import ARM_JOINTS, IKResult

MOTORS = (*ARM_JOINTS, "gripper")
LOAD_JOINTS = (1, 2, 3)        # shoulder_lift, elbow_flex, wrist_flex carry a vertical press


# ============================================================================ I/O
class RealArm:
    """The SO-101 follower through lerobot.

    torque=True   normal connection: lerobot configures the servos and they hold position
    torque=False  servos limp, for guiding the arm by hand (registration)
    torque=None   read only: no register is written
    """

    def __init__(self, torque: bool | None = True, port: str | None = None, robot_id: str = "follower",
                 max_step_deg: float = 8.0, position_p: int = 32):
        self.torque, self.port, self.robot_id, self.max_step_deg = torque, port or rig.follower_port(), robot_id, max_step_deg
        # Position-loop gain written to the servos at connection. lerobot's default of 16 is soft
        # (meant for teleoperation); 32, the Feetech default, halves the gravity sag and doubles
        # the force a given position error produces when pressing.
        self.position_p = position_p
        self.robot = None

    def connect(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # A calibration adopted from the servos (bythebook/tools/adopt_servo_calibration.py) takes
        # precedence over lerobot's default location.
        own = rig.ARM_CALIB_DIR / f"{self.robot_id}.json"
        cfg = SO101FollowerConfig(port=self.port, id=self.robot_id, max_relative_target=float(self.max_step_deg),
                                  disable_torque_on_disconnect=False, position_p_coefficient=self.position_p,
                                  calibration_dir=rig.ARM_CALIB_DIR if own.exists() else None)
        self.robot = SO101Follower(cfg)
        bus = self.robot.bus
        if self.torque:
            if not self.robot.calibration:
                raise RuntimeError(
                    f"no calibration file for id '{self.robot_id}' ({self.robot.calibration_fpath}). "
                    "Run `python -m bythebook.tools.adopt_servo_calibration` (keeps the servos' current "
                    "calibration) or lerobot-calibrate, see README.md.")
            # Same steps as SO101Follower.connect(), plus one guard: the goal register is set to
            # the present position while the servos are still limp, so that switching torque on
            # cannot make the arm jump to a stale goal.
            bus.connect()
            if not bus.is_calibrated:
                bus.disconnect(disable_torque=False)
                self.robot = None
                raise RuntimeError("calibration file and servo calibration differ: rerun "
                                   "`python -m bythebook.tools.adopt_servo_calibration` or lerobot-calibrate")
            # Feetech buses occasionally drop a status packet; a failed write would otherwise
            # abort the whole connection, so the configuration is retried a few times.
            for attempt in range(4):
                try:
                    bus.disable_torque(num_retry=3)
                    bus.sync_write("Goal_Position", bus.sync_read("Present_Position", num_retry=3), num_retry=3)
                    self.robot.configure()
                    break
                except ConnectionError as e:
                    if attempt == 3:
                        bus.disconnect(disable_torque=False)
                        self.robot = None
                        raise
                    print(f"note: servo bus hiccup while configuring ({e}); retrying")
                    time.sleep(0.5)
        else:
            bus.connect()
            if not bus.calibration:
                # No file yet: use what the servos themselves store. Fine for reading.
                bus.calibration = bus.read_calibration()
                print("note: no calibration file, using the calibration stored in the servos (read only)")
            if self.torque is False:
                bus.disable_torque()
        return self

    def disconnect(self, release: bool = True):
        """release=False leaves the servos holding their pose (used when parking failed)."""
        if self.robot is None:
            return
        if self.torque and release:
            self.robot.bus.disable_torque()
        self.robot.bus.disconnect(disable_torque=False)
        self.robot = None

    def read(self):
        pos = self.robot.bus.sync_read("Present_Position", num_retry=3)
        return np.array([pos[m] for m in ARM_JOINTS], float), float(pos["gripper"])

    def read_effort(self) -> np.ndarray:
        load = self.robot.bus.sync_read("Present_Load", list(ARM_JOINTS), normalize=False, num_retry=3)
        return np.array([load[m] for m in ARM_JOINTS], float)

    def command(self, q_deg, gripper: float | None = None):
        if not self.torque:
            raise RuntimeError("this connection is read-only")
        action = {f"{m}.pos": float(v) for m, v in zip(ARM_JOINTS, q_deg)}
        if gripper is not None:
            action["gripper.pos"] = float(gripper)
        for attempt in range(3):
            try:
                self.robot.send_action(action)
                return
            except ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(0.02)

    def calibrated_range_deg(self) -> np.ndarray:
        cal = self.robot.bus.calibration
        half = np.array([(cal[m].range_max - cal[m].range_min) / 2 * 360 / 4095 for m in ARM_JOINTS])
        return np.stack([-half, half], axis=1)


class SimArm:
    """Kinematic stand-in used by the tests and by dry runs.

    The joints follow commands exactly. A horizontal surface (table frame) pushes back:
    servo load on the pitch joints grows with how far the commanded tip is below it.
    Buttons on the surface register a press once pushed `travel_mm` deep.
    """

    def __init__(self, registration, q0_deg=(18.0, -101.8, 92.3, 44.7, -4.6), surface_z=0.0, buttons=None,
                 load_per_mm=60.0, noise=4.0, seed=0, droop_deg=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self.reg, self.chain = registration, registration.chain()
        self.droop = np.array(droop_deg, float)      # the encoders read the commanded pose plus this (gravity sag)
        self.q, self.gripper = np.array(q0_deg, float), 5.0
        self.surface_z, self.load_per_mm, self.noise = surface_z, load_per_mm, noise
        self.buttons = buttons or {}        # name -> dict(xy=(x, y), radius_mm, travel_mm, z=top height)
        self.presses, self._down = [], set()
        self.rng = np.random.default_rng(seed)
        self.torque = True
        self.log = []

    def connect(self):
        return self

    def disconnect(self, release: bool = True):
        pass

    def read(self):
        return getattr(self, "q_actual", self.q + self.droop).copy(), self.gripper

    def tip_table(self):
        return apply_T(self.reg.T, self.chain.tip(self.q + self.droop))

    def _surface_at(self, xy):
        z = self.surface_z
        for b in self.buttons.values():
            if np.linalg.norm(np.asarray(xy) - b["xy"]) * 1000 <= b["radius_mm"]:
                z = max(z, b["z"])
        return z

    def read_effort(self):
        p = self.tip_table()
        depth_mm = max(0.0, (self._surface_at(p[:2]) - p[2]) * 1000)
        e = self.rng.normal(0, self.noise, 5) + np.array([0, 120, -80, 40, 0])      # gravity-like bias
        e[list(LOAD_JOINTS)] += depth_mm * self.load_per_mm * np.array([1.0, -0.8, 0.5])
        return e

    def command(self, q_deg, gripper=None):
        self.q = np.array(q_deg, float)
        if gripper is not None:
            self.gripper = float(gripper)
        p = self.tip_table()
        # the surface stops the fingertip: the servos stall at the last pose above it
        if p[2] >= self._surface_at(p[:2]) - 1e-6:
            self.q_actual = self.q + self.droop
        self.log.append(p)
        for name, b in self.buttons.items():
            inside = np.linalg.norm(p[:2] - b["xy"]) * 1000 <= b["radius_mm"]
            deep = (b["z"] - p[2]) * 1000 >= b["travel_mm"]
            if inside and deep and name not in self._down:
                self._down.add(name)
                self.presses.append(name)
            elif not (inside and deep):
                self._down.discard(name)

    def calibrated_range_deg(self):
        return None


# ============================================================================ motion
@dataclass
class Limits:
    joint_speed_deg_s: float = 25.0
    press_speed_mm_s: float = 6.0
    rate_hz: float = 30.0
    travel_z: float = 0.10                 # free-space moves happen at this height above the table (m)
    floor_z: float = 0.015                 # no free-space path may bring the tip below this (m)
    # Workspace box in the BASE frame (m), from the reach map of a downward-pointing tip.
    x_range: tuple = (0.10, 0.30)
    y_range: tuple = (-0.18, 0.18)
    max_tilt_deg: float = 35.0
    max_joint_jump_deg: float = 6.0        # between consecutive IK solutions of a straight-line move
    contact_threshold: float = 30.0        # load units above baseline; 50-60 seen when the arm is blocked
    stall_mm: float = 1.0                  # rise of measured-minus-commanded tip height above its free-motion value
    stop_mm: float = 0.3                   # measured tip no longer descending: change over the last 3 steps above -this
    settle_tol_mm: float = 0.7             # closed-loop correction of the hover point stops below this error
    settle_iters: int = 4
    settle_gain: float = 0.7               # fraction of the measured error applied per iteration
    settle_max_mm: float = 10.0            # total correction allowed
    contact_blind_mm: float = 4.0          # no contact decision during the first millimetres of a descent
    contact_window_mm: float = 12.0        # no contact decision higher than this above the expected surface
    contact_steps: int = 4                 # a servo stick-slip clears within a few steps; a contact persists
    press_load: float = 500.0              # after contact, push until the load rises this much above free motion
    press_depth_mm: float = 18.0           # or until the command is this far below the contact point
    # 500 / 18 mm are what the LEADZM scale's keys needed (350 was not always enough).
    stall_end_mm: float = 2.5              # tip this far behind the command at the end of a descent = contact
    servo_max_mm: float = 15.0             # largest wrist-camera correction accepted


@dataclass
class PressResult:
    contact: bool
    contact_z_mm: float | None
    lowest_z_mm: float
    expected_z_mm: float
    trace: list = field(default_factory=list)      # (commanded z_mm, stall rise mm, load deviation) per step
    push_load: float = 0.0                           # load rise reached while pushing after contact
    late_contact: bool = False                       # contact only noticed at the bottom of the descent
    servo_mm: list = field(default_factory=lambda: [0.0, 0.0])   # wrist-camera correction applied


class MotionError(RuntimeError):
    pass


class Mover:
    def __init__(self, arm, registration, limits: Limits | None = None, sleep=time.sleep):
        self.arm, self.reg, self.lim, self.sleep = arm, registration, limits or Limits(), sleep
        self.chain = registration.chain()
        self.T_base_table = inv_T(registration.T)
        self.down_base = self.T_base_table[:3, :3] @ np.array([0, 0, -1.0])
        self.bounds = self.chain.joint_bounds_deg(arm.calibrated_range_deg())
        self.gripper = registration.gripper_closed if getattr(registration, "gripper_closed", None) is not None else 2.0
        self.q_park, _ = arm.read()
        self.q_cmd = self.q_park.copy()
        # Systematic landing error measured on the appliance (table frame, metres); see
        # `python -m bythebook.tools.press_mark --measure-bias`.
        self.landing_bias = np.asarray(getattr(registration, "landing_bias", None) or [0.0, 0.0, 0.0], float)
        self.landing_samples = list(getattr(registration, "landing_samples", None) or [])

    def bias_at(self, P_table) -> np.ndarray:
        """Correction for a target: the constant bias plus, when measured samples exist, their
        value interpolated over the table plane (the residual error of the arm model varies
        slowly with position, mostly with y). One sample: its value. Two or more: the least
        squares plane bias = a + b*x + c*y, or inverse-distance weights when they are collinear."""
        if not self.landing_samples:
            return self.landing_bias
        P = np.asarray(P_table, float)
        S = np.array([s["P"][:2] for s in self.landing_samples], float)
        B = np.array([s["bias"][:2] for s in self.landing_samples], float)
        if len(S) == 1:
            b = B[0]
        elif len(S) == 2:
            t = np.clip(((P[:2] - S[0]) @ (S[1] - S[0])) / max(1e-9, (S[1] - S[0]) @ (S[1] - S[0])), -0.5, 1.5)
            b = B[0] + t * (B[1] - B[0])
        else:
            A = np.c_[np.ones(len(S)), S - S.mean(0)]
            coef, *_ = np.linalg.lstsq(A, B, rcond=None)
            b = coef[0] + (P[:2] - S.mean(0)) @ coef[1:]
        return np.array([b[0], b[1], 0.0]) + self.landing_bias

    # ---------------------------------------------------------------- helpers
    def tip_table(self, q=None) -> np.ndarray:
        return apply_T(self.reg.T, self.chain.tip(self.q_cmd if q is None else q))

    def solve(self, P_table, seed=None, canonical: bool = False, approach=None) -> IKResult:
        """IK for a table-frame tip target. Seeded from the current command by default, so that
        straight-line moves stay continuous. canonical=True ignores the current pose and picks
        the least-tilted of the standard elbow-up configurations (used for free-space moves,
        where continuity with the folded rest pose would only drag the arm into odd postures)."""
        P_base = apply_T(self.T_base_table, np.asarray(P_table, float))
        if not (self.lim.x_range[0] <= P_base[0] <= self.lim.x_range[1]
                and self.lim.y_range[0] <= P_base[1] <= self.lim.y_range[1]):
            raise MotionError(f"target {np.round(P_base * 1000)} mm (base frame) is outside the workspace box")
        seed_deg = None if canonical else (self.q_cmd if seed is None else seed)
        d_base = self.down_base if approach is None else self.T_base_table[:3, :3] @ np.asarray(approach, float)
        r = self.chain.ik(P_base, d_base, seed_deg=seed_deg, bounds_deg=self.bounds,
                          max_tilt_deg=self.lim.max_tilt_deg)
        if not r.ok:
            raise MotionError(f"no IK solution: position error {r.pos_err_mm:.1f} mm, tilt {r.tilt_deg:.0f} deg")
        return r

    def _path(self, q_to):
        steps = max(2, int(np.ceil(np.max(np.abs(q_to - self.q_cmd)) / self.lim.joint_speed_deg_s * self.lim.rate_hz)))
        return [self.q_cmd + (q_to - self.q_cmd) * s for s in np.linspace(0, 1, steps + 1)[1:]]

    def move_joints(self, q_to, check_floor: bool = True):
        """Joint-interpolated move. The whole path is validated before anything is sent."""
        q_to = np.asarray(q_to, float)
        path = self._path(q_to)
        if check_floor:
            z_start = self.tip_table()[2]
            floor = min(self.lim.floor_z, z_start - 1e-4)       # allow leaving a low parked pose upwards
            zs = [self.tip_table(q)[2] for q in path]
            if min(zs) < floor:
                raise MotionError(f"path would bring the tip to {min(zs)*1000:.0f} mm above the table (floor "
                                  f"{floor*1000:.0f} mm)")
        for q in path:
            self.arm.command(q, self.gripper)
            self.q_cmd = q
            self.sleep(1.0 / self.lim.rate_hz)

    def move_linear(self, P_to, speed_mm_s=None, step_mm=0.5, on_step=None, approach=None) -> bool:
        """Straight tip motion in the table frame. on_step(P) -> True stops early. Returns
        True if it ran to the end."""
        P_from = self.tip_table()
        P_to = np.asarray(P_to, float)
        dist_mm = np.linalg.norm(P_to - P_from) * 1000
        n = max(1, int(np.ceil(dist_mm / step_mm)))
        dt = (dist_mm / n) / (speed_mm_s or self.lim.press_speed_mm_s)
        waypoints = [P_from + (P_to - P_from) * s for s in np.linspace(0, 1, n + 1)[1:]]
        def solve_line(first_canonical: bool):
            qs, seed = [], self.q_cmd
            for j, P in enumerate(waypoints):            # solve everything first: fail before moving
                q = self.solve(P, seed=seed, canonical=(first_canonical and j == 0), approach=approach).q_deg
                # The IK can land on another branch (elbow up/down, wrist flipped). A straight line
                # must be a continuous joint path; anything else is a jump and is refused.
                jump = float(np.max(np.abs(q - seed)))
                if jump > self.lim.max_joint_jump_deg and not (first_canonical and j == 0):
                    raise MotionError(f"straight-line move is not continuous in joint space ({jump:.0f} deg between steps)")
                qs.append(q)
                seed = q
            return qs

        try:
            qs = solve_line(False)
        except MotionError:
            # Near a joint limit the continuation from the current pose can get stuck on a bad
            # branch; the canonical configuration of the first waypoint is tried as a fresh start,
            # reached by a short checked joint move.
            qs = solve_line(True)
            if float(np.max(np.abs(qs[0] - self.q_cmd))) > self.lim.max_joint_jump_deg:
                self.move_joints(qs[0])
        prev = self.q_cmd
        for P, q in zip(waypoints, qs):
            self.arm.command(q, self.gripper)
            self.q_cmd = q
            # never faster in joint space than a free-space move, whatever the tip speed asks for
            self.sleep(max(dt, float(np.max(np.abs(q - prev))) / self.lim.joint_speed_deg_s))
            prev = q
            if on_step is not None and on_step(P):
                return False
        return True

    # ---------------------------------------------------------------- primitives
    def _tilt_deg(self) -> float:
        d = self.reg.T[:3, :3] @ self.chain.tool_dir(self.q_cmd)
        return float(np.degrees(np.arccos(np.clip(-d[2], -1, 1))))

    def _lift(self):
        """Up to travel height. A straight line only when the fingers already point down (as
        after a press); from any other pose, such as the folded rest pose, a checked joint move."""
        cur = self.tip_table()
        if cur[2] >= self.lim.travel_z - 0.005:
            return
        up = [cur[0], cur[1], self.lim.travel_z]
        if self._tilt_deg() <= self.lim.max_tilt_deg:
            try:
                self.move_linear(up, speed_mm_s=40.0, step_mm=2.0)
                return
            except MotionError:
                pass
        try:
            self.move_joints(self.solve(up, canonical=True).q_deg)
        except MotionError:
            pass                                    # the point above the parked tip may be unreachable

    def hover(self, P_table, approach=None):
        """Go above P_table at travel height: up, across, done."""
        top = np.array([P_table[0], P_table[1], self.lim.travel_z])
        q_top = self.solve(top, canonical=True, approach=approach).q_deg   # refuses unreachable targets before any motion
        self._lift()
        self.move_joints(q_top)

    def press(self, P_table, hover_mm=25.0, max_overtravel_mm=6.0, dwell_s=0.4, hold: bool = False,
              approach=None, servo=None) -> PressResult:
        """Press on P_table (the expected surface point, table frame, metres) along `approach`,
        a unit vector pointing into the surface (default straight down; pass the negated
        panel normal for a sloped control panel). `servo(mover, P)` may return a table-frame
        xy correction (metres) measured at the hover point, e.g. from the wrist camera; it is
        applied before the descent.

        Descend until contact, then keep pushing: the servos are position controlled with a
        soft gain, so force comes from commanding deeper than the surface. The push stops when
        the load has risen by `press_load` or the command is `press_depth_mm` below contact."""
        P = np.asarray(P_table, float) + self.bias_at(P_table)
        if P[2] < -0.002:
            raise MotionError("press target is below the table")
        d = np.array([0.0, 0.0, -1.0]) if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
        if d[2] > -0.5:
            raise MotionError("approach direction must point mostly downwards")
        bottom = P + d * (max_overtravel_mm + self.lim.press_depth_mm) / 1000
        seed = self.solve([P[0], P[1], self.lim.travel_z], canonical=True, approach=d).q_deg
        self.solve(bottom, seed=self.solve(P - d * hover_mm / 1000, seed=seed, approach=d).q_deg, approach=d)
        self.hover(P, approach=d)
        above = P - d * hover_mm / 1000
        self.move_linear(above, speed_mm_s=30.0, step_mm=1.0, approach=d)
        # Closed loop on the encoders: the arm sags under its own weight by a few millimetres,
        # so the commanded pose is nudged until the measured fingertip sits on the hover point.
        # The same correction is then carried down through the press.
        correction = np.zeros(3)
        for _ in range(self.lim.settle_iters):
            self.sleep(0.6)                                   # let the servos finish moving first
            err = above - self.tip_table(self.arm.read()[0])
            if np.linalg.norm(err) * 1000 < self.lim.settle_tol_mm:
                break
            step = err * self.lim.settle_gain
            if np.linalg.norm(correction + step) * 1000 > self.lim.settle_max_mm:
                break
            correction += step
            self.move_linear(above + correction, speed_mm_s=10.0, step_mm=0.5, approach=d)
        self.sleep(0.6)
        servo_delta = np.zeros(3)
        if servo is not None:
            delta = servo(self, P)
            if delta is not None:
                delta = np.array([delta[0], delta[1], 0.0], float)
                if np.linalg.norm(delta) <= self.lim.servo_max_mm / 1000:
                    servo_delta = delta
                    P = P + delta
                    above = above + delta
                    self.move_linear(above + correction, speed_mm_s=10.0, step_mm=0.5, approach=d)
                    self.sleep(0.4)

        history, trace, hits, state = [], [], 0, {"z": None}
        z_meas_hist = []

        def along(p):                                     # progress along the approach axis, mm
            return float((np.asarray(p) - P) @ d * 1000)

        def on_step(Pc):
            nonlocal hits
            self.sleep(0.03)
            e = self.arm.read_effort()[list(LOAD_JOINTS)]
            q_meas, _ = self.arm.read()
            # Contact shows as the measured fingertip no longer following the command along the
            # approach axis, together with a load rise. Load alone also rises when a joint lags.
            s_meas = along(self.tip_table(q_meas))
            z_meas = -s_meas                                  # "height" along the axis: bigger = further out
            z_meas_hist.append(z_meas)
            stall_raw = float(along(Pc - correction) - s_meas)
            # Baselines = median of samples at least 4 steps old, so slow drift (gravity sag that
            # changes with posture) is tracked but the change at contact is not absorbed.
            base = np.median(history[:-3][-10:], axis=0) if len(history) >= 6 else None
            history.append(np.concatenate([e, [stall_raw]]))
            dev = 0.0 if base is None else float(np.linalg.norm(e - base[:3]))
            stall = 0.0 if base is None else stall_raw - float(base[3])
            trace.append((-along(Pc - correction), stall, dev))
            descended_mm = along(Pc - correction) - along(above)
            if descended_mm < self.lim.contact_blind_mm:      # still settling from the hover point
                return False
            # Blocked means the measured fingertip stops going down while the command keeps going;
            # growing gravity sag only slows it. Both signs together make a contact.
            if -along(Pc - correction) > self.lim.contact_window_mm:
                return False                                   # the key cannot be touched this high up
            stopped = len(z_meas_hist) >= 4 and (z_meas_hist[-1] - z_meas_hist[-4]) > -self.lim.stop_mm
            hits = hits + 1 if (stopped and stall > self.lim.stall_mm and dev > self.lim.contact_threshold) else 0
            if hits >= self.lim.contact_steps:
                state["z"] = float(self.tip_table(q_meas)[2] * 1000)     # where the tip really stopped
                return True
            return False

        bottom_cmd = P + d * max_overtravel_mm / 1000 + correction
        self.move_linear(bottom_cmd, on_step=on_step, approach=d)
        contact = state["z"] is not None
        if not contact:
            # The detector may miss a soft contact; a tip that stays behind the command at the
            # bottom of the descent is blocked by something all the same.
            self.sleep(0.3)
            q_meas, _ = self.arm.read()
            behind = along(bottom_cmd - correction) - along(self.tip_table(q_meas))
            if behind > self.lim.stall_end_mm:
                contact = True
                state["z"] = float(self.tip_table(q_meas)[2] * 1000)
                state["late"] = True
        pushed = {"load": 0.0}
        if contact:
            base_load = np.median(history[:-3][-10:], axis=0)[:3]

            def push_step(Pc):
                self.sleep(0.05)
                pushed["load"] = float(np.linalg.norm(self.arm.read_effort()[list(LOAD_JOINTS)] - base_load))
                return pushed["load"] >= self.lim.press_load

            cur = self.tip_table()
            self.move_linear(cur + d * self.lim.press_depth_mm / 1000, speed_mm_s=4.0, step_mm=0.5, on_step=push_step,
                             approach=d)
        lowest = float(self.tip_table(self.arm.read()[0])[2] * 1000)
        state["load"] = pushed["load"]
        self.sleep(dwell_s)
        if not hold:                                   # hold=True leaves the tip down for a measurement
            self.move_linear(above + correction, speed_mm_s=30.0, step_mm=1.0, approach=d)
        return PressResult(contact, state["z"], lowest, float(P[2] * 1000), trace, state.get("load", 0.0),
                           bool(state.get("late", False)), (servo_delta[:2] * 1000).tolist())

    def retreat(self, P_table, back_mm: float = 90.0):
        """After a press: up to travel height and `back_mm` towards the base, out of the
        overhead camera's view of the appliance, without going back to the parked pose."""
        P = np.asarray(P_table, float)
        back = self.reg.T[:3, :3] @ np.array([-1.0, 0.0, 0.0])      # the base's -x axis in the table frame
        self._lift()
        for dist in (back_mm, back_mm * 0.6, back_mm * 0.3):
            try:
                q = self.solve(np.array([P[0], P[1], self.lim.travel_z]) + back * dist / 1000, canonical=True).q_deg
                self.move_joints(q)
                return
            except MotionError:
                continue

    def park(self):
        """Back to the rest pose (rig_state/rest_pose.json, recorded with
        `python -m bythebook.arm set-rest`), or to the pose the arm was found in, via travel height."""
        self._lift()
        q_rest = self.q_park
        if rig.REST_POSE_FILE.exists():
            q_rest = np.array(json.loads(rig.REST_POSE_FILE.read_text())["q_deg"], float)
        self.move_joints(q_rest, check_floor=False)


class Session:
    """`with Session(arm, registration) as mover:` parks the arm before releasing torque.
    If parking fails the servos are left holding, so the arm does not drop."""

    def __init__(self, arm, registration, limits=None, sleep=time.sleep):
        self.arm, self.reg, self.limits, self.sleep = arm, registration, limits, sleep

    def __enter__(self) -> Mover:
        self.arm.connect()
        self.mover = Mover(self.arm, self.reg, self.limits, self.sleep)
        return self.mover

    def __exit__(self, *exc):
        try:
            self.mover.park()
        except Exception as e:      # noqa: BLE001
            print(f"\nPARKING FAILED ({e}). The servos are left holding the arm; support it by hand "
                  "before powering off or releasing torque.")
            self.arm.disconnect(release=False)
            return False
        self.arm.disconnect()
        return False


# ============================================================================ CLI
def main():
    from .registration import Registration

    ap = argparse.ArgumentParser(description="Arm state and single presses. Nothing moves without --execute.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("state", help="read joints and load, change nothing")
    sub.add_parser("set-rest", help="record the arm's current pose (put it there by hand, torque off) as the rest pose")
    p = sub.add_parser("press", help="press down at a table-frame point (mm)")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("z", type=float)
    p.add_argument("--execute", action="store_true", help="really move the arm (default: simulate)")
    p.add_argument("--threshold", type=float)
    a = ap.parse_args()

    if a.cmd == "set-rest":
        arm = RealArm(torque=None).connect()
        q, g = arm.read()
        arm.disconnect()
        rig.REST_POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
        rig.REST_POSE_FILE.write_text(json.dumps({"q_deg": q.tolist(), "gripper": g, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        print("rest pose saved:", {m: round(float(v), 1) for m, v in zip(ARM_JOINTS, q)}, "->", rig.REST_POSE_FILE)
        return
    if a.cmd == "state":
        arm = RealArm(torque=None).connect()
        q, g = arm.read()
        print("joints (deg):", {m: round(float(v), 1) for m, v in zip(ARM_JOINTS, q)}, "gripper:", round(g, 1))
        print("load:", {m: float(v) for m, v in zip(ARM_JOINTS, arm.read_effort())})
        try:
            reg = Registration.load()
            print("tip in table frame (mm):", np.round(apply_T(reg.T, reg.chain().tip(q)) * 1000, 1))
        except FileNotFoundError as e:
            print(e)
        arm.disconnect()
        return

    reg = Registration.load()
    P = np.array([a.x, a.y, a.z]) / 1000
    limits = Limits()
    if a.threshold:
        limits.contact_threshold = a.threshold
    if a.execute:
        if input(f"The arm will press at {P * 1000} mm. Workspace clear? Type 'go': ").strip() != "go":
            return
        arm, sleep = RealArm(torque=True), time.sleep
    else:
        ro = RealArm(torque=None).connect()
        q0, _ = ro.read()
        ro.disconnect()
        arm, sleep = SimArm(reg, q0_deg=q0, surface_z=P[2]), (lambda s: None)
        print("dry run from the arm's current pose (pass --execute to move it)")
    with Session(arm, reg, limits, sleep) as mover:
        r = mover.press(P)
    print(f"contact: {r.contact}  at z = {r.contact_z_mm} mm (expected {r.expected_z_mm:.1f}), lowest {r.lowest_z_mm:.1f} mm, "
          f"push load {r.push_load:.0f}")
    print("commanded z : stall mm : load deviation")
    print(" ".join(f"{z:.1f}:{st:.1f}:{d:.0f}" for z, st, d in r.trace))


if __name__ == "__main__":
    main()
