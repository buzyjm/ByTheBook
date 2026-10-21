"""Carry the base registration over to a new servo calibration.

    python -m bythebook.tools.recalibrated

lerobot-calibrate rewrites the zero (homing offset) and range of every servo, which changes
what the joint angles in degrees mean. The physical arm is unchanged, so the registration
only needs its joint zero offsets shifted by the same amounts:

    deg_new = deg_old + shift,   shift = (homing_old - homing_new - d_mid) * 360 / 4095

(lerobot reports Present_Position = actual - homing_offset, and degrees are measured from
the middle of the calibrated range.)

The registration's gripper value is converted through the raw servo position. The new
calibration is then adopted as the one the registration refers to.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from .. import rig
from ..kinematics import ARM_JOINTS
from ..registration import Registration

CAL_FILE = rig.ARM_CALIB_DIR / "follower.json"


def shifts_deg(old: dict, new: dict) -> dict:
    out = {}
    for m in ARM_JOINTS:
        o, n = old[m], new[m]
        d_mid = (n["range_min"] + n["range_max"]) / 2 - (o["range_min"] + o["range_max"]) / 2
        out[m] = (o["homing_offset"] - n["homing_offset"] - d_mid) * 360 / 4095
    return out


def gripper_pct(old: dict, new: dict, pct_old: float) -> float:
    o, n = old["gripper"], new["gripper"]
    raw_old = o["range_min"] + pct_old / 100 * (o["range_max"] - o["range_min"])
    raw_new = raw_old + o["homing_offset"] - n["homing_offset"]
    return float(np.clip((raw_new - n["range_min"]) / (n["range_max"] - n["range_min"]) * 100, 0, 100))


def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    old = json.loads(CAL_FILE.read_text())
    robot = SO101Follower(SO101FollowerConfig(port=rig.follower_port(), id="follower", calibration_dir=rig.ARM_CALIB_DIR))
    robot.bus.connect()
    try:
        cal = robot.bus.read_calibration()
    finally:
        robot.bus.disconnect(disable_torque=False)
    new = {m: {"id": c.id, "drive_mode": c.drive_mode, "homing_offset": c.homing_offset,
               "range_min": c.range_min, "range_max": c.range_max} for m, c in cal.items()}
    if all(old[m][k] == new[m][k] for m in new for k in ("homing_offset", "range_min", "range_max")):
        print("servo calibration unchanged; nothing to do")
        return
    sh = shifts_deg(old, new)
    print("zero shifts (deg):", {m: round(v, 2) for m, v in sh.items()})

    reg = Registration.load()
    reg.joint_offsets_deg = [float(reg.joint_offsets_deg[i] - sh[m]) for i, m in enumerate(ARM_JOINTS)]
    if reg.gripper_closed is not None:
        g_old = reg.gripper_closed
        reg.gripper_closed = gripper_pct(old, new, g_old)
        print(f"gripper closed value {g_old:.1f} -> {reg.gripper_closed:.1f}")
    print("joint offsets now", np.round(reg.joint_offsets_deg, 2).tolist())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copy(rig.REGISTRATION_FILE, rig.STATE_DIR / f"base_registration_before_{stamp}.json")
    shutil.copy(CAL_FILE, rig.ARM_CALIB_DIR / f"follower_before_{stamp}.json")
    reg.save()
    robot.calibration = cal
    robot._save_calibration()
    print(f"registration converted and saved; new calibration adopted ({CAL_FILE})")


if __name__ == "__main__":
    main()
