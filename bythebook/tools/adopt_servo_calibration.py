"""Write the calibration the follower's servos already store to a lerobot calibration file.

    python -m bythebook.tools.adopt_servo_calibration

Reads homing offsets and position limits from the servos (nothing is written to them)
and saves them as rig_state/arm_calibration/follower.json. lerobot then finds file
and servos in agreement and connects without asking for a new calibration.

Why not simply recalibrate: lerobot-calibrate overwrites the offsets and limits in the
servos. Joint angles in existing recordings and trained policies are expressed in the
current calibration, so recalibrating changes what those numbers mean. The base
registration of this package fits joint zero offsets itself, so it does not need a
fresh calibration.
"""
from __future__ import annotations

from .. import rig


def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    rig.ARM_CALIB_DIR.mkdir(parents=True, exist_ok=True)
    robot = SO101Follower(SO101FollowerConfig(port=rig.follower_port(), id="follower", calibration_dir=rig.ARM_CALIB_DIR))
    robot.bus.connect()
    try:
        cal = robot.bus.read_calibration()
    finally:
        robot.bus.disconnect(disable_torque=False)
    for name, c in cal.items():
        span = (c.range_max - c.range_min) * 360 / 4095
        print(f"  {name:14s} id {c.id}  homing_offset {c.homing_offset:6d}  range {c.range_min}-{c.range_max} ({span:.0f} deg)")
        if c.range_max - c.range_min < 200:
            raise SystemExit(f"{name}: stored range is implausibly small; this servo was never calibrated. "
                             "Use lerobot-calibrate instead.")
    robot.calibration = cal
    robot._save_calibration()
    print("saved", robot.calibration_fpath)


if __name__ == "__main__":
    main()
