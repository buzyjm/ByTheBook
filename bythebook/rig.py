"""Rig description: where the cameras are, how they were calibrated, where files live.

Frames used everywhere in this package
    table : defined by rig-calibrate (origin at the calibration board center, z up, metres)
    cam   : OpenCV camera frame (x right, y down, z forward)
    base  : URDF base_link of the follower arm
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
RIG_MAP = Path(os.environ.get("RIG_MAP", "~/rig_map.json")).expanduser()
RIG_CALIB = Path(os.environ.get("RIG_CALIB", "~/rig_calib/latest.json")).expanduser()
# Rig-specific files derived by the tools in this package (not committed).
STATE_DIR = Path(os.environ.get("BYTHEBOOK_STATE_DIR", REPO_DIR / "rig_state")).expanduser()
HIRES_FILE = STATE_DIR / "hires.json"
REGISTRATION_FILE = STATE_DIR / "base_registration.json"
ARM_CALIB_DIR = STATE_DIR / "arm_calibration"
REST_POSE_FILE = STATE_DIR / "rest_pose.json"      # where the arm goes at the end of every session
URDF = PKG_DIR / "assets" / "so101" / "so101_new_calib.urdf"
RESULTS_DIR = REPO_DIR / "results"

FIXED_ROLES = ("ego", "side_left", "side_right")

# Same definitions as rig-calibrate, so any of its boards can be recognised.
BOARDS = {
    "small": dict(squares=(5, 6), square_mm=15.0, marker_mm=11.0, dictionary="DICT_5X5_50", first_id=0),
    "letter": dict(squares=(4, 5), square_mm=48.0, marker_mm=38.0, dictionary="DICT_5X5_100", first_id=20),
    "tabloid": dict(squares=(5, 7), square_mm=52.0, marker_mm=42.0, dictionary="DICT_5X5_100", first_id=40),
    "calibio": dict(squares=(4, 5), square_mm=48.0, marker_mm=36.0, dictionary="DICT_4X4_50", first_id=0),
}


@dataclass(frozen=True)
class CameraModel:
    role: str
    path: str                      # /dev/v4l/by-id/...
    K: np.ndarray                  # 3x3
    dist: np.ndarray               # OpenCV distortion coefficients
    size: tuple[int, int]          # (width, height) the intrinsics refer to
    T_table_cam: np.ndarray | None  # 4x4, None for the uncalibrated wrist camera

    def with_mode(self, size, A) -> "CameraModel":
        """Same camera in another video mode. `A` (3x3) maps pixels of the calibrated
        mode to pixels of the new mode; distortion is unchanged by a scale and crop."""
        return replace(self, K=np.asarray(A, float) @ self.K, size=(int(size[0]), int(size[1])))


def load_rig_map() -> dict:
    return json.loads(RIG_MAP.read_text())


def load_cameras(hires: bool = False) -> dict[str, CameraModel]:
    """Calibrated cameras by role. With hires=True, cameras that have an entry in
    rig_state/hires.json are returned in their high-resolution mode."""
    calib = json.loads(RIG_CALIB.read_text())
    modes = json.loads(HIRES_FILE.read_text()) if (hires and HIRES_FILE.exists()) else {}
    cams = {}
    for role, c in calib["cameras"].items():
        ext = calib.get("extrinsics", {}).get(role)
        cam = CameraModel(
            role=role,
            path=c["path"],
            K=np.array(c["K"], float),
            dist=np.array(c["dist"], float),
            size=tuple(c["image_size"]),
            T_table_cam=np.array(ext["T_table_cam"], float) if ext else None,
        )
        if role in modes:
            cam = cam.with_mode(modes[role]["size"], modes[role]["A"])
        cams[role] = cam
    return cams


def camera_path(role: str) -> str:
    """Device path of any camera in rig_map.json, calibrated or not (e.g. wrist)."""
    for c in load_rig_map()["cameras"]:
        if c["role"] == role:
            return c["path"]
    raise KeyError(f"no camera with role {role!r} in {RIG_MAP}")


def follower_port() -> str:
    return load_rig_map()["arms"]["follower"]
