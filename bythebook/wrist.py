"""Closing the loop with the wrist camera: see the key next to the fingertip, then correct.

The camera is fixed to the gripper, so the fingertip always appears at the same pixel and a
key that is exactly under the fingertip (at the hover height) always appears at one fixed
pixel too, `u_ref`. A press then becomes: hover, find the key in the wrist image, move by
the pixel offset converted to millimetres, descend. The conversion is a 2x2 Jacobian J0
measured once by moving the arm 10 mm in x and y and watching the key move in the image;
it rotates with the gripper's yaw, which forward kinematics provides. No hand-eye
calibration and no wrist-camera intrinsics are needed.

Calibration file: rig_state/wrist_servo.json
    {"u_ref": [u, v], "J0": [[..], [..]], "yaw0": deg, "hover_mm": 25, "key_area_px": n}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import rig
from .cameras import grab
from .geometry import apply_T

SERVO_FILE = rig.STATE_DIR / "wrist_servo.json"
WRIST_SIZE = (1280, 720)
OUT_DIR = rig.RESULTS_DIR / "wrist_servo"       # calibration and per-press debug pictures


@dataclass
class WristServo:
    u_ref: np.ndarray            # pixel where a centred key appears at hover height
    J0: np.ndarray               # 2x2, pixel shift per metre of arm motion (table frame), at yaw0
    yaw0: float                  # gripper yaw (deg, table frame) when J0 was measured
    hover_mm: float
    key_area_px: float           # size of the calibration key in the wrist image, for matching

    def save(self, path=SERVO_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"u_ref": self.u_ref.tolist(), "J0": self.J0.tolist(), "yaw0": self.yaw0,
                                    "hover_mm": self.hover_mm, "key_area_px": self.key_area_px,
                                    "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))

    @staticmethod
    def load(path=SERVO_FILE) -> "WristServo":
        d = json.loads(path.read_text())
        return WristServo(np.array(d["u_ref"], float), np.array(d["J0"], float), float(d["yaw0"]),
                          float(d["hover_mm"]), float(d["key_area_px"]))

    def jacobian(self, yaw_deg: float) -> np.ndarray:
        """Pixel shift per metre of table-frame motion at the current gripper yaw."""
        a = np.radians(yaw_deg - self.yaw0)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return self.J0 @ R.T

    def correction(self, u_key, yaw_deg: float) -> np.ndarray:
        """Table-frame xy move (metres) that brings the key under the fingertip."""
        J = self.jacobian(yaw_deg)
        return np.linalg.solve(J, self.u_ref - np.asarray(u_key, float))


def gripper_yaw_deg(mover) -> float:
    """Heading of the gripper in the table frame: direction from the base's shoulder axis to
    the fingertip, projected on the table plane. Enough for rotating the image Jacobian."""
    q, _ = mover.arm.read()
    tip = apply_T(mover.reg.T, mover.chain.tip(q))
    base = mover.reg.T[:3, 3]
    v = tip[:2] - base[:2]
    return float(np.degrees(np.arctan2(v[1], v[0])))


class KeyFinder:
    """Finds the key nearest to a pixel in a wrist image with SAM (point prompt), falling
    back to the darkest/brightest blob around the pixel when SAM is not available."""

    def __init__(self, perception=None):
        self.p = perception

    def find(self, img, near_uv, expect_area=None, search_px: float = 90.0):
        """-> (centroid (u, v), area px) of the key, or None."""
        u0, v0 = near_uv
        if self.p is not None:
            masks = []
            for du, dv in ((0, 0), (-25, 0), (25, 0), (0, -25), (0, 25)):
                m = self.p.point_mask(img, (u0 + du, v0 + dv))
                if m is not None and m.any():
                    masks.append(m)
            cands = []
            for m in masks:
                ys, xs = np.nonzero(m)
                c = np.array([xs.mean(), ys.mean()])
                area = float(m.sum())
                if np.linalg.norm(c - [u0, v0]) > search_px or area > 0.05 * img.shape[0] * img.shape[1]:
                    continue
                score = np.linalg.norm(c - [u0, v0]) / search_px + (abs(np.log(area / expect_area)) if expect_area else 0)
                cands.append((score, c, area))
            if cands:
                _, c, area = min(cands, key=lambda t: t[0])
                return c, area
        return self._blob(img, near_uv, expect_area, search_px)

    @staticmethod
    def _blob(img, near_uv, expect_area, search_px):
        u0, v0 = int(near_uv[0]), int(near_uv[1])
        r = int(search_px)
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, u0 - r), max(0, v0 - r), min(W, u0 + r), min(H, v0 + r)
        g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        best = None
        for inv in (bw, 255 - bw):
            n, lab, stats, cents = cv2.connectedComponentsWithStats(inv, 8)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 80 or area > 0.6 * inv.size:
                    continue
                c = cents[i] + [x0, y0]
                score = np.linalg.norm(c - [u0, v0]) / search_px + (abs(np.log(area / expect_area)) if expect_area else 0)
                if best is None or score < best[0]:
                    best = (score, c, float(area))
        return None if best is None else (best[1], best[2])


# ----------------------------------------------------------------------------- the servo callable
def wrist_image() -> np.ndarray:
    return grab("wrist", size=WRIST_SIZE, warm=6)


def make_servo(servo: WristServo, finder: KeyFinder, log=print):
    """The callable Mover.press() accepts as `servo`: wrist image -> key -> table-frame
    correction (metres), or None when the key is not found. A picture of every decision is
    written to results/wrist_servo/."""

    def correct(mover, P):
        img = wrist_image()
        found = finder.find(img, servo.u_ref, servo.key_area_px)
        if found is None:
            log("  wrist camera: key not found near the reference pixel; no correction")
            return None
        c, _ = found
        yaw = gripper_yaw_deg(mover)
        d = servo.correction(c, yaw)
        log(f"  wrist camera: key at {np.round(c)} px, reference {np.round(servo.u_ref)} px -> correction "
            f"{np.round(d * 1000, 1)} mm (yaw {yaw:.0f} deg)")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.drawMarker(img, (int(c[0]), int(c[1])), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.drawMarker(img, (int(servo.u_ref[0]), int(servo.u_ref[1])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 30, 2)
        cv2.imwrite(str(OUT_DIR / f"servo_{time.strftime('%H%M%S')}.jpg"), img)
        return d

    return correct
