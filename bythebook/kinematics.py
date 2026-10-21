"""Forward and inverse kinematics of the SO-101 follower, straight from the URDF.

Kept dependency-free (numpy + scipy) on purpose: the press tip is not a URDF frame
(this rig has custom fingertips), and the registration fits joint zero offsets, both
of which are awkward with an off-the-shelf solver. placo is only used in the tests
as an independent check of the FK.

Joint angles are in degrees in the lerobot convention (zero = middle of the calibrated
range). `offsets_deg` is added before the URDF is evaluated, to absorb the difference
between that zero and the URDF zero.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import make_T
from .rig import URDF

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Direction the fingers point, in the gripper_link frame (towards gripper_frame_link).
TOOL_AXIS = np.array([0.0, 0.0, -1.0])


def _origin_T(el) -> np.ndarray:
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return make_T(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)


@dataclass
class IKResult:
    q_deg: np.ndarray        # lerobot-convention joint angles
    pos_err_mm: float
    tilt_deg: float          # angle between the tool axis and the requested approach direction
    ok: bool


class Chain:
    def __init__(self, urdf: Path = URDF, offsets_deg=None, tip_offset=None):
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        self.origins, lim = [], []
        parent = "base_link"
        for name in ARM_JOINTS:
            j = joints[name]
            assert j.find("parent").get("link") == parent, f"unexpected URDF chain at {name}"
            parent = j.find("child").get("link")
            self.origins.append(_origin_T(j.find("origin")))
            axis = [float(v) for v in j.find("axis").get("xyz").split()]
            assert axis == [0.0, 0.0, 1.0], f"{name}: fk() assumes z-axis joints, got {axis}"
            lim.append([float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))])
        assert parent == "gripper_link"
        self.limits_deg = np.degrees(np.array(lim))            # URDF limits, URDF zero
        self.nominal_tip = _origin_T(joints["gripper_frame_joint"].find("origin"))[:3, 3]
        self.offsets_deg = np.zeros(5) if offsets_deg is None else np.asarray(offsets_deg, float)
        self.tip_offset = self.nominal_tip.copy() if tip_offset is None else np.asarray(tip_offset, float)

    # ------------------------------------------------------------------ forward
    def fk(self, q_deg, offsets_deg=None) -> np.ndarray:
        """T_base_gripper_link for lerobot-convention joint angles (degrees)."""
        off = self.offsets_deg if offsets_deg is None else offsets_deg
        q = np.radians(np.asarray(q_deg, float)[:5] + off)
        T = np.eye(4)
        for origin, angle in zip(self.origins, q):
            c, s = np.cos(angle), np.sin(angle)
            T = T @ origin
            # every SO-101 joint axis is the local z axis: rotate the x and y columns
            x, y = T[:, 0].copy(), T[:, 1].copy()
            T[:, 0], T[:, 1] = c * x + s * y, -s * x + c * y
        return T

    def tip(self, q_deg, offsets_deg=None, tip_offset=None) -> np.ndarray:
        """Press tip position in the base frame."""
        T = self.fk(q_deg, offsets_deg)
        return T[:3, :3] @ (self.tip_offset if tip_offset is None else tip_offset) + T[:3, 3]

    def tool_dir(self, q_deg) -> np.ndarray:
        return self.fk(q_deg)[:3, :3] @ TOOL_AXIS

    # ------------------------------------------------------------------ inverse
    def joint_bounds_deg(self, calibrated_range_deg=None) -> np.ndarray:
        """Limits in the lerobot convention. Optionally intersected with the range the
        servos were calibrated over ((5,2) array, degrees)."""
        b = self.limits_deg - self.offsets_deg[:, None]
        if calibrated_range_deg is not None:
            c = np.asarray(calibrated_range_deg, float)
            b = np.stack([np.maximum(b[:, 0], c[:, 0]), np.minimum(b[:, 1], c[:, 1])], axis=1)
        return b

    def ik(self, target, approach_dir, seed_deg=None, bounds_deg=None,
           pos_tol_mm: float = 0.3, max_tilt_deg: float = 30.0) -> IKResult:
        """Joint angles that put the press tip on `target` (base frame, metres) with the
        fingers pointing along `approach_dir` as closely as the 5 joints allow.

        Position is weighted far above orientation, so the tip lands on the target and
        the remaining freedom is spent on reducing tilt.
        """
        target = np.asarray(target, float)
        d = np.asarray(approach_dir, float)
        d = d / np.linalg.norm(d)
        bounds = self.joint_bounds_deg() if bounds_deg is None else np.asarray(bounds_deg, float)
        lo, hi = bounds[:, 0] + 0.5, bounds[:, 1] - 0.5

        # With the fingers vertical, wrist_roll moves neither the tip nor the tool axis: the
        # problem is redundant. A weak pull towards the seed picks one solution and keeps
        # consecutive solutions continuous (1 mm of position outweighs 30 deg of drift).
        def make_residual(ref):
            def residual(q):
                T = self.fk(q)
                p = T[:3, :3] @ self.tip_offset + T[:3, 3]
                return np.concatenate([(p - target) * 1e4, T[:3, :3] @ TOOL_AXIS - d, (q - ref) * 0.01])
            return residual

        seeds = []
        if seed_deg is not None:
            seeds.append(np.clip(np.asarray(seed_deg, float)[:5], lo, hi))
        pan = np.degrees(np.arctan2(-target[1], target[0] - self.origins[0][0, 3]))
        for lift, elbow, wrist in ((0, 0, 60), (-30, 40, 70), (30, -20, 80), (-60, 80, 60), (45, 20, 20)):
            seeds.append(np.clip(np.array([pan, lift, elbow, wrist, 0.0]) - self.offsets_deg, lo, hi))

        best = None
        for i, s in enumerate(seeds):
            residual = make_residual(s)
            sol = least_squares(residual, s, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=60)
            r = residual(sol.x)
            pos_err = float(np.linalg.norm(r[:3]) / 10.0)
            tilt = float(np.degrees(np.arccos(np.clip(self.tool_dir(sol.x) @ d, -1, 1))))
            res = IKResult(sol.x, pos_err, tilt, pos_err <= pos_tol_mm and tilt <= max_tilt_deg)
            # Prefer valid solutions; among them the one from the caller's seed (continuity),
            # otherwise the least tilted.
            key = (not res.ok, pos_err > pos_tol_mm, 0.0 if (i == 0 and seed_deg is not None and res.ok) else tilt)
            if best is None or key < best[0]:
                best = (key, res)
            if i == 0 and seed_deg is not None and res.ok:
                break
        return best[1]
