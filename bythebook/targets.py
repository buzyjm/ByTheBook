"""Known points on the table, measured by the calibrated cameras.

Any rig-calibrate ChArUco board lying anywhere in view provides identifiable points
(its chessboard corners). Their table coordinates are triangulated from the cameras,
so the board does not have to be on its calibration mark. This is what the base
registration touches, and it doubles as a health check of the camera extrinsics:
if the cameras still agree with each other, the reprojection errors are small and
the points lie on the table plane.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import rig
from .cameras import grab_all
from .geometry import project, triangulate


def detect_board(img: np.ndarray, names=None):
    """Find the rig-calibrate board visible in `img`. -> (board name, {corner id: (u, v)}) or (None, {})."""
    ar = cv2.aruco
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    best = (None, {})
    for name in names or rig.BOARDS:
        spec = rig.BOARDS[name]
        d = ar.getPredefinedDictionary(getattr(ar, spec["dictionary"]))
        n_markers = (spec["squares"][0] * spec["squares"][1]) // 2
        ids = np.arange(spec["first_id"], spec["first_id"] + n_markers, dtype=np.int32)
        board = ar.CharucoBoard(tuple(spec["squares"]), spec["square_mm"] / 1000, spec["marker_mm"] / 1000, d, ids)
        board.setLegacyPattern(False)
        dp = ar.DetectorParameters()
        dp.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
        cc, ci, _, _ = ar.CharucoDetector(board, ar.CharucoParameters(), dp).detectBoard(gray)
        if ci is not None and len(ci) > len(best[1]):
            best = (name, {int(i): tuple(map(float, p)) for i, p in zip(ci.ravel(), cc.reshape(-1, 2))})
    return best


def measure_board(cams: dict, images: dict):
    """Triangulate every board corner seen by two or more cameras.

    -> (points {id: xyz table (m)}, report dict with per-camera reprojection errors and
        the height of the points above the table plane)
    """
    seen, board_name = {}, None
    for role, img in images.items():
        name, corners = detect_board(img)
        if name:
            board_name = board_name or name
            if name == board_name:
                seen[role] = corners
    ids = sorted({i for c in seen.values() for i in c})
    points, errs = {}, {r: [] for r in seen}
    for i in ids:
        roles = [r for r in seen if i in seen[r]]
        if len(roles) < 2:
            continue
        P, e = triangulate([cams[r] for r in roles], [seen[r][i] for r in roles])
        points[i] = P
        for r, ei in zip(roles, e):
            errs[r].append(float(ei))
    report = {
        "board": board_name,
        "corners_per_camera": {r: len(c) for r, c in seen.items()},
        "triangulated": len(points),
        "reproj_rms_px": {r: float(np.sqrt(np.mean(np.square(v)))) for r, v in errs.items() if v},
        "reproj_max_px": {r: float(np.max(v)) for r, v in errs.items() if v},
    }
    if points:
        z = np.array([p[2] for p in points.values()]) * 1000
        report["z_mm"] = {"mean": float(z.mean()), "min": float(z.min()), "max": float(z.max())}
        if board_name and len(points) >= 4:
            report["scale_error_pct"] = _scale_error(points, rig.BOARDS[board_name])
    return points, report, seen


def _scale_error(points, spec) -> float:
    """Measured vs printed distance between neighbouring corners, in percent."""
    nx = spec["squares"][0] - 1
    d = []
    for i, p in points.items():
        for j in (i + 1, i + nx):
            if j in points and not (j == i + 1 and (i + 1) % nx == 0):
                d.append(np.linalg.norm(points[j] - p) * 1000)
    return float((np.mean(d) / spec["square_mm"] - 1) * 100) if d else float("nan")


def find_board_points(save_overlay: bool = False, hires: bool = True):
    """Capture, measure, and optionally save a numbered picture. -> ({id: xyz}, overlay path or None)"""
    cams = {r: c for r, c in rig.load_cameras(hires=hires).items() if c.T_table_cam is not None}
    images = grab_all(cams, average=3)
    points, report, seen = measure_board(cams, images)
    if len(points) < 6:
        raise RuntimeError(f"board not measured well enough: {report}")
    overlay = None
    if save_overlay:
        out = rig.RESULTS_DIR / "board_points"
        out.mkdir(parents=True, exist_ok=True)
        for role, img in images.items():
            for i, P in points.items():
                u, v = project(cams[role], P)[0]
                cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), 2)
                cv2.putText(img, str(i), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(str(out / f"{role}.jpg"), img)
        overlay = out
    return points, overlay


def board_object_points(name: str) -> np.ndarray:
    """Chessboard corners of a rig-calibrate board in its own frame, metres, (n, 3), indexed by corner id."""
    ar = cv2.aruco
    spec = rig.BOARDS[name]
    d = ar.getPredefinedDictionary(getattr(ar, spec["dictionary"]))
    n_markers = (spec["squares"][0] * spec["squares"][1]) // 2
    ids = np.arange(spec["first_id"], spec["first_id"] + n_markers, dtype=np.int32)
    board = ar.CharucoBoard(tuple(spec["squares"]), spec["square_mm"] / 1000, spec["marker_mm"] / 1000, d, ids)
    board.setLegacyPattern(False)
    return np.asarray(board.getChessboardCorners(), float)
