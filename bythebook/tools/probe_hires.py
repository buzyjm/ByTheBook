"""Find how a high-resolution video mode maps onto the calibrated 640x480 mode.

    python -m bythebook.tools.probe_hires            # all fixed cameras
    python -m bythebook.tools.probe_hires --size 1920 1080

The intrinsics in ~/rig_calib were measured at 640x480. Webcams produce their other
modes by cropping and scaling the same sensor image, so pixels of the two modes are
related by  u_hi = s * u_lo + tx,  v_hi = s * v_lo + ty.  This tool measures (s, tx, ty)
by matching image features between one frame of each mode (static scene, no board
needed) and stores the result in rig_state/hires.json. Nothing is changed in
~/rig_calib.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from .. import rig
from ..cameras import grab


def fit_mode_map(lo: np.ndarray, hi: np.ndarray):
    """-> (A 3x3, rms px in the hi-res image, inlier count, anisotropy)."""
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(lo, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(hi, cv2.COLOR_BGR2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if m.distance < 0.7 * n.distance]
    if len(good) < 30:
        raise RuntimeError(f"only {len(good)} feature matches: is the scene textured and static?")
    a = np.float64([k1[m.queryIdx].pt for m in good])
    b = np.float64([k2[m.trainIdx].pt for m in good])
    _, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    inl = mask.ravel().astype(bool)
    a, b = a[inl], b[inl]
    # independent x and y scales first, to check the mode really is an isotropic scale + crop
    sx, tx = np.polyfit(a[:, 0], b[:, 0], 1)
    sy, ty = np.polyfit(a[:, 1], b[:, 1], 1)
    s = float(np.mean([sx, sy]))
    tx, ty = float(np.mean(b[:, 0] - s * a[:, 0])), float(np.mean(b[:, 1] - s * a[:, 1]))
    A = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])
    rms = float(np.sqrt(np.mean(np.sum((a * s + [tx, ty] - b) ** 2, axis=1))))
    return A, rms, int(inl.sum()), float(sx / sy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roles", nargs="+", default=list(rig.FIXED_ROLES))
    ap.add_argument("--size", nargs=2, type=int, default=[1920, 1080])
    a = ap.parse_args()
    cams = rig.load_cameras()
    out = json.loads(rig.HIRES_FILE.read_text()) if rig.HIRES_FILE.exists() else {}
    for role in a.roles:
        cam = cams[role]
        lo = grab(cam, average=4)
        hi = grab(cam, size=tuple(a.size), average=4)
        A, rms, n, aniso = fit_mode_map(lo, hi)
        ok = rms < 1.5 and abs(aniso - 1) < 0.004
        print(f"{role:11s} {cam.size} -> {tuple(a.size)}: scale {A[0,0]:.4f} offset ({A[0,2]:.1f}, {A[1,2]:.1f}) px, "
              f"rms {rms:.2f} px over {n} matches, x/y scale ratio {aniso:.4f}  {'OK' if ok else 'REJECTED'}")
        if ok:
            out[role] = {"size": list(a.size), "A": A.tolist(), "rms_px": rms, "matches": n}
    rig.HIRES_FILE.parent.mkdir(parents=True, exist_ok=True)
    rig.HIRES_FILE.write_text(json.dumps(out, indent=2))
    print("saved", rig.HIRES_FILE)


if __name__ == "__main__":
    main()
