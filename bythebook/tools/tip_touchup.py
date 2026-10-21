"""Measure the press tip with the robot and the cameras, after the marker registration.

    python -m bythebook.tools.tip_touchup run --execute      # press corners, click the tip, fit
    python -m bythebook.tools.tip_touchup fit                # refit from saved clicks

Hand touches locate the fingertip to 5 mm at best. Here the arm itself presses down on
corners of the calibration board lying on the table and holds there while the cameras
take pictures. You click the point where the fingertip meets the board in two or more
views; the clicks are triangulated into a 3-D tip position for that joint configuration.
Four to six corners give the tip offset to about 1 mm, and the distance between where
the tip landed and the corner it aimed at is a direct measurement of the whole chain's
accuracy before the fix.

Needs: base_registration.json from the marker registration, the small board removed from
the gripper, the gripper closed as for pressing, a display for the click windows.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from .. import rig
from ..arm import LOAD_JOINTS, Limits, MotionError, RealArm, Session
from ..cameras import grab
from ..geometry import apply_T, project, ray_plane_z
from ..kinematics import Chain
from ..registration import Registration, print_report
from ..targets import find_board_points

CLICKS_FILE = rig.STATE_DIR / "tip_touchup.json"


def ensure_display():
    """Terminals opened by the desktop app have no DISPLAY; point at the running X session."""
    import glob
    import os

    if not os.environ.get("DISPLAY"):
        socks = sorted(glob.glob("/tmp/.X11-unix/X*"))
        if not socks:
            raise RuntimeError("no X display found: run this from a terminal on the robot PC's desktop")
        os.environ["DISPLAY"] = ":" + socks[0].rsplit("X", 1)[1]
    if not os.environ.get("XAUTHORITY"):
        for cand in (os.path.expanduser("~/.Xauthority"), f"/run/user/{os.getuid()}/gdm/Xauthority"):
            if os.path.exists(cand):
                os.environ["XAUTHORITY"] = cand
                break


def click_point(img, title: str):
    """Show `img`, return the clicked pixel or None (right click / close = skip)."""
    ensure_display()
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title + "\nLEFT click: fingertip contact point. RIGHT click or close: skip this view")
    ax.set_axis_off()
    got = plt.ginput(1, timeout=0, mouse_add=1, mouse_pop=None, mouse_stop=3)
    plt.close(fig)
    return None if not got else (float(got[0][0]), float(got[0][1]))


def observe_tip(cams, images, P_pred, z_surface, name):
    """Crops around the predicted tip in every camera; the user clicks the contact point.

    The tip rests on the board, so its height is the board's height: each click is
    intersected with that plane (no depth from oblique views), and the views are averaged.
    -> (P_tip or None, clicks {role: (u, v)}, spread between views in mm)
    """
    clicks, pts = {}, []
    for role, img in images.items():
        u, v = project(cams[role], P_pred)[0]
        r = 160
        x0, y0 = int(max(0, u - r)), int(max(0, v - r))
        crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        pt = click_point(crop, f"{name}: {role}  (click the centre of the fingertip's contact patch)")
        if pt is None:
            continue
        uv = (pt[0] / 3 + x0, pt[1] / 3 + y0)
        clicks[role] = uv
        pts.append(ray_plane_z(cams[role], uv, z_surface)[0])
    if not pts:
        return None, clicks, None
    pts = np.array(pts)
    spread = float(np.max(np.linalg.norm(pts - pts.mean(0), axis=1)) * 1000) if len(pts) > 1 else 0.0
    return pts.mean(0), clicks, spread


def board_axes(pts, k, n_cols=3):
    """Unit directions along the row and along the column of the board at corner k, and the
    corner ids that name them. Corner ids run along rows (0,1,2 / 3,4,5 / ...)."""
    P = {i: np.array(p) for i, p in pts.items()}
    if (k + 1) in P and (k + 1) % n_cols != 0:
        row_ref = k + 1
    else:
        row_ref = k - 1
    col_ref = k + n_cols if (k + n_cols) in P else k - n_cols
    row = P[row_ref] - P[k]
    col = P[col_ref] - P[k]
    return row / np.linalg.norm(row), col / np.linalg.norm(col), row_ref, col_ref


def measured_offset(pts, k):
    """Ask for ruler measurements of the tip relative to corner k. -> P_tip or None."""
    row, col, row_ref, col_ref = board_axes(pts, k)
    print(f"  Look from straight above. Where is the centre of the fingertip's contact patch relative to corner {k}?")
    try:
        a = float(input(f"    along the row, towards corner {row_ref} (mm, negative = away from it): ").strip() or "nan")
        b = float(input(f"    along the column, towards corner {col_ref} (mm, negative = away from it): ").strip() or "nan")
    except ValueError:
        return None
    if np.isnan(a) or np.isnan(b):
        return None
    return np.array(pts[k]) + a / 1000 * row + b / 1000 * col


def descend_and_hold(mover, P, hover_mm=25.0, max_overtravel_mm=6.0):
    """Like Mover.press, but stays down at contact (or at the overtravel limit) and returns."""
    mover.hover(P)
    mover.move_linear(P + [0, 0, hover_mm / 1000], speed_mm_s=30.0, step_mm=1.0)
    time.sleep(0.3)
    history, hits = [], 0

    def on_step(Pc):
        nonlocal hits
        time.sleep(0.03)
        e = mover.arm.read_effort()[list(LOAD_JOINTS)]
        q_meas, _ = mover.arm.read()
        stall = float((mover.tip_table(q_meas)[2] - Pc[2]) * 1000)
        base = np.median(history[:-3][-10:], axis=0) if len(history) >= 6 else None
        history.append(e)
        dev = 0.0 if base is None else float(np.linalg.norm(e - base))
        hits = hits + 1 if (stall > mover.lim.stall_mm and dev > mover.lim.contact_threshold) else 0
        return hits >= mover.lim.contact_steps

    mover.move_linear(P - [0, 0, max_overtravel_mm / 1000], on_step=on_step)
    time.sleep(0.5)


def cmd_run(a):
    reg = Registration.load()
    cams = {r: c for r, c in rig.load_cameras(hires=True).items() if c.T_table_cam is not None}
    print("Locating the board on the table (keep the arm out of view) ...")
    pts, overlay = find_board_points(save_overlay=True)
    # corners nearest the base first, spread in y
    base_xy = reg.T[:3, 3][:2]
    order = sorted(pts, key=lambda k: np.linalg.norm(np.array(pts[k][:2]) - base_xy))
    near = order[: max(4, len(order) // 2)]
    chosen = sorted(near, key=lambda k: pts[k][1])
    chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, min(a.corners, len(chosen))).round().astype(int)]
    print(f"corners to press: {chosen}  (numbered picture: {overlay})")
    if not a.execute:
        print("dry run: add --execute to press them")
        return
    what = "you measure its offset with a ruler" if a.ruler else "you click in the camera pictures"
    if input(f"The arm will press each corner and hold while {what}. "
             "Small board removed from the gripper? Workspace clear? Type 'go': ").strip() != "go":
        return

    records = [] if a.fresh else (json.loads(CLICKS_FILE.read_text()) if CLICKS_FILE.exists() else [])
    chain = reg.chain()
    with Session(RealArm(torque=True), reg, Limits()) as mover:
        for k in chosen:
            P = np.array(pts[k])
            try:
                descend_and_hold(mover, P)
            except MotionError as e:
                print(f"  corner {k}: skipped ({e})")
                continue
            q_meas = np.mean([mover.arm.read()[0] for _ in range(5)], axis=0)
            P_model = apply_T(reg.T, chain.tip(q_meas))
            if a.ruler:
                P_tip, clicks, spread = measured_offset(pts, k), {}, 0.0
            else:
                images = {r: grab(c) for r, c in cams.items()}
                P_tip, clicks, spread = observe_tip(cams, images, P_model, float(P[2]), f"corner {k}")
            if P_tip is None:
                print(f"  corner {k}: no measurement, skipped")
            else:
                miss = (P_tip - P) * 1000
                print(f"  corner {k}: tip landed {np.round(miss[:2], 1)} mm from the corner (|{np.linalg.norm(miss[:2]):.1f}|); "
                      f"model error {np.round((P_model - P_tip) * 1000, 1)} mm; "
                      f"{len(clicks)} views, spread {spread:.1f} mm")
                records.append({"corner": int(k), "P_corner": P.tolist(), "P_tip": P_tip.tolist(),
                                "q_deg": q_meas.tolist(), "clicks": clicks, "spread_mm": spread})
                CLICKS_FILE.parent.mkdir(parents=True, exist_ok=True)
                CLICKS_FILE.write_text(json.dumps(records, indent=1))
            mover.move_linear(P + [0, 0, 0.03], speed_mm_s=30.0, step_mm=1.0)
    cmd_fit(a)


def cmd_fit(a):
    reg = Registration.load()
    records = json.loads(CLICKS_FILE.read_text())
    if len(records) < 3:
        raise SystemExit(f"only {len(records)} tip observations; need at least 3")
    chain = Chain(offsets_deg=reg.joint_offsets_deg)
    Q = np.array([r["q_deg"] for r in records])[:, :5]
    P = np.array([r["P_tip"] for r in records])

    def residual(tip):
        return ((apply_T(reg.T, np.array([chain.tip(q, tip_offset=tip) for q in Q])) - P) * 1000).ravel()

    tip = least_squares(residual, np.array(reg.tip_offset)).x
    err = np.linalg.norm(apply_T(reg.T, np.array([chain.tip(q, tip_offset=tip) for q in Q])) - P, axis=1) * 1000
    loo = []
    for i in range(len(Q)):
        keep = np.arange(len(Q)) != i
        ti = least_squares(lambda t: ((apply_T(reg.T, np.array([chain.tip(q, tip_offset=t) for q in Q[keep]])) - P[keep]) * 1000).ravel(),
                           tip).x
        loo.append(np.linalg.norm(apply_T(reg.T, chain.tip(Q[i], tip_offset=ti)) - P[i]) * 1000)
    print(f"\ntip offset from {len(Q)} camera-observed presses: residual rms {np.sqrt(np.mean(err ** 2)):.1f} mm, "
          f"worst {err.max():.1f} mm, leave-one-out rms {np.sqrt(np.mean(np.square(loo))):.1f} mm")
    print(f"  old tip offset {np.round(np.array(reg.tip_offset) * 1000, 1)} mm -> new {np.round(tip * 1000, 1)} mm")
    reg.tip_offset = tip.tolist()
    reg.rms_mm, reg.max_mm, reg.loo_rms_mm, reg.n_points = float(np.sqrt(np.mean(err ** 2))), float(err.max()), float(np.sqrt(np.mean(np.square(loo)))), len(Q)
    print_report(reg)
    reg.save()
    print(f"saved {rig.REGISTRATION_FILE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--execute", action="store_true")
    r.add_argument("--fresh", action="store_true")
    r.add_argument("--corners", type=int, default=5)
    r.add_argument("--ruler", action="store_true",
                   help="measure the tip's offset from the corner by eye with a ruler instead of clicking in pictures")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("fit").set_defaults(fn=cmd_fit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
