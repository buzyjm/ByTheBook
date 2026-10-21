"""Press a part you pick by its mark number: perception and pressing without the language model.

    python -m bythebook.tools.press_mark --prompt "kitchen scale"              # look, number the parts
    python -m bythebook.tools.press_mark --prompt "kitchen scale" --execute    # then press the mark you choose

Step 1 captures the fixed cameras, finds the appliance, numbers the candidate parts in
every view and writes results/press_mark/marks_<role>.jpg. Step 2 asks for the mark
number of the same part in each view (0 = not visible there), triangulates, draws the
target into targets_<role>.jpg, and presses it after confirmation.
"""
from __future__ import annotations

import argparse
import pickle
import time

import cv2
import numpy as np

from .. import rig
from ..arm import Limits, MotionError, RealArm, Session
from ..cameras import grab_all
from ..perception import Perception, draw_marks
from ..pipeline import LocateError, View, describe, draw_targets, locate

OUT = rig.RESULTS_DIR / "press_mark"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default="kitchen scale")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--surface-z-mm", type=float, help="height of the pressed surface, for a part seen by one camera")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--push-load", type=float,
                    help=f"load rise to push to after contact (default {Limits.press_load:.0f})")
    ap.add_argument("--push-depth-mm", type=float,
                    help=f"deepest push below the contact point (default {Limits.press_depth_mm:.0f} mm)")
    ap.add_argument("--measure-bias", action="store_true",
                    help="hold at the bottom of the press while you measure the landing offset with a ruler, "
                         "then store it as the landing correction for all future presses")
    ap.add_argument("--record", nargs="*", metavar="ROLE",
                    help="record video during the press; no argument = all fixed cameras at 640x480")
    ap.add_argument("--record-size", nargs=2, type=int, default=[640, 480])
    ap.add_argument("--reuse", action="store_true", help="use the pictures and marks of the previous run")
    ap.add_argument("--marks", help="mark per view, e.g. ego=3,side_left=7,side_right=0 (skips the prompt)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cams = {r: c for r, c in rig.load_cameras(hires=True).items() if c.T_table_cam is not None}
    session = OUT / "session.pkl"
    if a.reuse:
        with open(session, "rb") as f:
            images, views = pickle.load(f)
        print(f"reusing the capture from {time.ctime(session.stat().st_mtime)} (the scale must not have moved)")
    else:
        images = grab_all(cams)
        perception = Perception()
        views = {}
        for role, img in images.items():
            dets = perception.detect(img, a.prompt)
            if not dets:
                print(f"{role}: '{a.prompt}' not found")
                continue
            marks, crop, origin, scale = perception.segment_parts(img, dets[0]["box"])
            cv2.imwrite(str(OUT / f"marks_{role}.jpg"), draw_marks(crop, marks, origin, scale))
            views[role] = View(role, dets[0]["box"], crop, origin, scale, marks)
            print(f"{role}: found (score {dets[0]['score']:.2f}), {len(marks)} marks -> {OUT}/marks_{role}.jpg")
        with open(session, "wb") as f:
            pickle.dump((images, views), f)
    if len(views) < 1:
        return
    chosen = dict(kv.split("=") for kv in a.marks.split(",")) if a.marks else None
    if chosen is None:
        print("\nLook at the pictures. Give the mark number of the SAME part in each view (0 = not visible).")
    for role, v in views.items():
        s = chosen.get(role, "0") if chosen is not None else input(f"  {role}: ").strip()
        if s.isdigit() and int(s) > 0:
            v.matches["target"] = {"part": "target", "mark": int(s), "confidence": "high", "reason": "chosen"}
    z = None if a.surface_z_mm is None else a.surface_z_mm / 1000
    try:
        loc = locate("target", cams, views, surface_z=z)
    except LocateError as e:
        print(e)
        return
    P = np.array(loc.P_table)
    print(describe(loc))
    draw_targets(cams, images, {"target": loc}, OUT)
    print(f"red cross drawn in {OUT}/targets_*.jpg")
    if not a.execute:
        print("add --execute to press it")
        return
    from ..registration import Registration
    reg = Registration.load()
    lim = Limits()
    if a.threshold:
        lim.contact_threshold = a.threshold
    if a.push_load:
        lim.press_load = a.push_load
    if a.push_depth_mm:
        lim.press_depth_mm = a.push_depth_mm
    if input("Check the red cross. Press it? Type 'go': ").strip() != "go":
        return
    from ..recorder import Recorder
    rec = Recorder(a.record or list(cams), OUT, size=tuple(a.record_size)) if a.record is not None else None
    try:
        if rec:
            rec.start()
        with Session(RealArm(torque=True), reg, lim) as mover:
            if a.measure_bias:
                r = mover.press(P, hold=True)
                print(f"contact: {r.contact} at z = {r.contact_z_mm} mm (expected {r.expected_z_mm:.1f}), lowest {r.lowest_z_mm:.1f}")
                print("commanded z : stall mm : load deviation")
                print(" ".join(f"{z:.1f}:{st:.1f}:{d:.0f}" for z, st, d in r.trace))
                if not r.contact or r.contact_z_mm > r.expected_z_mm + 8:
                    print("The tip is probably NOT on the key: do not measure, press ENTER twice to lift.")
                print("\nThe tip is held on the target. Look from straight above. Where is the fingertip's contact")
                print("point relative to the centre of the key? Directions are in the TABLE frame: +x is away from")
                print("the arm's base (towards the pan), +y is towards the arm's left when it faces the scale.")
                try:
                    dx = float(input("  offset along +x, mm (negative = towards the base): ").strip() or "nan")
                    dy = float(input("  offset along +y, mm: ").strip() or "nan")
                except ValueError:
                    dx = dy = float("nan")
                mover.move_linear(mover.tip_table() + [0, 0, 0.03], speed_mm_s=30.0, step_mm=1.0)
                if dx == dx and dy == dy:
                    # the correction that would have centred this press: what was applied, minus the error seen
                    applied = mover.bias_at(P)
                    sample = {"P": P.tolist(), "bias": [float(applied[0] - dx / 1000), float(applied[1] - dy / 1000)]}
                    samples = [s_ for s_ in (reg.landing_samples or []) if np.linalg.norm(np.array(s_["P"][:2]) - P[:2]) > 0.02]
                    reg.landing_samples = samples + [sample]
                    reg.save()
                    print(f"landing correction at this key {np.round(np.array(sample['bias']) * 1000, 1)} mm; "
                          f"{len(reg.landing_samples)} sample(s) saved, interpolated between keys")
                return
            r = mover.press(P)
    except MotionError as e:
        print("motion refused:", e)
        return
    finally:
        if rec:
            files = rec.stop()
            print("videos:", ", ".join(f"{role}: {p.name} ({rec.frames.get(role, 0)} frames)" for role, p in files.items()))
    print(f"contact: {r.contact}  at z = {r.contact_z_mm} mm (expected {r.expected_z_mm:.1f}), lowest {r.lowest_z_mm:.1f} mm, "
          f"push load {r.push_load:.0f} (target {lim.press_load:.0f})")
    print("commanded z : stall mm : load deviation")
    print(" ".join(f"{z:.1f}:{st:.1f}:{d:.0f}" for z, st, d in r.trace))


if __name__ == "__main__":
    main()
