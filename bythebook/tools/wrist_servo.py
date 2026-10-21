"""Calibrate and test the wrist-camera correction (see bythebook/wrist.py).

    python -m bythebook.tools.wrist_servo calibrate --key X Y Z --execute
    python -m bythebook.tools.wrist_servo check --key X Y Z --execute

X Y Z is a key's position in table millimetres, as printed by press_mark or the pipeline
(a key that presses well, e.g. ON/OFF). Calibration, about five minutes:
  1. the arm hovers over the key; the wrist image is segmented and numbered, you type the
     number of the key (picture: results/wrist_servo/marks.jpg);
  2. the arm moves 10 mm in +x, -x, +y, -y and watches the key move in the image (Jacobian);
  3. the arm presses the key and holds; you measure with a ruler where the fingertip sits
     relative to the key centre (as for --measure-bias), which fixes the reference pixel.
`check` hovers over a key, prints the correction the wrist camera asks for, and presses with it.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from ..arm import Limits, RealArm, Session
from ..perception import Perception, draw_marks
from ..registration import Registration
from ..wrist import OUT_DIR, SERVO_FILE, KeyFinder, WristServo, gripper_yaw_deg, make_servo, wrist_image


def hover_and_settle(mover, P, hover_mm):
    mover.hover(P)
    above = P + [0, 0, hover_mm / 1000]
    mover.move_linear(above, speed_mm_s=30.0, step_mm=1.0)
    time.sleep(0.8)
    return above


def cmd_calibrate(a):
    reg = Registration.load()
    P = np.array(a.key) / 1000
    hover_mm = a.hover_mm
    perception = Perception()
    finder = KeyFinder(perception)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if input(f"The arm will hover over {np.round(P * 1000)} mm, make four 10 mm moves, then press the key "
             "and hold. Workspace clear? Type 'go': ").strip() != "go":
        return
    with Session(RealArm(torque=True), reg, Limits()) as mover:
        above = hover_and_settle(mover, P, hover_mm)
        img = wrist_image()
        # candidate parts in the wrist image, numbered, for the operator to pick the key
        box = [0, 0, img.shape[1], img.shape[0]]
        marks, crop, origin, scale = perception.segment_parts(img, box, margin=0.0, on_object_only=False, max_marks=40)
        cv2.imwrite(str(OUT_DIR / "marks.jpg"), draw_marks(crop, marks, origin, scale))
        s = input(f"  Which mark is the key in {OUT_DIR}/marks.jpg? ").strip()
        mark = next((k for k in marks if str(k.id) == s), None)
        if mark is None:
            print("  no such mark; aborting (the arm parks)")
            return
        u0, area = np.array(mark.center, float), mark.area_px
        print(f"  key at {np.round(u0)} px, area {area:.0f} px")
        yaw0 = gripper_yaw_deg(mover)

        shifts = {}
        for name, dv in (("+x", [0.01, 0, 0]), ("-x", [-0.01, 0, 0]), ("+y", [0, 0.01, 0]), ("-y", [0, -0.01, 0])):
            mover.move_linear(above + dv, speed_mm_s=15.0, step_mm=1.0)
            time.sleep(0.7)
            found = finder.find(wrist_image(), u0, area)
            mover.move_linear(above, speed_mm_s=15.0, step_mm=1.0)
            time.sleep(0.5)
            if found is None:
                print(f"  move {name}: key lost; aborting")
                return
            shifts[name] = np.array(found[0]) - u0
            print(f"  move {name}: key shifted {np.round(shifts[name], 1)} px")
        J0 = np.column_stack([(shifts["+x"] - shifts["-x"]) / 0.02, (shifts["+y"] - shifts["-y"]) / 0.02])
        print(f"  Jacobian (px per m):\n{np.round(J0)}")
        if abs(np.linalg.det(J0)) < 1e4:
            print("  the image barely moved; is the key really in view? aborting")
            return

        r = mover.press(P, hold=True)
        print(f"  pressed: contact {r.contact} at {r.contact_z_mm} mm, push load {r.push_load:.0f}")
        print("  Look from straight above: where is the fingertip's contact point relative to the key centre?")
        print("  +x towards the pan, +y towards the ON/OFF side; empty = 0.")
        ex = float(input("    offset along +x, mm: ").strip() or "0")
        ey = float(input("    offset along +y, mm: ").strip() or "0")
        mover.move_linear(mover.tip_table() + [0, 0, 0.03], speed_mm_s=30.0, step_mm=1.0)
        e = np.array([ex, ey]) / 1000
        u_ref = u0 - J0 @ e
        servo = WristServo(u_ref, J0, yaw0, hover_mm, float(area))
        servo.save()
        print(f"  reference pixel {np.round(u_ref)} (key seen at {np.round(u0)}, tip was {np.round(e * 1000, 1)} mm off)")
        print(f"saved {SERVO_FILE}")


def cmd_check(a):
    reg = Registration.load()
    servo = WristServo.load()
    finder = KeyFinder(Perception())
    P = np.array(a.key) / 1000
    if input(f"The arm will press {np.round(P * 1000)} mm with the wrist-camera correction. Type 'go': ").strip() != "go":
        return
    with Session(RealArm(torque=True), reg, Limits()) as mover:
        r = mover.press(P, servo=make_servo(servo, finder))
    print(f"contact {r.contact} at {r.contact_z_mm} mm, push load {r.push_load:.0f}, correction applied {r.servo_mm} mm")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("calibrate", cmd_calibrate), ("check", cmd_check)):
        sp = sub.add_parser(name)
        sp.add_argument("--key", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
        sp.add_argument("--hover-mm", type=float, default=25.0)
        sp.add_argument("--execute", action="store_true")
        sp.set_defaults(fn=fn)
    a = ap.parse_args()
    if not a.execute:
        print("add --execute to move the arm")
        return
    a.fn(a)


if __name__ == "__main__":
    main()
