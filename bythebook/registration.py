"""Where is the arm in the table frame, and where exactly is the press tip?

The cameras are calibrated in the table frame, the arm lives in its own base frame.
This module fits, from N "touches" (the closed gripper tip placed on a point whose
table coordinates are known, joint angles recorded):

    T_table_base        6 parameters
    tip offset          3 parameters (this rig has custom fingertips, so the URDF tip is only a prior)
    joint zero offsets  3 parameters (shoulder_lift, elbow_flex, wrist_flex)

The shoulder_pan offset is the same thing as rotating the base, and the wrist_roll
offset is the same thing as rotating the tip offset, so neither is fitted.

    python -m bythebook.registration collect     # on site, arm limp, guided by prompts
    python -m bythebook.registration solve       # refit from the saved touches
    python -m bythebook.registration verify      # project the predicted tip into the cameras
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from . import rig
from .geometry import apply_T, rigid_fit, rotvec_T
from .kinematics import Chain

FITTED_JOINTS = (1, 2, 3)          # indices into ARM_JOINTS
TOUCHES_FILE = rig.STATE_DIR / "registration_touches.json"


@dataclass
class Registration:
    T_table_base: list
    tip_offset: list
    joint_offsets_deg: list
    rms_mm: float
    loo_rms_mm: float | None
    max_mm: float
    n_points: int
    gripper_closed: float | None = None     # gripper reading while touching (0-100)
    landing_bias: list | None = None        # constant correction added to every press target (table frame, m)
    landing_samples: list | None = None     # [{"P": [x, y, z], "bias": [bx, by]}, ...] measured corrections (m)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def chain(self) -> Chain:
        return Chain(offsets_deg=self.joint_offsets_deg, tip_offset=self.tip_offset)

    @property
    def T(self) -> np.ndarray:
        return np.array(self.T_table_base, float)

    def save(self, path=rig.REGISTRATION_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path=rig.REGISTRATION_FILE) -> "Registration":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `python -m bythebook.registration collect` on site first")
        return Registration(**json.loads(path.read_text()))


def _unpack(x, chain, fit_tip, fit_offsets):
    T = rotvec_T(x[:3], x[3:6])
    i = 6
    tip = chain.nominal_tip.copy()
    if fit_tip:
        tip = x[i:i + 3]
        i += 3
    off = np.zeros(5)
    if fit_offsets:
        off[list(FITTED_JOINTS)] = x[i:i + len(FITTED_JOINTS)]
    return T, tip, off


def _fit(Q, P, chain, fit_tip, fit_offsets, sigma_mm, tip_prior_mm, offset_prior_deg):
    # Start from a plain rigid fit with the nominal model.
    T0 = rigid_fit(np.array([chain.tip(q, np.zeros(5), chain.nominal_tip) for q in Q]), P)
    x0 = [*Rotation.from_matrix(T0[:3, :3]).as_rotvec(), *T0[:3, 3]]
    if fit_tip:
        x0 += list(chain.nominal_tip)
    if fit_offsets:
        x0 += [0.0] * len(FITTED_JOINTS)

    def residual(x):
        T, tip, off = _unpack(x, chain, fit_tip, fit_offsets)
        pred = apply_T(T, np.array([chain.tip(q, off, tip) for q in Q]))
        r = [((pred - P) * 1000.0 / sigma_mm).ravel()]
        if fit_tip:
            r.append((tip - chain.nominal_tip) * 1000.0 / tip_prior_mm)
        if fit_offsets:
            r.append(off[list(FITTED_JOINTS)] / offset_prior_deg)
        return np.concatenate(r)

    sol = least_squares(residual, np.array(x0, float), x_scale="jac")
    T, tip, off = _unpack(sol.x, chain, fit_tip, fit_offsets)
    err_mm = np.linalg.norm(apply_T(T, np.array([chain.tip(q, off, tip) for q in Q])) - P, axis=1) * 1000.0
    return T, tip, off, err_mm


def solve_registration(Q_deg, P_table, fit_tip=True, fit_offsets=True, leave_one_out=True,
                       sigma_mm=1.5, tip_prior_mm=15.0, offset_prior_deg=5.0) -> Registration:
    """Fit the registration. Q_deg (N,5) joint angles, P_table (N,3) touched points (metres).

    The priors keep the extra parameters near their nominal values unless the data say
    otherwise, which matters because touches are usually almost coplanar (on the table).
    Fewer than 6 touches: only the rigid transform is fitted. Fewer than 10: no joint offsets.
    """
    Q, P = np.asarray(Q_deg, float)[:, :5], np.asarray(P_table, float)
    n = len(Q)
    if n < 4:
        raise ValueError("need at least 4 touches")
    fit_tip = fit_tip and n >= 6
    fit_offsets = fit_offsets and n >= 10
    chain = Chain()
    args = (fit_tip, fit_offsets, sigma_mm, tip_prior_mm, offset_prior_deg)
    T, tip, off, err = _fit(Q, P, chain, *args)

    loo = None
    if leave_one_out and n >= 6:
        held = []
        for i in range(n):
            keep = np.arange(n) != i
            Ti, tipi, offi, _ = _fit(Q[keep], P[keep], chain, fit_tip and keep.sum() >= 6,
                                     fit_offsets and keep.sum() >= 10, sigma_mm, tip_prior_mm, offset_prior_deg)
            held.append(np.linalg.norm(apply_T(Ti, chain.tip(Q[i], offi, tipi)) - P[i]) * 1000.0)
        loo = float(np.sqrt(np.mean(np.square(held))))
    return Registration(T.tolist(), tip.tolist(), off.tolist(), float(np.sqrt(np.mean(err ** 2))), loo,
                        float(err.max()), n)


# ----------------------------------------------------------------------------- on-site tools
def _load_touches():
    return json.loads(TOUCHES_FILE.read_text()) if TOUCHES_FILE.exists() else []


def print_report(reg: Registration):
    t = np.array(reg.T_table_base)[:3, 3] * 1000
    print(f"\nregistration from {reg.n_points} touches")
    print(f"  fit residual      rms {reg.rms_mm:.1f} mm, worst {reg.max_mm:.1f} mm")
    if reg.loo_rms_mm is not None:
        print(f"  leave-one-out     rms {reg.loo_rms_mm:.1f} mm   (the honest number; aim for under 3 mm)")
    print(f"  base in table     x {t[0]:.0f}  y {t[1]:.0f}  z {t[2]:.0f} mm")
    print(f"  tip offset        {np.round(np.array(reg.tip_offset) * 1000, 1)} mm (gripper_link frame)")
    print(f"  joint offsets     {np.round(reg.joint_offsets_deg, 2)} deg")


def cmd_collect(a):
    from .arm import RealArm
    from .targets import find_board_points

    print("Step 1. Keep the arm OUT of the camera views. Locating the calibration board ...")
    pts, overlay = find_board_points(save_overlay=True)
    print(f"  {len(pts)} corner points measured by the cameras; numbered picture: {overlay}")
    for k, p in sorted(pts.items()):
        print(f"    point {k:3d}: x {p[0]*1000:7.1f}  y {p[1]*1000:7.1f}  z {p[2]*1000:5.1f} mm")

    arm = RealArm(torque=False)
    arm.connect()
    touches = [] if a.fresh else _load_touches()
    print("\nStep 2. The arm is limp. Close the gripper by hand, then for each point: hold the tip\n"
          "of the closed gripper on the corner, type the point number and ENTER. Use at least 10\n"
          "points, and vary the wrist angle between touches (steep, shallow). Empty line to finish.")
    try:
        while True:
            s = input(f"[{len(touches)} recorded] point number: ").strip()
            if not s:
                break
            if not s.isdigit() or int(s) not in pts:
                print("  unknown point")
                continue
            reads = [arm.read() for _ in range(10)]
            q = np.mean([r[0] for r in reads], axis=0)
            touches.append({"point": int(s), "P_table": list(map(float, pts[int(s)])), "q_deg": q.tolist(),
                            "gripper": float(np.mean([r[1] for r in reads]))})
            TOUCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
            TOUCHES_FILE.write_text(json.dumps(touches, indent=1))
    finally:
        arm.disconnect()
    cmd_solve(a)


def cmd_solve(a):
    touches = _load_touches()
    reg = solve_registration([t["q_deg"] for t in touches], [t["P_table"] for t in touches])
    grips = [t["gripper"] for t in touches if "gripper" in t]
    reg.gripper_closed = float(np.median(grips)) if grips else None
    print_report(reg)
    reg.save()
    print(f"saved {rig.REGISTRATION_FILE}")


def cmd_verify(a):
    """Read the joints (no torque change) and draw where the model says the tip is."""
    import cv2

    from .arm import RealArm
    from .cameras import grab
    from .geometry import project

    reg = Registration.load()
    arm = RealArm(torque=None)
    arm.connect()
    q, _ = arm.read()
    arm.disconnect()
    P = apply_T(reg.T, reg.chain().tip(q))
    print("tip in table frame (mm):", np.round(P * 1000, 1))
    out = rig.RESULTS_DIR / "registration_verify"
    out.mkdir(parents=True, exist_ok=True)
    for role, cam in rig.load_cameras(hires=True).items():
        img = grab(cam)
        u, v = project(cam, P)[0]
        cv2.drawMarker(img, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
        cv2.imwrite(str(out / f"{role}.jpg"), img)
    print(f"overlays in {out}: the red cross should sit on the gripper tip in every view")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--fresh", action="store_true", help="discard touches recorded earlier")
    c.set_defaults(fn=cmd_collect)
    sub.add_parser("solve").set_defaults(fn=cmd_solve)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
