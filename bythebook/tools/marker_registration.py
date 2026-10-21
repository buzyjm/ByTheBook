"""Precise base registration from a calibration board fixed to the gripper.

    python -m bythebook.tools.marker_registration run --execute   # move through poses, capture, fit
    python -m bythebook.tools.marker_registration run             # dry run: poses only, no motion
    python -m bythebook.tools.marker_registration fit             # refit from saved observations

Touching corners by hand gives 3 to 4 mm; a board rigidly taped to the gripper, seen by two
or more calibrated cameras, gives corner positions to about 1 mm, and the arm can show it in
dozens of poses. Fitted: the base pose in the table frame (6), the board pose on the gripper
(6, a nuisance) and the zero offsets of shoulder_lift, elbow_flex and wrist_flex (3). The
press-tip offset is then solved from the hand touches with everything else held fixed.

Needs a rough registration first (python -m bythebook.registration collect) to move safely.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .. import rig
from ..arm import Limits, MotionError, RealArm, Session
from ..geometry import apply_T, inv_T, make_T, rigid_fit, rotvec_T, triangulate
from ..kinematics import ARM_JOINTS, Chain
from ..registration import FITTED_JOINTS, TOUCHES_FILE, Registration, print_report
from ..targets import board_object_points, detect_board

OBS_FILE = rig.STATE_DIR / "marker_observations.json"
BOARD = "small"


# ----------------------------------------------------------------------------- fit
def fit_marker_registration(Q, obs, board_pts, T_init, sigma_mm=1.0, offset_prior_deg=8.0, sigmas=None):
    """Q: (K,5) joint readings; obs: list of {corner id: xyz table} per pose; board_pts: (n,3) board frame;
    sigmas: optional per-pose measurement sigma in mm (triangulated ~1, single-camera PnP ~3).

    -> dict(T_table_base, T_gripper_board, joint_offsets_deg, rms_mm, loo_rms_mm, n_poses, n_points)
    """
    chain = Chain()
    Q = np.asarray(Q, float)[:, :5]
    rows = [(k, i, P, sig) for k, (o, sig) in enumerate(zip(obs, sigmas or [sigma_mm] * len(obs))) for i, P in o.items()]
    pose_idx = np.array([r[0] for r in rows])
    b = np.array([board_pts[r[1]] for r in rows])
    P = np.array([r[2] for r in rows])
    S = np.array([r[3] for r in rows])

    # initial board pose on the gripper from the pose with most corners
    k0 = max(range(len(obs)), key=lambda k: len(obs[k]))
    ids0 = sorted(obs[k0])
    T_table_board0 = rigid_fit(board_pts[ids0], np.array([obs[k0][i] for i in ids0]))
    T_gb0 = inv_T(chain.fk(Q[k0])) @ inv_T(T_init) @ T_table_board0

    def unpack(x):
        T = rotvec_T(x[:3], x[3:6])
        Tgb = rotvec_T(x[6:9], x[9:12])
        off = np.zeros(5)
        off[list(FITTED_JOINTS)] = x[12:15]
        return T, Tgb, off

    def predict(T, Tgb, off, Qs, pidx, bpts):
        fks = [T @ chain.fk(q, off) @ Tgb for q in Qs]
        return np.array([fks[k][:3, :3] @ bp + fks[k][:3, 3] for k, bp in zip(pidx, bpts)])

    def residual(x, Qs, pidx, bpts, Ps, Ss):
        T, Tgb, off = unpack(x)
        r = ((predict(T, Tgb, off, Qs, pidx, bpts) - Ps) * 1000 / Ss[:, None]).ravel()
        return np.concatenate([r, x[12:15] / offset_prior_deg])

    x0 = np.concatenate([Rotation.from_matrix(T_init[:3, :3]).as_rotvec(), T_init[:3, 3],
                         Rotation.from_matrix(T_gb0[:3, :3]).as_rotvec(), T_gb0[:3, 3], np.zeros(3)])

    def solve(Qs, pidx, bpts, Ps, Ss):
        return least_squares(residual, x0, args=(Qs, pidx, bpts, Ps, Ss), x_scale="jac").x

    x = solve(Q, pose_idx, b, P, S)
    T, Tgb, off = unpack(x)
    err = np.linalg.norm(predict(T, Tgb, off, Q, pose_idx, b) - P, axis=1) * 1000

    loo = []
    for k in range(len(obs)):
        keep = pose_idx != k
        if keep.sum() < 30 or len(set(pose_idx[keep])) < 5:
            continue
        xk = solve(Q, pose_idx[keep], b[keep], P[keep], S[keep])
        Tk, Tgbk, offk = unpack(xk)
        e = np.linalg.norm(predict(Tk, Tgbk, offk, Q, pose_idx[~keep], b[~keep]) - P[~keep], axis=1) * 1000
        loo.extend(e.tolist())
    return {
        "T_table_base": T.tolist(), "T_gripper_board": Tgb.tolist(), "joint_offsets_deg": off.tolist(),
        "rms_mm": float(np.sqrt(np.mean(err ** 2))), "max_mm": float(err.max()),
        "loo_rms_mm": float(np.sqrt(np.mean(np.square(loo)))) if loo else None,
        "n_poses": len(obs), "n_points": len(rows),
    }


def fit_joint(Q, obs, board_pts, T_init, touches, sigmas=None, touch_sigma_mm=3.0, offset_prior_deg=8.0,
              tip_prior_mm=15.0):
    """Marker observations and hand touches in one fit, sharing the base pose and joint offsets.
    Unknowns: base pose (6), board pose on the gripper (6), joint offsets (3), tip offset (3).
    -> dict like fit_marker_registration plus tip_offset, touch_rms_mm, touch_max_mm, touch_loo_rms_mm."""
    chain = Chain()
    Q = np.asarray(Q, float)[:, :5]
    rows = [(k, i, P, sig) for k, (o, sig) in enumerate(zip(obs, sigmas or [1.0] * len(obs))) for i, P in o.items()]
    pidx = np.array([r[0] for r in rows])
    b = np.array([board_pts[r[1]] for r in rows])
    Pm = np.array([r[2] for r in rows])
    S = np.array([r[3] for r in rows])
    Qt = np.array([t["q_deg"] for t in touches])[:, :5]
    Pt = np.array([t["P_table"] for t in touches])

    k0 = max(range(len(obs)), key=lambda k: len(obs[k]))
    ids0 = sorted(obs[k0])
    T_gb0 = inv_T(chain.fk(Q[k0])) @ inv_T(T_init) @ rigid_fit(board_pts[ids0], np.array([obs[k0][i] for i in ids0]))

    def unpack(x):
        off = np.zeros(5)
        off[list(FITTED_JOINTS)] = x[12:15]
        return rotvec_T(x[:3], x[3:6]), rotvec_T(x[6:9], x[9:12]), off, x[15:18]

    def marker_pred(T, Tgb, off):
        fks = [T @ chain.fk(q, off) @ Tgb for q in Q]
        return np.array([fks[k][:3, :3] @ bp + fks[k][:3, 3] for k, bp in zip(pidx, b)])

    def touch_pred(T, off, tip, Qs):
        return apply_T(T, np.array([chain.tip(q, off, tip) for q in Qs]))

    def residual(x, tsel):
        T, Tgb, off, tip = unpack(x)
        r1 = ((marker_pred(T, Tgb, off) - Pm) * 1000 / S[:, None]).ravel()
        r2 = ((touch_pred(T, off, tip, Qt[tsel]) - Pt[tsel]) * 1000 / touch_sigma_mm).ravel()
        return np.concatenate([r1, r2, x[12:15] / offset_prior_deg, (tip - chain.nominal_tip) * 1000 / tip_prior_mm])

    x0 = np.concatenate([Rotation.from_matrix(T_init[:3, :3]).as_rotvec(), T_init[:3, 3],
                         Rotation.from_matrix(T_gb0[:3, :3]).as_rotvec(), T_gb0[:3, 3], np.zeros(3), chain.nominal_tip])
    all_t = np.ones(len(Qt), bool)
    x = least_squares(residual, x0, args=(all_t,), x_scale="jac").x
    T, Tgb, off, tip = unpack(x)
    em = np.linalg.norm(marker_pred(T, Tgb, off) - Pm, axis=1) * 1000
    et = np.linalg.norm(touch_pred(T, off, tip, Qt) - Pt, axis=1) * 1000
    loo = []
    for i in range(len(Qt)):
        sel = all_t.copy()
        sel[i] = False
        Ti, _, offi, tipi = unpack(least_squares(residual, x, args=(sel,), x_scale="jac").x)
        loo.append(np.linalg.norm(touch_pred(Ti, offi, tipi, Qt[i:i + 1])[0] - Pt[i]) * 1000)
    return {
        "T_table_base": T.tolist(), "T_gripper_board": Tgb.tolist(), "joint_offsets_deg": off.tolist(),
        "tip_offset": tip.tolist(), "rms_mm": float(np.sqrt(np.mean(em ** 2))), "max_mm": float(em.max()),
        "touch_rms_mm": float(np.sqrt(np.mean(et ** 2))), "touch_max_mm": float(et.max()),
        "touch_loo_rms_mm": float(np.sqrt(np.mean(np.square(loo)))), "n_poses": len(obs), "n_points": len(rows),
        "n_touches": len(Qt),
    }


# ----------------------------------------------------------------------------- poses
def sample_poses(rough, n_poses=40, seed=1, z_range=(0.09, 0.20), x_range=(0.10, 0.28), y_max=0.16,
                 max_tilt_deg=45.0, calibrated_range_deg=None):
    """Joint configurations spread over the whole workspace, in joint space on purpose: the
    fit needs shoulder, elbow and wrist angles to vary widely, otherwise the joint zero
    offsets and the board pose on the gripper cannot be told apart. Only the fingertip
    region is constrained (height above the table, inside the reach box, fingers roughly
    down so the board stays clear of the arm). Ordered greedily to shorten travel.
    """
    chain = rough.chain()
    bounds = chain.joint_bounds_deg(calibrated_range_deg)
    lo, hi = bounds[:, 0] + 2, bounds[:, 1] - 2
    lo[0], hi[0] = max(lo[0], -50), min(hi[0], 50)
    lo[4], hi[4] = max(lo[4], -85), min(hi[4], 85)
    down = np.linalg.inv(rough.T)[:3, :3] @ np.array([0, 0, -1.0])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(20000):
        q = rng.uniform(lo, hi)
        p_base = chain.tip(q)
        p_table = apply_T(rough.T, p_base)
        tilt = np.degrees(np.arccos(np.clip(chain.tool_dir(q) @ down, -1, 1)))
        if (z_range[0] <= p_table[2] <= z_range[1] and x_range[0] <= p_base[0] <= x_range[1]
                and abs(p_base[1]) <= y_max and tilt <= max_tilt_deg):
            out.append(q)
        if len(out) >= 4 * n_poses:
            break
    # farthest-point selection on the three pitch joints for diversity, then greedy ordering
    out = np.array(out)
    chosen = [0]
    while len(chosen) < min(n_poses, len(out)):
        d = np.min([np.linalg.norm(out[:, 1:4] - out[c, 1:4], axis=1) for c in chosen], axis=0)
        chosen.append(int(np.argmax(d)))
    sel = out[chosen]
    order, left = [0], list(range(1, len(sel)))
    while left:
        k = min(left, key=lambda j: np.max(np.abs(sel[j] - sel[order[-1]])))
        order.append(k)
        left.remove(k)
    return [sel[k] for k in order]


def board_pose_single_view(cam, corners, board_pts):
    """Board corners in the table frame from one camera, via PnP on the known board geometry."""
    import cv2

    ids = sorted(corners)
    obj = board_pts[ids].astype(np.float64)
    img = np.array([corners[i] for i in ids], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return {}, None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam.K, cam.dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    T_table_board = cam.T_table_cam @ make_T(cv2.Rodrigues(rvec)[0], tvec.ravel())
    return {i: apply_T(T_table_board, board_pts[i]).tolist() for i in ids}, err


def capture_observation(cams, grab_all, board_pts):
    """-> (points {id: xyz}, sigma_mm, corners per camera, worst agreement px)."""
    images = grab_all(cams, average=2)
    seen = {}
    for role, img in images.items():
        name, corners = detect_board(img, names=[BOARD])
        if name and len(corners) >= 6:
            seen[role] = corners
    ids = sorted({i for c in seen.values() for i in c})
    pts, worst = {}, 0.0
    for i in ids:
        roles = [r for r in seen if i in seen[r]]
        if len(roles) < 2:
            continue
        P, err = triangulate([cams[r] for r in roles], [seen[r][i] for r in roles])
        if err.max() <= 4.0:
            pts[i] = P.tolist()
            worst = max(worst, float(err.max()))
    counts = {r: len(c) for r, c in seen.items()}
    if len(pts) >= 4:
        return pts, 1.0, counts, worst
    if seen:   # one camera only: pose of the whole board from its known geometry, less precise in depth
        role = max(seen, key=lambda r: len(seen[r]))
        pts, err = board_pose_single_view(cams[role], seen[role], board_pts)
        if pts and err is not None and err < 1.5:
            return pts, 3.0, counts, err
    return {}, None, counts, 0.0


def cmd_run(a):
    from ..cameras import grab_all

    rough = Registration.load()
    cams = {r: c for r, c in rig.load_cameras(hires=True).items() if c.T_table_cam is not None}
    ro = RealArm(torque=None).connect()          # the servos' calibrated range bounds the poses
    rng_deg = ro.calibrated_range_deg()
    ro.disconnect()
    poses = sample_poses(rough, n_poses=a.poses, seed=a.seed, calibrated_range_deg=rng_deg)
    q_all = np.array(poses)
    print(f"{len(poses)} poses; joint ranges covered (deg): " + ", ".join(
        f"{n} {q_all[:, i].min():.0f}..{q_all[:, i].max():.0f}" for i, n in enumerate(ARM_JOINTS)))
    if not a.execute:
        print("dry run: add --execute to move the arm")
        return

    obs = json.loads(OBS_FILE.read_text()) if (OBS_FILE.exists() and not a.fresh) else []
    if input(f"The arm will visit {len(poses)} poses, fingertip 9 to 20 cm above the table, board on the "
             "gripper. Workspace clear? Type 'go': ").strip() != "go":
        return
    lim = Limits()
    lim.floor_z = 0.06               # the board hangs below the fingertip: keep well clear of the table
    board_pts = board_object_points(BOARD)
    with Session(RealArm(torque=True), rough, lim) as mover:
        mover.gripper = mover.arm.read()[1]          # leave the gripper as it is, the board is taped to it
        for n, q in enumerate(poses, 1):
            try:
                mover.move_joints(q)
            except MotionError as e:
                print(f"  pose {n}: skipped ({e})")
                continue
            time.sleep(0.8)
            q_meas = np.mean([mover.arm.read()[0] for _ in range(5)], axis=0)
            pts, sigma, counts, worst = capture_observation(cams, grab_all, board_pts)
            how = "none" if not pts else ("triangulated" if sigma == 1.0 else "single camera")
            print(f"  pose {n:2d}: corners per camera {counts}, {len(pts):2d} points, {how}, {worst:.1f} px")
            if pts:
                obs.append({"q_deg": q_meas.tolist(), "sigma_mm": sigma,
                            "points": {str(i): p for i, p in pts.items()}})
                OBS_FILE.parent.mkdir(parents=True, exist_ok=True)
                OBS_FILE.write_text(json.dumps(obs))
    print(f"{len(obs)} usable poses saved to {OBS_FILE}")
    cmd_fit(a)


def cmd_fit(a):
    obs = json.loads(OBS_FILE.read_text())
    rough = Registration.load()
    Q = [o["q_deg"] for o in obs]
    pts = [{int(i): np.array(p) for i, p in o["points"].items()} for o in obs]
    sigmas = [o.get("sigma_mm", 1.0) for o in obs]
    if len(obs) < 6:
        raise SystemExit(f"only {len(obs)} usable poses; need at least 6 (run again, observations accumulate)")
    q_all = np.array(Q)
    print(f"\n{len(obs)} marker poses; joint ranges covered (deg): " + ", ".join(
        f"{n} {q_all[:, i].min():.0f}..{q_all[:, i].max():.0f}" for i, n in enumerate(ARM_JOINTS)))
    marker_only = fit_marker_registration(Q, pts, board_object_points(BOARD), rough.T, sigmas=sigmas)
    print(f"marker-only fit: residual rms {marker_only['rms_mm']:.2f} mm, offsets {np.round(marker_only['joint_offsets_deg'], 1)}")
    touches = json.loads(TOUCHES_FILE.read_text())
    res = fit_joint(Q, pts, board_object_points(BOARD), rough.T, touches, sigmas=sigmas)
    print(f"joint fit with {res['n_touches']} touches: marker rms {res['rms_mm']:.2f} mm (worst {res['max_mm']:.1f}); "
          f"touch rms {res['touch_rms_mm']:.1f} mm (worst {res['touch_max_mm']:.1f}), "
          f"leave-one-touch-out rms {res['touch_loo_rms_mm']:.1f} mm")
    grips = [t["gripper"] for t in touches if "gripper" in t]
    reg = Registration(res["T_table_base"], res["tip_offset"], res["joint_offsets_deg"], res["touch_rms_mm"],
                       res["touch_loo_rms_mm"], res["touch_max_mm"], res["n_touches"],
                       gripper_closed=float(np.median(grips)) if grips else None)
    print_report(reg)
    reg.save()
    (rig.STATE_DIR / "marker_fit.json").write_text(json.dumps(res, indent=1))
    print(f"saved {rig.REGISTRATION_FILE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--execute", action="store_true")
    r.add_argument("--fresh", action="store_true", help="discard observations from an earlier run")
    r.add_argument("--poses", type=int, default=40)
    r.add_argument("--seed", type=int, default=1, help="change to get a different set of poses")
    r.set_defaults(fn=cmd_run)
    sub.add_parser("fit").set_defaults(fn=cmd_fit)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
