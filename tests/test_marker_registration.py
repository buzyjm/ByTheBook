import numpy as np

from bythebook.geometry import apply_T, rotvec_T
from bythebook.kinematics import Chain
from bythebook.targets import board_object_points
from bythebook.tools.marker_registration import fit_joint, fit_marker_registration, sample_poses


def test_recovers_base_and_offsets_from_board_observations():
    rng = np.random.default_rng(0)
    board = board_object_points("small")
    true_off = np.array([0, -6.0, -3.5, -7.0, 0])
    true = Chain(offsets_deg=true_off)
    T_true = rotvec_T([0.01, -0.02, 0.05], [-0.288, -0.012, -0.028])
    T_gb = rotvec_T([1.2, 0.3, 0.2], [0.02, -0.01, -0.06])       # board hanging off the gripper
    from bythebook.registration import Registration
    rough_reg = Registration(rotvec_T([0, 0, 0.0], [-0.280, -0.005, -0.015]).tolist(), Chain().nominal_tip.tolist(),
                             [0, -6.0, -3.5, -7.0, 0], 3.0, 4.0, 5.0, 12)
    Q, obs = [], []
    for q_true in sample_poses(rough_reg, n_poses=40):
        T = T_true @ true.fk(q_true) @ T_gb
        ids = rng.choice(len(board), size=12, replace=False)
        obs.append({int(i): apply_T(T, board[i]) + rng.normal(0, 2e-3, 3) for i in ids})   # single-camera quality
        Q.append(q_true + rng.normal(0, 0.05, 5))
    rough = rotvec_T([0, 0, 0.0], [-0.280, -0.005, -0.015])        # a 4 mm-class touch registration
    res = fit_marker_registration(Q, obs, board, rough, sigmas=[3.0] * len(obs))
    assert res["rms_mm"] < 4.0          # 2 mm noise per axis gives a 3.5 mm 3-D rms
    assert res["loo_rms_mm"] < 4.0
    assert np.allclose(res["joint_offsets_deg"], true_off, atol=0.5)
    T = np.array(res["T_table_base"])
    assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) < 2e-3

    # What matters: where the model puts a point on the gripper at press-like poses near the table.
    fitted = Chain(offsets_deg=res["joint_offsets_deg"])
    probe = Chain().nominal_tip
    errs = []
    for x in np.linspace(0.14, 0.26, 4):
        for y in np.linspace(-0.1, 0.1, 4):
            r = true.ik([x, y, 0.03], [0, 0, -1.0])
            if r.ok:
                p_true = apply_T(T_true, true.tip(r.q_deg, tip_offset=probe))
                p_fit = apply_T(T, fitted.tip(r.q_deg, tip_offset=probe))
                errs.append(np.linalg.norm(p_true - p_fit) * 1000)
    assert np.sqrt(np.mean(np.square(errs))) < 2.0, errs

    # tip offset from a few noisy hand touches, with the base fixed
    true_tip = Chain().nominal_tip + [0.012, 0.006, 0.004]
    touches = []
    for _ in range(12):
        target = np.array([-0.288 + rng.uniform(0.15, 0.27), rng.uniform(-0.1, 0.1), 0.0])    # table frame
        r = Chain(offsets_deg=true_off, tip_offset=true_tip).ik(apply_T(np.linalg.inv(T_true), target), [0, 0, -1.0])
        if r.ok:
            touches.append({"q_deg": r.q_deg.tolist(), "P_table": (target + rng.normal(0, 2e-3, 3)).tolist()})
    res2 = fit_joint(Q, obs, board, rough, touches, sigmas=[3.0] * len(obs))
    assert np.linalg.norm(np.array(res2["tip_offset"]) - true_tip) < 2.5e-3
    assert res2["touch_loo_rms_mm"] < 4.0
