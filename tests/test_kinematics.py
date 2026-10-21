import numpy as np
import pytest

from bythebook.kinematics import ARM_JOINTS, TOOL_AXIS, Chain
from bythebook.rig import URDF


def test_tool_axis_matches_urdf_tip():
    c = Chain()
    v = c.nominal_tip / np.linalg.norm(c.nominal_tip)
    assert np.degrees(np.arccos(v @ TOOL_AXIS)) < 6.0


def test_fk_matches_placo():
    placo = pytest.importorskip("placo")
    robot = placo.RobotWrapper(str(URDF))
    c = Chain()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(c.limits_deg[:, 0], c.limits_deg[:, 1])
        for name, v in zip(ARM_JOINTS, q):
            robot.set_joint(name, np.radians(v))
        robot.update_kinematics()
        assert np.allclose(robot.get_T_world_frame("gripper_link"), c.fk(q), atol=1e-6)
        ref_tip = robot.get_T_world_frame("gripper_frame_link")[:3, 3]
        assert np.allclose(ref_tip, c.tip(q), atol=1e-6)


def test_ik_reaches_table_points_pointing_down():
    c = Chain()
    down = np.array([0, 0, -1.0])
    seed = None
    n_ok = 0
    # A grid on a plane 5 cm below the base origin height... the base sits on the table,
    # so table height is z = 0 in the base frame; press targets are 0 to 6 cm above it.
    for x in np.linspace(0.16, 0.30, 5):
        for y in np.linspace(-0.12, 0.12, 5):
            for z in (0.0, 0.03, 0.06):
                r = c.ik([x, y, z], down, seed_deg=seed)
                if r.ok:
                    n_ok += 1
                    seed = r.q_deg
                    assert np.linalg.norm(c.tip(r.q_deg) - [x, y, z]) < 0.4e-3
                    assert np.all(r.q_deg >= c.limits_deg[:, 0]) and np.all(r.q_deg <= c.limits_deg[:, 1])
    assert n_ok >= 60, n_ok   # of 75


def test_ik_respects_offsets():
    off = np.array([0.0, 3.0, -2.0, 4.0, 0.0])
    c = Chain(offsets_deg=off)
    r = c.ik([0.22, 0.05, 0.02], [0, 0, -1.0])
    assert r.ok
    assert np.allclose(Chain().tip(r.q_deg + off), [0.22, 0.05, 0.02], atol=0.4e-3)
