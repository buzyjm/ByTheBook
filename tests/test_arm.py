import numpy as np
import pytest

from bythebook.arm import Limits, MotionError, Mover, Session, SimArm
from bythebook.geometry import rotvec_T
from bythebook.kinematics import Chain
from bythebook.registration import Registration


def _reg():
    # base 25 cm behind the table origin, facing +x of the table, sitting on the table
    T = rotvec_T([0, 0, 0.1], [-0.25, 0.02, 0.0])
    return Registration(T.tolist(), Chain().nominal_tip.tolist(), [0, 2.0, -1.5, 1.0, 0], 1.0, 1.5, 2.0, 12,
                        gripper_closed=3.0)


def _button(reg, name="tare", top_mm=30.0):
    xy = (reg.T @ np.array([0.20, 0.03, 0, 1]))[:2]
    return {name: dict(xy=xy, radius_mm=6.0, travel_mm=1.0, z=top_mm / 1000)}


def test_press_detects_contact_and_presses_button():
    reg = _reg()
    buttons = _button(reg)
    arm = SimArm(reg, surface_z=0.027, buttons=buttons)        # scale body 27 mm, button 3 mm proud
    with Session(arm, reg, Limits(), sleep=lambda s: None) as m:
        q_start = m.q_park.copy()
        target = np.array([*buttons["tare"]["xy"], 0.030])
        r = m.press(target)
    assert r.contact
    assert abs(r.contact_z_mm - 30.0) < 1.0            # the tip stopped at the surface
    assert arm.presses == ["tare"]
    assert r.lowest_z_mm > 30.0 - 6.0 - 6.0            # never deeper than overtravel + push depth
    assert r.push_load >= 110
    assert np.allclose(arm.q, q_start)                 # parked where it started
    assert r.lowest_z_mm > 28.0                        # the measured tip never dug into the scale body


def test_press_reports_no_contact_when_surface_is_lower_than_expected():
    reg = _reg()
    arm = SimArm(reg, surface_z=0.0)
    with Session(arm, reg, Limits(), sleep=lambda s: None) as m:
        xy = (reg.T @ np.array([0.20, 0.0, 0, 1]))[:2]
        r = m.press([*xy, 0.030])
    assert not r.contact
    assert r.lowest_z_mm >= 30.0 - 6.0 - 0.6           # stopped at the overtravel limit (no contact: no push)


def test_target_outside_workspace_is_refused_before_moving():
    reg = _reg()
    arm = SimArm(reg)
    m = Mover(arm.connect(), reg, sleep=lambda s: None)
    far = (reg.T @ np.array([0.45, 0.0, 0.03, 1]))[:3]
    with pytest.raises(MotionError):
        m.press(far)
    assert arm.log == []                               # nothing was commanded


def test_free_moves_stay_above_floor():
    reg = _reg()
    arm = SimArm(reg, surface_z=0.0)
    with Session(arm, reg, sleep=lambda s: None) as m:
        for bx, by in ((0.14, -0.12), (0.26, 0.12), (0.18, 0.0)):
            P = (reg.T @ np.array([bx, by, 0.0, 1]))[:3]
            m.hover(P)
            assert abs(m.tip_table()[2] - m.lim.travel_z) < 1e-3
    assert min(p[2] for p in arm.log) > 0.014


def test_no_joint_jump_from_the_real_parked_pose():
    """Regression: leaving the folded rest pose used to request a far IK branch in one step."""
    reg = Registration(rotvec_T([0, 0, 0.0], [-0.288, -0.012, -0.028]).tolist(),
                       [0.0146, 0.0071, -0.0942], [0, -6.1, -3.5, -7.0, 0], 3.1, 4.2, 5.4, 12, gripper_closed=2.6)
    arm = SimArm(reg, surface_z=0.0)
    cmds = []
    orig = arm.command

    def spy(q, g=None):
        cmds.append(np.array(q, float))
        orig(q, g)

    arm.command = spy
    with Session(arm, reg, Limits(), sleep=lambda s: None) as m:
        m.press([0.0, 0.009, 0.0])
    q = np.vstack([m.q_park[None], np.array(cmds)])
    assert np.max(np.abs(np.diff(q, axis=0))) < 2.0     # degrees per command


def test_press_compensates_gravity_sag_and_detects_contact():
    """The encoders read the arm a few millimetres below the command; the press must still
    land on the surface and notice the contact."""
    reg = _reg()
    buttons = _button(reg)
    arm = SimArm(reg, surface_z=0.027, buttons=buttons, droop_deg=(0.0, 1.6, 0.4, 0.8, 0.0))
    with Session(arm, reg, Limits(), sleep=lambda s: None) as m:
        target = np.array([*buttons["tare"]["xy"], 0.030])
        sag = np.linalg.norm(m.tip_table(arm.q) - m.tip_table(arm.read()[0])) * 1000
        r = m.press(target)
    assert sag > 3.0                                   # the scenario really has sag
    assert r.contact
    assert abs(r.contact_z_mm - 30.0) < 1.0
    assert arm.presses == ["tare"]


def test_press_along_a_tilted_approach_still_finds_the_surface():
    reg = _reg()
    buttons = _button(reg)
    arm = SimArm(reg, surface_z=0.027, buttons=buttons, droop_deg=(0.0, 1.0, 0.3, 0.5, 0.0))
    tilt = np.radians(20.0)
    approach = np.array([-np.sin(tilt), 0.0, -np.cos(tilt)])      # leaning towards the base
    with Session(arm, reg, Limits(), sleep=lambda s: None) as m:
        r = m.press(np.array([*buttons["tare"]["xy"], 0.030]), approach=approach)
    assert r.contact
    assert abs(r.contact_z_mm - 30.0) < 1.5
    assert arm.presses == ["tare"]


def test_retreat_moves_back_at_travel_height_without_parking():
    reg = _reg()
    arm = SimArm(reg, surface_z=0.0)
    with Session(arm, reg, Limits(), sleep=lambda s: None) as m:
        P = (reg.T @ np.array([0.22, 0.0, 0.03, 1]))[:3]
        m.press(P)
        m.retreat(P)
        tip = m.tip_table(arm.read()[0])
        assert abs(tip[2] - m.lim.travel_z) < 2e-3
        back = (np.linalg.inv(reg.T) @ np.append(tip, 1))[:3]
        assert back[0] < 0.22 - 0.05                       # at least 5 cm nearer the base than the key
        assert not np.allclose(arm.read()[0], m.q_park)


def test_landing_bias_interpolates_between_measured_keys():
    reg = _reg()
    reg.landing_samples = [{"P": [-0.012, 0.0685, 0.0435], "bias": [0.002, 0.0]},
                           {"P": [-0.008, -0.0715, 0.0466], "bias": [-0.003, 0.0]}]
    m = Mover(SimArm(reg).connect(), reg, sleep=lambda s: None)
    assert np.allclose(m.bias_at([-0.012, 0.0685, 0.04])[:2], [0.002, 0.0], atol=1e-6)
    assert np.allclose(m.bias_at([-0.008, -0.0715, 0.04])[:2], [-0.003, 0.0], atol=1e-6)
    mid = m.bias_at([-0.010, -0.0015, 0.04])
    assert abs(mid[0] + 0.0005) < 3e-4 and abs(mid[1]) < 1e-6
