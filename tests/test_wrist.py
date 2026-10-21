import numpy as np

from bythebook.arm import Limits, Session, SimArm
from bythebook.geometry import rotvec_T
from bythebook.kinematics import Chain
from bythebook.registration import Registration
from bythebook.wrist import KeyFinder, WristServo


def test_correction_inverts_the_rotated_jacobian():
    J0 = np.array([[-4000.0, 300.0], [200.0, 3800.0]])            # px per metre at yaw0
    s = WristServo(np.array([600.0, 450.0]), J0, yaw0=10.0, hover_mm=25.0, key_area_px=900)
    # the gripper turned by 30 deg: the same table-frame move shows up rotated in the image
    yaw = 40.0
    a = np.radians(yaw - 10.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    true_offset = np.array([0.004, -0.003])                        # key is 4 mm +x, 3 mm -y from the tip
    # moving the arm by +offset would put the key at u_ref, so now it appears J*offset away on the other side
    u_key = s.u_ref - (J0 @ R.T) @ true_offset
    d = s.correction(u_key, yaw)
    assert np.allclose(d, true_offset, atol=1e-6)


def test_blob_finder_locates_a_dark_key_on_a_light_panel():
    img = np.full((720, 1280, 3), 200, np.uint8)
    import cv2
    cv2.circle(img, (640, 420), 22, (40, 40, 40), -1)
    c, area = KeyFinder(None).find(img, (660, 400), expect_area=np.pi * 22 ** 2)
    assert np.linalg.norm(c - [640, 420]) < 1.5
    assert 1300 < area < 1700


def test_press_applies_the_servo_correction_before_descending():
    T = rotvec_T([0, 0, 0.0], [-0.22, 0.0, 0.0])
    reg = Registration(T.tolist(), Chain().nominal_tip.tolist(), [0] * 5, 1.0, 1.5, 2.0, 12, gripper_closed=3.0)
    key = (T @ np.array([0.20, 0.02, 0.03, 1]))[:3]
    arm = SimArm(reg, surface_z=0.027, buttons={"k": dict(xy=key[:2], radius_mm=5, travel_mm=1.0, z=0.03)})
    aimed = key + np.array([0.007, -0.004, 0.0])                   # 8 mm off: would miss a 5 mm key
    calls = []

    def servo(mover, P):
        calls.append(P.copy())
        return key[:2] - P[:2]                                      # what a wrist camera would measure

    with Session(arm, reg, Limits(), sleep=lambda s: None) as m:
        r = m.press(aimed, servo=servo)
    assert len(calls) == 1
    assert np.allclose(r.servo_mm, [-7.0, 4.0], atol=0.2)
    assert r.contact and arm.presses == ["k"]
