import numpy as np
import pytest

from bythebook.arm import Limits, Session, SimArm
from bythebook.geometry import make_T, pixel_ray, project, rotvec_T
from bythebook.kinematics import Chain
from bythebook.perception import Mark
from bythebook.pipeline import LocateError, View, execute_plan, locate
from bythebook.registration import Registration
from bythebook.rig import CameraModel
from bythebook.vlm import ScriptedVLM


def _cam(role, pos, target=(0, 0, 0)):
    z = np.array(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
    return CameraModel(role, "", K, np.zeros(5), (1920, 1080), make_T(np.stack([x, y, z], axis=1), pos))


CAMS = {"ego": _cam("ego", np.array([-0.2, 0.05, 0.75])),
        "side_left": _cam("side_left", np.array([0.15, 0.40, 0.40])),
        "side_right": _cam("side_right", np.array([0.2, -0.33, 0.38]))}


def _view(role, part, P, mark_id=3, conf="high", shift_px=(0, 0)):
    uv = project(CAMS[role], P)[0] + shift_px
    mark = Mark(mark_id, tuple(uv), (0, 0, 1, 1), 100.0, 0.9, np.zeros((1, 2)))
    v = View(role, [0, 0, 10, 10], np.zeros((4, 4, 3), np.uint8), (0, 0), 1.0, [mark])
    v.matches = {part: {"part": part, "mark": mark_id, "confidence": conf, "reason": ""}}
    return v


P_TRUE = np.array([0.03, -0.02, 0.031])


def test_locate_anchors_on_the_overhead_view():
    views = {r: _view(r, "TARE", P_TRUE) for r in CAMS}
    loc = locate("TARE", CAMS, views)
    assert loc.method == "anchored" and len(loc.views) == 3
    assert np.linalg.norm(np.array(loc.P_table) - P_TRUE) < 2e-4


def test_locate_uses_geometry_in_side_views_even_when_the_model_answers_wrongly():
    """The side view's model answer points at a neighbouring button, but that view also has a
    mark on the right button: the anchor picks it geometrically."""
    views = {r: _view(r, "TARE", P_TRUE) for r in CAMS}
    right = project(CAMS["side_left"], P_TRUE)[0]
    wrong = right + (60, -45)
    marks = [Mark(3, tuple(wrong), (0, 0, 1, 1), 100.0, 0.9, np.zeros((1, 2))),
             Mark(4, tuple(right), (0, 0, 1, 1), 100.0, 0.9, np.zeros((1, 2)))]
    v = View("side_left", [0, 0, 10, 10], np.zeros((4, 4, 3), np.uint8), (0, 0), 1.0, marks)
    v.matches = {"TARE": {"part": "TARE", "mark": 3, "confidence": "high", "reason": ""}}
    views["side_left"] = v
    loc = locate("TARE", CAMS, views)
    assert loc.method == "anchored" and len(loc.views) == 3
    assert np.linalg.norm(np.array(loc.P_table) - P_TRUE) < 2e-4


def test_locate_without_anchor_refuses_when_two_views_disagree():
    views = {"side_left": _view("side_left", "TARE", P_TRUE),
             "side_right": _view("side_right", "TARE", P_TRUE, shift_px=(70, 50))}
    with pytest.raises(LocateError):
        locate("TARE", CAMS, views)


def test_locate_anchor_alone_needs_a_surface_height():
    views = {"ego": _view("ego", "TARE", P_TRUE)}
    with pytest.raises(LocateError):
        locate("TARE", CAMS, views)
    loc = locate("TARE", CAMS, views, surface_z=0.031)
    assert loc.method == "plane" and np.linalg.norm(np.array(loc.P_table) - P_TRUE) < 3e-4


def test_execute_plan_presses_and_stops_on_failed_check():
    T = rotvec_T([0, 0, 0.05], [-0.22, 0.0, 0.0])
    reg = Registration(T.tolist(), Chain().nominal_tip.tolist(), [0] * 5, 1.0, 1.5, 2.0, 12, gripper_closed=3.0)
    P_on = (T @ np.array([0.20, -0.04, 0.03, 1]))[:3]
    P_tare = (T @ np.array([0.20, 0.04, 0.03, 1]))[:3]
    buttons = {"ON/OFF": dict(xy=P_on[:2], radius_mm=6, travel_mm=1.0, z=0.03),
               "TARE": dict(xy=P_tare[:2], radius_mm=6, travel_mm=1.0, z=0.03)}
    arm = SimArm(reg, surface_z=0.027, buttons=buttons)
    from bythebook.pipeline import Location
    locs = {"ON/OFF": Location("ON/OFF", P_on.tolist(), "triangulated", ["ego", "side_right"], 1.0),
            "TARE": Location("TARE", P_tare.tolist(), "triangulated", ["ego", "side_right"], 1.0)}
    steps = [{"part": "ON/OFF", "operation": "press", "times": 1, "hold_seconds": 0, "expect": "display shows 0 g"},
             {"part": "TARE", "operation": "press", "times": 2, "hold_seconds": 0, "expect": "display shows 0 g"},
             {"part": "ON/OFF", "operation": "press_and_hold", "times": 1, "hold_seconds": 2, "expect": "display off"}]
    vlm = ScriptedVLM({"check": [{"achieved": True, "observed": "0 g", "changed": True},
                                 {"achieved": False, "observed": "12 g", "changed": False}]})
    img = np.zeros((1080, 1920, 3), np.uint8)
    views = {"ego": View("ego", [800, 400, 1100, 700], img, (0, 0), 1.0, [])}
    with Session(arm, reg, Limits(), sleep=lambda s: None) as mover:
        recs = execute_plan(steps, locs, mover, vlm, lambda: {"ego": img}, views, log=lambda m: None, pause=lambda s: None)
    assert [r["ok"] for r in recs] == [True, False]            # third step never attempted
    assert arm.presses == ["ON/OFF", "TARE", "TARE"]
    assert all(p["contact"] for r in recs for p in r["presses"])


def test_execute_plan_with_operator_and_reading_steps():
    T = rotvec_T([0, 0, 0.0], [-0.22, 0.0, 0.0])
    reg = Registration(T.tolist(), Chain().nominal_tip.tolist(), [0] * 5, 1.0, 1.5, 2.0, 12, gripper_closed=3.0)
    P_on = (T @ np.array([0.20, -0.04, 0.03, 1]))[:3]
    arm = SimArm(reg, surface_z=0.027, buttons={"ON/OFF": dict(xy=P_on[:2], radius_mm=6, travel_mm=1.0, z=0.03)})
    from bythebook.pipeline import Location
    locs = {"ON/OFF": Location("ON/OFF", P_on.tolist(), "triangulated", ["ego", "side_right"], 1.0)}
    steps = [{"part": "ON/OFF key", "operation": "press", "times": 1, "hold_seconds": 0, "note": "", "expect": "0.00 g"},
             {"part": "Weighing pan", "operation": "operator_places", "times": 1, "hold_seconds": 0,
              "note": "place the block on the centre of the pan", "expect": "the block's weight in g"},
             {"part": "Display", "operation": "read_display", "times": 1, "hold_seconds": 0,
              "note": "read the weight", "expect": "a weight in g"}]
    locs["ON/OFF key"] = locs["ON/OFF"]
    vlm = ScriptedVLM({"check": {"achieved": True, "observed": "0.00 g", "changed": True},
                       "display": {"display_visible": True, "on": True, "text": "52.31", "unit": "g"}})
    img = np.zeros((1080, 1920, 3), np.uint8)
    views = {"ego": View("ego", [800, 400, 1100, 700], img, (0, 0), 1.0, [])}
    prompts = []
    with Session(arm, reg, Limits(), sleep=lambda s: None) as mover:
        recs = execute_plan(steps, locs, mover, vlm, lambda: {"ego": img}, views, log=lambda m: None,
                            pause=lambda s: None, ask=lambda msg: prompts.append(msg))
    assert [r["ok"] for r in recs] == [True, True, True]
    assert len(prompts) == 1 and "block" in prompts[0]
    assert recs[2]["reading"]["text"] == "52.31"
    assert arm.presses == ["ON/OFF"]


def test_check_heights_replaces_the_outlier():
    from bythebook.pipeline import Location, check_heights
    P_key = np.array([0.03, -0.02, 0.045])                       # really at the panel height
    views = {r: _view(r, "ON/OFF", P_key) for r in CAMS}
    good = Location("TARE", [0.0, -0.07, 0.045], "anchored", ["ego", "side_left", "side_right"], 1.0)
    bad = Location("ON/OFF", [P_key[0], P_key[1], 0.071], "anchored", ["ego", "side_left"], 4.0)   # rim-to-rim match
    locs = {"TARE": good, "ON/OFF": bad}
    check_heights(locs, CAMS, views, log=lambda m: None)
    assert locs["TARE"] is good
    assert abs(locs["ON/OFF"].P_table[2] - 0.045) < 1e-6 and locs["ON/OFF"].method.startswith("plane")
    assert np.linalg.norm(np.array(locs["ON/OFF"].P_table[:2]) - P_key[:2]) < 1e-3


def test_locate_prefers_the_side_marks_that_agree_with_each_other():
    """In one side view an unrelated mark lies almost exactly on the anchor's ray (a rim
    behind the key); the mark on the key itself is a little further off the ray but agrees
    with the other side view. The three-view combination must win."""
    views = {r: _view(r, "ON/OFF", P_TRUE) for r in CAMS}
    right = project(CAMS["side_left"], P_TRUE)[0]
    o, d = pixel_ray(CAMS["ego"], project(CAMS["ego"], P_TRUE)[0])
    impostor = project(CAMS["side_left"], o + d[0] * (np.linalg.norm(P_TRUE - o) - 0.03))[0]   # 3 cm nearer the ego camera, on its ray
    marks = [Mark(1, tuple(impostor), (0, 0, 1, 1), 100.0, 0.9, np.zeros((1, 2))),
             Mark(2, tuple(right + (4.0, -3.0)), (0, 0, 1, 1), 100.0, 0.9, np.zeros((1, 2)))]
    v = View("side_left", [0, 0, 10, 10], np.zeros((4, 4, 3), np.uint8), (0, 0), 1.0, marks)
    v.matches = {"ON/OFF": {"part": "ON/OFF", "mark": 1, "confidence": "high", "reason": ""}}
    views["side_left"] = v
    loc = locate("ON/OFF", CAMS, views)
    assert len(loc.views) == 3
    assert abs(loc.P_table[2] - P_TRUE[2]) < 3e-3

