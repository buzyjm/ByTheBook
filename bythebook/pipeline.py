"""Manual -> parts -> plan -> locate -> press -> check, on the real rig.

    python -m bythebook.pipeline --manual scale.pdf --task "switch the scale on and set it to zero"

Without --execute nothing moves: the run stops after locating the parts, writes pictures
of where the arm would press (results/<time>/targets_*.jpg) and simulates the
presses. With --execute the same plan is carried out after a typed confirmation.

    --vlm claude|local   Claude API (default) or a model served by vLLM (tools/serve_local_vlm.sh)
    --plan-only          manual reading and planning only, no cameras, no arm
    --images DIR         use DIR/<role>.jpg instead of the live cameras
    --script FILE        answer from a prepared JSON instead of calling a language model
    --surface-z-mm Z     height of the pressed surface above the table, used when a part is
                         only seen by one camera
    --wrist-servo        correct every press from the wrist camera (tools/wrist_servo.py)
    --record             record the fixed cameras during execution

Every run writes report.json, log.txt and the captured pictures to results/<time>/.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import rig
from .arm import Limits, MotionError, RealArm, Session, SimArm
from .cameras import grab_all
from .geometry import project, ray_plane_z, triangulate
from .perception import Perception, draw_marks
from .recorder import Recorder
from .registration import Registration
from .vlm import ClaudeVLM, LocalVLM, ScriptedVLM
from .wrist import KeyFinder, WristServo, make_servo

CONF_RANK = {"low": 0, "medium": 1, "high": 2}
ANCHOR_ORDER = ("ego", "side_right", "side_left")     # most top-down first: labels are legible there


@dataclass
class View:
    role: str
    box: list
    crop: np.ndarray
    origin: tuple
    scale: float
    marks: list
    matches: dict = field(default_factory=dict)      # part -> {"mark", "confidence", "reason"}

    def pixel(self, part: str, min_conf: str = "medium"):
        m = self.matches.get(part)
        if not m or m["mark"] <= 0 or CONF_RANK[m["confidence"]] < CONF_RANK[min_conf]:
            return None
        mark = next((k for k in self.marks if k.id == m["mark"]), None)
        return None if mark is None else mark.center


@dataclass
class Location:
    part: str
    P_table: list
    method: str               # "triangulated" or "plane"
    views: list
    reproj_px: float | None


class LocateError(RuntimeError):
    pass


def describe(loc: Location) -> str:
    """One log line: position in mm, method, views used and their agreement."""
    return (f"{loc.part}: {np.round(np.array(loc.P_table) * 1000, 1)} mm ({loc.method}, views {loc.views}"
            + (f", agreement {loc.reproj_px:.1f} px)" if loc.reproj_px is not None else ")"))


# ----------------------------------------------------------------------------- perception stage
def segment_views(cams, images, perception, manual, prompt=None, log=print) -> dict:
    """Detect the appliance in every camera and mark its candidate parts (GPU perception only)."""
    views = {}
    text = prompt or manual.get("detection_prompt", "")
    if not text or len(text.split()) > 5:            # a model that misfilled the field: use the appliance name
        text = manual["appliance"]
    for role, img in images.items():
        dets = perception.detect(img, text)
        if not dets:
            log(f"  {role}: '{text}' not found")
            continue
        box = dets[0]["box"]
        marks, crop, origin, scale = perception.segment_parts(img, box)
        if not marks:
            log(f"  {role}: no candidate parts")
            continue
        views[role] = View(role, box, crop, origin, scale, marks)
    return views


def panel_closeup(v: View, image, manual_parts, margin: float = 0.4):
    """A tight, enlarged crop around the keys and the display matched in the first pass, with
    only the marks inside it. The key labels are legible there, which the whole-appliance
    picture does not offer."""
    kinds = {p["name"]: p["kind"] for p in manual_parts}
    bw, bh = v.box[2] - v.box[0], v.box[3] - v.box[1]
    chosen = [next((k for k in v.marks if k.id == m["mark"]), None)
              for part, m in v.matches.items()
              if m["mark"] > 0 and kinds.get(part) in ("button", "knob", "display")
              and not any(w in part.lower() for w in ("panel", "cord", "cable", "plug"))]
    chosen = [k for k in chosen if k is not None and (k.bbox[2] - k.bbox[0]) < 0.35 * bw and (k.bbox[3] - k.bbox[1]) < 0.35 * bh]
    if len(chosen) < 3:
        return None
    xs = [c for k in chosen for c in (k.bbox[0], k.bbox[2])]
    ys = [c for k in chosen for c in (k.bbox[1], k.bbox[3])]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    mx, my = max(30, (x1 - x0) * margin), max(30, (y1 - y0) * margin)
    H, W = image.shape[:2]
    x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))
    inside = [k for k in v.marks if x0 <= k.center[0] <= x1 and y0 <= k.center[1] <= y1
              and (k.bbox[2] - k.bbox[0]) < 0.8 * (x1 - x0)]
    if len(inside) < 3:
        return None
    crop = image[y0:y1, x0:x1]
    scale = 1024 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, (x0, y0), scale, inside


def align_views(views: dict, vlm, manual, images=None, log=print) -> dict:
    """Ask the language model which mark is which part, in every view. In the anchor view
    the keys are asked about a second time on a close-up of the control panel, where their
    labels can be read; that answer wins for keys."""
    for role, v in views.items():
        ans = vlm.align(manual["parts"], v.crop, draw_marks(v.crop, v.marks, v.origin, v.scale), [k.id for k in v.marks], view=role)
        v.matches = {m["part"]: m for m in ans["matches"]}
        log(f"  {role}: {len(v.marks)} marks, " + ", ".join(
            f"{p}={m['mark']}({m['confidence'][0]})" for p, m in v.matches.items() if m["mark"] > 0))
        if images is not None and role == ANCHOR_ORDER[0]:
            cu = panel_closeup(v, images[role], manual["parts"])
            if cu is not None:
                crop, origin, scale, inside = cu
                keys = [p for p in manual["parts"] if p["kind"] in ("button", "switch", "knob", "display")]
                ans2 = vlm.align(keys, crop, draw_marks(crop, inside, origin, scale), [k.id for k in inside],
                                 view=f"{role}, close-up of the control panel")
                changed = []
                for m in ans2["matches"]:
                    if m["mark"] > 0 and m["part"] in v.matches and v.matches[m["part"]]["mark"] != m["mark"]:
                        changed.append(f"{m['part']} {v.matches[m['part']]['mark']}->{m['mark']}")
                    if m["mark"] > 0:
                        v.matches[m["part"]] = m
                log(f"  {role} close-up: " + (", ".join(changed) if changed else "same answers"))
    return views


def locate(part: str, cams, views, surface_z=None, max_reproj_px: float = 8.0, anchor_px: float = 5.0) -> Location:
    """3D position of a part.

    The model's answer in the anchor view (the overhead camera when it has one) is trusted;
    in every other view the mark is chosen geometrically: the one whose centre triangulates
    with the anchor mark within `anchor_px`. The model's own answers in those views only
    confirm or contradict. Without an anchor answer, the old rule applies: two or more views
    that agree. One view alone needs a known surface height.
    """
    px = {r: v.pixel(part) for r, v in views.items()}
    px = {r: p for r, p in px.items() if p is not None and r in cams}
    anchor = next((r for r in ANCHOR_ORDER if r in px), None)
    if anchor:
        # Candidate marks in every other view: those near the anchor's ray. Several marks can lie
        # along that ray at different depths (a key and a rim behind it), so the combination of
        # candidates that agrees best across ALL views decides, with more views preferred.
        cands = {}
        for role, v in views.items():
            if role == anchor or role not in cams or not v.marks:
                continue
            scored = sorted(((triangulate([cams[anchor], cams[role]], [px[anchor], k.center])[1].max(), k)
                             for k in v.marks), key=lambda t: t[0])
            near = [k for e, k in scored if e <= 2 * anchor_px][:4]
            if near:
                cands[role] = near
        best = None
        for n_other in range(len(cands), 0, -1):
            for roles_o in itertools.combinations(sorted(cands), n_other):
                for combo in itertools.product(*(cands[r] for r in roles_o)):
                    roles = [anchor, *roles_o]
                    uvs = [px[anchor], *(k.center for k in combo)]
                    P, err = triangulate([cams[r] for r in roles], uvs)
                    if err.max() <= (anchor_px if n_other == 1 else max_reproj_px) and (best is None or err.max() < best[2]):
                        best = (P, roles, float(err.max()))
            if best is not None:
                break
        if best is not None:
            return Location(part, best[0].tolist(), "anchored", best[1], best[2])
        if surface_z is not None:
            return Location(part, ray_plane_z(cams[anchor], px[anchor], surface_z)[0].tolist(), "plane", [anchor], None)
        raise LocateError(f"'{part}': only the {anchor} view sees it and no other view has a mark on the same spot; "
                          "pass --surface-z-mm")
    roles = sorted(px)
    best = None
    subsets = [roles] if len(roles) >= 2 else []
    if len(roles) >= 3:
        subsets += [[a, b] for i, a in enumerate(roles) for b in roles[i + 1:]]
    for sub in subsets:
        P, err = triangulate([cams[r] for r in sub], [px[r] for r in sub])
        if err.max() <= max_reproj_px and (best is None or (len(sub), -err.max()) > (len(best[1]), -best[2])):
            best = (P, sub, float(err.max()))
    if best:
        return Location(part, best[0].tolist(), "triangulated", best[1], best[2])
    if surface_z is not None and roles:
        r = max(roles, key=lambda k: -cams[k].T_table_cam[2, 2])
        P = ray_plane_z(cams[r], px[r], surface_z)[0]
        return Location(part, P.tolist(), "plane", [r], None)
    if len(roles) >= 2:
        raise LocateError(f"'{part}': the cameras disagree about where it is (marks {px}); not pressing")
    raise LocateError(f"'{part}': found in {len(roles)} camera view(s); need two, or --surface-z-mm")


def check_heights(locations: dict, cams, views, log=print, tol_mm: float = 12.0):
    """Keys of one control panel sit at about the same height. A part whose height stands
    out (a rim matched to a rim in another view can do that) is re-located by intersecting
    its overhead ray with the height of the others."""
    anchored = {p: l for p, l in locations.items() if l.method == "anchored"}
    if len(anchored) < 2:
        return
    for part, loc in list(anchored.items()):
        others = [l for p, l in anchored.items() if p != part]
        ref = float(np.median([l.P_table[2] for l in others]))
        if len(anchored) == 2 and len(loc.views) >= len(others[0].views):
            if len(loc.views) == len(others[0].views) and abs(loc.P_table[2] - ref) * 1000 > tol_mm:
                log(f"  warning: {part} and {[p for p in anchored if p != part][0]} differ in height by "
                    f"{abs(loc.P_table[2] - ref) * 1000:.0f} mm and neither is better supported; keeping both")
            continue
        if abs(loc.P_table[2] - ref) * 1000 > tol_mm and len(loc.views) <= max(len(l.views) for l in others):
            anchor = loc.views[0]
            uv = views[anchor].pixel(part)
            P = ray_plane_z(cams[anchor], uv, ref)[0]
            log(f"  {part}: height {loc.P_table[2] * 1000:.0f} mm is far from the other parts ({ref * 1000:.0f} mm); "
                f"using their height -> {np.round(P * 1000, 1)} mm")
            locations[part] = Location(part, P.tolist(), "plane at the height of the other parts", [anchor], None)


def panel_approach(cams, views, manual_parts, located: dict, log=print, max_tilt_deg: float = 35.0):
    """Direction to press along: the negated normal of the control panel, fitted through the
    3-D positions of the keys and the display (anchored ones only). Falls back to straight
    down with fewer than three well-spread points or an implausible slope."""
    pts = {p: np.array(l.P_table) for p, l in located.items() if l.method == "anchored"}
    for p in manual_parts:
        name = p["name"]
        if name in pts or p["kind"] not in ("button", "knob", "display") or "panel" in name.lower():
            continue
        try:
            l = locate(name, cams, views)
            if l.method == "anchored":
                pts[name] = np.array(l.P_table)
        except LocateError:
            pass
    saved = rig.STATE_DIR / "panel_normal.json"
    fallback = (np.array([0.0, 0.0, -1.0]), None)
    if saved.exists():
        d = json.loads(saved.read_text())
        fallback = (np.array(d["approach"]), d["tilt_deg"])
    if len(pts) < 4:
        return fallback
    X = np.array(list(pts.values()))
    c = X.mean(0)
    _, sv, vt = np.linalg.svd(X - c)
    if sv[1] < 0.12 * sv[0] or sv[1] < 0.012:    # points nearly on a line: the plane is not defined
        log(f"  control panel: the {len(pts)} located parts are too close to a line for a plane fit")
        return fallback
    n = vt[2]
    if n[2] < 0:
        n = -n
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    if tilt > max_tilt_deg:
        log(f"  panel normal tilted {tilt:.0f} deg: not plausible")
        return fallback
    log(f"  control panel from {len(pts)} parts: tilted {tilt:.0f} deg, pressing along its normal")
    saved.write_text(json.dumps({"approach": (-n).tolist(), "tilt_deg": tilt, "parts": list(pts)}))
    return -n, tilt


def locate_with_fallback(part, cams, views, surface_z, located: dict, log=print) -> Location:
    """locate(), and when a part has no cross-view agreement, intersect its anchor ray with
    the height of the parts already located (the keys of one control panel share it)."""
    try:
        loc = locate(part, cams, views, surface_z=surface_z)
    except LocateError:
        same_height = [l.P_table[2] for l in located.values() if l.method != "plane"]
        if not same_height:
            raise
        loc = locate(part, cams, views, surface_z=float(np.mean(same_height)))
        loc.method = "plane at the height of the other parts"
    log("  " + describe(loc))
    return loc


def draw_targets(cams, images, locations: dict, out_dir: Path):
    for role, img in images.items():
        if role not in cams:
            continue
        canvas = img.copy()
        for part, loc in locations.items():
            u, v = project(cams[role], np.array(loc.P_table))[0]
            cv2.drawMarker(canvas, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 36, 2)
            cv2.putText(canvas, part, (int(u) + 14, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"targets_{role}.jpg"), canvas)


def display_crops(images, views, margin=1.0, min_side=400):
    """Close-ups of the display: around the mark the model matched to the display part in each
    view, enlarged; the whole appliance where no display mark is known."""
    crops = []
    for role, v in views.items():
        img = images.get(role)
        if img is None:
            continue
        m = next((mm for part, mm in v.matches.items() if "display" in part.lower() and mm["mark"] > 0), None)
        mark = next((k for k in v.marks if m and k.id == m["mark"]), None)
        if mark is None:
            x0, y0, x1, y1 = v.box
            mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        else:
            x0, y0, x1, y1 = mark.bbox
            mx, my = max(40, (x1 - x0) * margin), max(40, (y1 - y0) * margin)
        crop = img[int(max(0, y0 - my)):int(y1 + my), int(max(0, x0 - mx)):int(x1 + mx)]
        if crop.size and min(crop.shape[:2]) < min_side:
            f = min_side / min(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        if crop.size:
            crops.append(crop)
    return crops


# ----------------------------------------------------------------------------- execution stage
def press_steps(steps):
    return [s for s in steps if s["operation"].startswith("press")]


def execute_plan(steps, locations, mover, vlm, capture, views, log=print, pause=time.sleep, ask=input,
                 dry_run: bool = False, save_dir: Path | None = None, settle_s: float = 4.0, approach=None,
                 servo=None):
    """Work through the plan, checking the display after every step. Presses are done by the
    arm; operator steps wait for the human helper at the terminal; read_display steps record
    what the display shows. Stops at the first step that fails. -> list of per-step records."""
    records = []
    for i, step in enumerate(steps, 1):
        op = step["operation"]
        shot = capture()
        before = display_crops(shot, views)
        if save_dir:
            for role, img in shot.items():
                cv2.imwrite(str(save_dir / f"step{i}_before_{role}.jpg"), img)
        presses, reading = [], None
        motion_error = None
        if op.startswith("press"):
            loc = locations[step["part"]]
            for _ in range(max(1, int(step["times"]))):
                dwell = float(step["hold_seconds"]) if op == "press_and_hold" else 0.4
                try:
                    r = mover.press(np.array(loc.P_table), dwell_s=max(0.4, dwell), approach=approach, servo=servo)
                except MotionError as e:                # this press is lost, the run goes on
                    motion_error = str(e)
                    log(f"  motion refused: {e}")
                    break
                presses.append({"contact": r.contact, "late_contact": r.late_contact, "contact_z_mm": r.contact_z_mm,
                                "lowest_z_mm": r.lowest_z_mm, "push_load": r.push_load, "servo_mm": r.servo_mm,
                                "trace": [[round(a, 1), round(b, 2), round(c)] for a, b, c in r.trace]})
                pause(0.5)
            mover.retreat(np.array(loc.P_table))   # lift and step back so the cameras see the display
            pause(0.8)
        elif op in ("operator_places", "operator_removes"):
            ask(f"\n  STEP {i} for the helper: {step.get('note') or op}. Keep hands clear of the arm, then press ENTER: ")
            pause(1.0)
        elif op == "read_display":
            pause(0.5)
        pause(settle_s)                                 # start-up sequences and stabilisation take a few seconds
        shot = capture()
        after = display_crops(shot, views)
        if save_dir:
            for role, img in shot.items():
                cv2.imwrite(str(save_dir / f"step{i}_after_{role}.jpg"), img)
            for k, c in enumerate(after):
                cv2.imwrite(str(save_dir / f"step{i}_after_display{k}.jpg"), c)
        if dry_run:                                   # the pictures are static: nothing to check
            check = {"achieved": True, "observed": "not checked (dry run)", "changed": False}
        elif op == "read_display":
            reading = vlm.read_display(after)
            check = {"achieved": bool(reading["display_visible"]), "observed": f"{reading['text']} {reading['unit']}".strip(),
                     "changed": False}
        else:
            check = vlm.check_step(step, before, after)
            if not check["achieved"]:                    # maybe still settling: look once more
                pause(settle_s)
                shot = capture()
                after = display_crops(shot, views)
                check = vlm.check_step(step, before, after)
        if motion_error:
            check = {"achieved": False, "observed": check.get("observed", ""), "changed": False, "motion_error": motion_error}
        rec = {"step": i, **step, "presses": presses, "reading": reading, "check": check, "ok": bool(check["achieved"])}
        records.append(rec)
        what = f"{op} '{step['part']}' x{step['times']}" if op.startswith("press") else f"{op}: {step.get('note', '')}"
        log(f"  step {i}: {what} -> " + (f"contact {[p['contact'] for p in presses]}, " if presses else "")
            + f"display: {check['observed']!r}, {'OK' if rec['ok'] else 'NOT as expected (' + step['expect'] + ')'}")
        if not rec["ok"]:
            break
    return records


def run_with_replanning(plan, task, manual, locate_fn, mover, vlm, capture, views, log=print, pause=time.sleep,
                        ask=input, max_replans: int = 1, dry_run: bool = False, save_dir: Path | None = None,
                        approach=None, servo=None):
    """Execute the plan; when a step fails, ask the model for a new plan from the observed
    state (once by default) and continue with it. -> (all step records, final plan steps)."""
    steps, records, replans = list(plan["steps"]), [], 0
    while True:
        recs = execute_plan(steps, locate_fn(steps), mover, vlm, capture, views, log, pause, ask, dry_run, save_dir,
                            approach=approach, servo=servo)
        records += recs
        if len(recs) == len(steps) and all(r["ok"] for r in recs):
            return records, steps
        if replans >= max_replans or not recs:
            return records, steps
        replans += 1
        last = recs[-1]
        if last["presses"] and all(p["contact"] for p in last["presses"]):
            mover.lim.press_load = min(mover.lim.press_load * 1.4, 700.0)
            mover.lim.press_depth_mm = min(mover.lim.press_depth_mm + 4.0, 22.0)
            log(f"  the key was touched but nothing happened: pushing harder next time "
                f"(load {mover.lim.press_load:.0f}, depth {mover.lim.press_depth_mm:.0f} mm)")
        now = vlm.read_display(display_crops(capture(), views))
        observed = f"{now['text']} {now['unit']}".strip() if now["display_visible"] else "display off or not visible"
        done = [f"{r['operation']} {r['part']} x{r['times']}" for r in records if r["ok"]]
        state = (f"display now shows: {observed!r}. Steps already done: {done}. The last step "
                 f"({recs[-1]['operation']} {recs[-1]['part']}) did not give the expected result ({recs[-1]['expect']}).")
        log(f"  replanning from the observed state ({observed!r})")
        new_plan = vlm.plan(manual["parts"], task, state=state)
        if not new_plan["feasible"] or not new_plan["steps"]:
            return records, steps
        steps = new_plan["steps"]
        for s_ in steps:
            log(f"  new plan: {s_['part']}: {s_['operation']} x{s_['times']} {s_.get('note', '')} -> {s_['expect']}")


# ----------------------------------------------------------------------------- CLI
def _manual_info(vlm, pdf: Path, log, tag: str = "claude") -> dict:
    cache = rig.RESULTS_DIR / "manual_cache" / (hashlib.sha256(pdf.read_bytes()).hexdigest()[:16] + f"_{tag}.json")
    if cache.exists():
        log(f"manual: using cached reading {cache}")
        return json.loads(cache.read_text())
    info = vlm.read_manual()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", type=Path, help="the appliance's user manual (PDF)")
    ap.add_argument("--task", required=True)
    ap.add_argument("--execute", action="store_true", help="move the real arm")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--script", type=Path)
    ap.add_argument("--surface-z-mm", type=float)
    ap.add_argument("--prompt", help="override the detector prompt, e.g. 'kitchen scale'")
    ap.add_argument("--roles", nargs="+", default=list(rig.FIXED_ROLES))
    ap.add_argument("--vlm", choices=["claude", "local"], default="claude",
                    help="claude: Claude API; local: a model served by vLLM at http://127.0.0.1:8000/v1")
    ap.add_argument("--local-model", default="Qwen3-VL-32B", help="served model name for --vlm local")
    ap.add_argument("--record", action="store_true", help="record the fixed cameras during execution (1280x720)")
    ap.add_argument("--push-load", type=float,
                    help=f"load rise to push to after contact (default {Limits.press_load:.0f})")
    ap.add_argument("--push-depth-mm", type=float,
                    help=f"deepest push below the contact point (default {Limits.press_depth_mm:.0f} mm)")
    ap.add_argument("--approach-tilt-deg", type=float,
                    help="press along a direction tilted this much from vertical towards the pan (+x)")
    ap.add_argument("--fit-panel", action="store_true",
                    help="press along the normal of a plane fitted through the keys (experimental)")
    ap.add_argument("--wrist-servo", action="store_true",
                    help="correct every press from the wrist camera (needs bythebook/tools/wrist_servo calibrate)")
    a = ap.parse_args(argv)

    out = rig.RESULTS_DIR / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    if a.script:
        vlm = ScriptedVLM(a.script)
        manual = vlm.read_manual()
    else:
        if not a.manual:
            ap.error("--manual is required unless --script is given")
        vlm = LocalVLM(a.manual, model=a.local_model) if a.vlm == "local" else ClaudeVLM(a.manual)
        manual = _manual_info(vlm, a.manual, log, tag=a.vlm)
    log(f"appliance: {manual['appliance']}; parts: {[p['name'] for p in manual['parts']]}")
    report = {"task": a.task, "manual": manual, "executed": bool(a.execute)}

    def finish(status):
        report["status"] = status
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        (out / "log.txt").write_text("\n".join(lines))
        log(f"{status}. Results in {out}")
        return report

    if a.plan_only:
        plan = vlm.plan(manual["parts"], a.task)
        report["plan"] = plan
        for s in plan["steps"]:
            log(f"  {s['part']}: {s['operation']} x{s['times']} {s.get('note', '')} -> {s['expect']}")
        return finish("planned" if plan["feasible"] else f"not feasible: {plan['reason']}")

    # ---- look
    cams = {r: c for r, c in rig.load_cameras(hires=True).items() if r in a.roles and c.T_table_cam is not None}

    rec = None

    def capture():
        if a.images:
            return {r: cv2.imread(str(a.images / f"{r}.jpg")) for r in cams if (a.images / f"{r}.jpg").exists()}
        if rec is not None:
            # The recorder owns the cameras. Its 1280x720 frames are the 1920x1080 frames scaled
            # by 2/3 (measured on this rig), so scaling them back keeps every pixel coordinate valid.
            return {r: cv2.resize(f, (cams[r].size[0], cams[r].size[1]), interpolation=cv2.INTER_CUBIC)
                    for r, f in rec.latest.items() if r in cams}
        return grab_all(cams)

    images = capture()
    for r, img in images.items():
        cv2.imwrite(str(out / f"scene_{r}.jpg"), img)
    perception = Perception()
    log("looking for the appliance and its parts")
    views = segment_views(cams, images, perception, manual, a.prompt, log)
    if a.vlm == "local" and not a.wrist_servo:       # the local model shares the GPU: make room for it
        import gc
        import torch
        del perception
        gc.collect()
        torch.cuda.empty_cache()
    views = align_views(views, vlm, manual, images, log)
    for r, v in views.items():
        cv2.imwrite(str(out / f"marks_{r}.jpg"), draw_marks(v.crop, v.marks, v.origin, v.scale))
    report["views"] = {r: {"box": v.box, "matches": v.matches, "marks": [k.to_json() for k in v.marks]}
                       for r, v in views.items()}
    if not views:
        return finish("failed: appliance not found in any camera")

    # ---- plan
    state = vlm.read_display(display_crops(images, views))
    report["initial_display"] = state
    plan = vlm.plan(manual["parts"], a.task, state=json.dumps(state, ensure_ascii=False))
    report["plan"] = plan
    if not plan["feasible"] or not plan["steps"]:
        return finish(f"not feasible: {plan['reason']}")
    for s in plan["steps"]:
        log(f"  plan: {s['part']}: {s['operation']} x{s['times']} {s.get('note', '')} -> {s['expect']}")

    # ---- locate
    locations = {}
    z = None if a.surface_z_mm is None else a.surface_z_mm / 1000
    try:
        parts = list(dict.fromkeys(s["part"] for s in press_steps(plan["steps"])))
        # Parts with cross-view agreement first: their height then serves the others.
        pending = []
        for part in parts:
            try:
                locations[part] = locate(part, cams, views, surface_z=z)
                log("  " + describe(locations[part]))
            except LocateError:
                pending.append(part)
        for part in pending:
            locations[part] = locate_with_fallback(part, cams, views, z, locations, log)
        check_heights(locations, cams, views, log)
        # Pressing straight down is the default: the keys, the recessed display and the raised
        # key bar do not share one plane, so a fitted panel normal over-tilts and the fingertip
        # slides along the surface instead of compressing the key.
        approach, panel_tilt = np.array([0.0, 0.0, -1.0]), None
        if a.approach_tilt_deg is not None:
            t = np.radians(a.approach_tilt_deg)
            approach, panel_tilt = np.array([np.sin(t), 0.0, -np.cos(t)]), a.approach_tilt_deg
            log(f"  pressing along a direction tilted {a.approach_tilt_deg:.0f} deg towards the pan (given)")
        elif a.fit_panel:
            approach, panel_tilt = panel_approach(cams, views, manual["parts"], locations, log)
    except LocateError as e:
        return finish(f"failed: {e}")
    report["approach"] = approach.tolist()
    report["panel_tilt_deg"] = panel_tilt
    report["locations"] = {p: asdict(l) for p, l in locations.items()}
    draw_targets(cams, images, locations, out)
    log(f"press targets drawn in {out}/targets_*.jpg")

    # ---- act
    try:
        reg = Registration.load()
    except FileNotFoundError as e:
        return finish(f"located, not pressed: {e}")
    if a.execute:
        if a.images:
            return finish("refusing --execute with --images: the pictures may not show the current scene")
        if input(f"Check {out}/targets_*.jpg. Arm workspace clear? Type 'go' to press: ").strip() != "go":
            return finish("located, cancelled by the operator")
        arm, sleep, cap = RealArm(torque=True), time.sleep, capture
    else:
        ro = RealArm(torque=None).connect()
        q0, _ = ro.read()
        ro.disconnect()
        zs = np.mean([l.P_table[2] for l in locations.values()])
        arm, sleep, cap = SimArm(reg, q0_deg=q0, surface_z=zs), (lambda s: None), (lambda: images)
        log("dry run: presses are simulated from the arm's current pose")
    def locate_fn(steps):
        for part in dict.fromkeys(s_["part"] for s_ in press_steps(steps)):
            if part not in locations:
                locations[part] = locate_with_fallback(part, cams, views, z, locations, log)
        return locations

    servo = None
    if a.wrist_servo and a.execute:
        servo = make_servo(WristServo.load(), KeyFinder(perception), log)
    limits = Limits()
    if a.push_load:
        limits.press_load = a.push_load
    if a.push_depth_mm:
        limits.press_depth_mm = a.push_depth_mm
    if a.record and a.execute:
        rec = Recorder(list(cams), out, size=(1280, 720), tag="run").start()
        time.sleep(1.5)
        if set(rec.latest) != set(cams):
            rec.stop()
            return finish(f"failed: recording delivered no frames from {sorted(set(cams) - set(rec.latest))}")
    try:
        with Session(arm, reg, limits, sleep=sleep) as mover:
            report["steps"], final_steps = run_with_replanning(
                plan, a.task, manual, locate_fn, mover, vlm, cap, views, log, pause=sleep, dry_run=not a.execute,
                save_dir=out, approach=approach, servo=servo)
    except (MotionError, LocateError) as e:
        return finish(f"failed: {e}")
    finally:
        if rec is not None:
            files = rec.stop()
            log("videos: " + ", ".join(p.name for p in files.values()))
    report["final_plan"] = final_steps
    done = len(report["steps"]) >= len(final_steps) and all(r["ok"] for r in report["steps"][-len(final_steps):])
    readings = [r["reading"] for r in report["steps"] if r.get("reading")]
    if readings:
        log(f"  reading: {readings[-1]['text']} {readings[-1]['unit']}".rstrip())
    return finish(("success" if done else "failed: a step did not have the expected effect")
                  + ("" if a.execute else " (dry run)"))


if __name__ == "__main__":
    main()
