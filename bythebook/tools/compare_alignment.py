"""How well does a vision-language model match numbered marks to the manual's parts?

    python -m bythebook.tools.compare_alignment --vlm local --truth appliances/leadzm_scale/alignment_truth.json
    python -m bythebook.tools.compare_alignment --vlm claude ...

Uses the pictures and marks of the last `press_mark` run (results/press_mark/session.pkl)
and the part list of the manual spec. The ground truth is given as the mark ids of each part
in the ego view; it is turned into 3-D points by triangulation with the best-agreeing marks
of the side views, so a prediction in any view counts as correct when the ray through the
predicted mark passes within `--tol-mm` of the part's 3-D point. 0 (no mark) is an abstention.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .. import rig
from ..geometry import pixel_ray, ray_plane_z
from ..perception import draw_marks


def truth_points(cams, views, truth_ego: dict, panel_z: float):
    """{part: xyz} from the ego mark ids: the ego camera looks almost straight down, so the
    ray through a mark's centre meets the control-panel plane within a couple of millimetres."""
    ego = views["ego"]
    pts = {}
    for part, ids in truth_ego.items():
        centers = [k.center for k in ego.marks if k.id in ids]
        if centers:
            pts[part] = ray_plane_z(cams["ego"], np.mean(centers, axis=0), panel_z)[0]
    return pts


def ray_point_distance(cam, uv, P):
    o, d = pixel_ray(cam, uv)
    v = P - o
    return float(np.linalg.norm(v - (v @ d[0]) * d[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", choices=["claude", "local"], required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=Path("appliances/leadzm_scale/leadzm_scale.json"))
    ap.add_argument("--manual", type=Path, default=Path("appliances/leadzm_scale/leadzm_scale_manual.pdf"))
    ap.add_argument("--session", type=Path, default=rig.RESULTS_DIR / "press_mark" / "session.pkl")
    ap.add_argument("--tol-mm", type=float, default=10.0)
    ap.add_argument("--panel-z-mm", type=float, default=41.0)
    ap.add_argument("--local-model", default="Qwen3-VL-32B")
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()

    cams = {r: c for r, c in rig.load_cameras(hires=True).items() if c.T_table_cam is not None}
    with open(a.session, "rb") as f:
        images, views = pickle.load(f)
    truth = json.loads(a.truth.read_text())
    spec = json.loads(a.spec.read_text())
    parts = [{"name": p["name"], "kind": p["kind"], "label": p["label"], "function": p["function"],
              "location": "control panel"} for p in spec["panel_parts"]]
    pts = truth_points(cams, views, truth["ego"], a.panel_z_mm / 1000)
    print("ground-truth points (mm):", {k: np.round(v * 1000).tolist() for k, v in pts.items()})

    from ..vlm import ClaudeVLM, LocalVLM
    vlm = LocalVLM(a.manual, model=a.local_model) if a.vlm == "local" else ClaudeVLM(a.manual)
    rows, t0 = [], time.time()
    for run in range(a.runs):
        for role, v in views.items():
            marked = draw_marks(v.crop, v.marks, v.origin, v.scale)
            ans = vlm.align(parts, v.crop, marked, [k.id for k in v.marks], view=role)
            for m in ans["matches"]:
                part = m["part"]
                if part not in pts:
                    continue
                mark = next((k for k in v.marks if k.id == m["mark"]), None)
                if m["mark"] <= 0 or mark is None:
                    verdict, dist = "abstain", None
                else:
                    dist = ray_point_distance(cams[role], mark.center, pts[part]) * 1000
                    verdict = "correct" if dist <= a.tol_mm else "wrong"
                rows.append({"run": run, "view": role, "part": part, "mark": m["mark"], "confidence": m["confidence"],
                             "dist_mm": dist, "verdict": verdict})
                print(f"  run {run} {role:10s} {part:14s} mark {m['mark']:3d} ({m['confidence'][0]})  "
                      f"{verdict:8s}" + (f" {dist:5.1f} mm" if dist is not None else ""))
    n = len(rows)
    counts = {k: sum(r["verdict"] == k for r in rows) for k in ("correct", "wrong", "abstain")}
    print(f"\n{a.vlm}: {counts['correct']}/{n} correct, {counts['wrong']} wrong, {counts['abstain']} abstained, "
          f"{(time.time() - t0) / max(1, a.runs * len(views)):.1f} s per view")
    out = rig.RESULTS_DIR / f"alignment_{a.vlm}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"vlm": a.vlm, "rows": rows, "counts": counts, "usage": getattr(vlm, "usage", [])}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
