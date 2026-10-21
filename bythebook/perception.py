"""Find the appliance in a camera image and cut it into numbered candidate parts.

Open-vocabulary detection (GroundingDINO) locates the appliance; SAM's automatic mask
generator proposes parts inside the crop; the parts are drawn as numbered marks for a
vision-language model to match against the manual (set-of-mark prompting, as in
CheckManual's no-CAD track).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DINO_MODEL = "IDEA-Research/grounding-dino-base"
# ViT-L rather than ViT-H: it peaks at about 3 GB instead of 6 GB, so it fits beside a locally
# served language model on the 32 GB card, and the part masks are as good for this purpose.
SAM_MODEL = "facebook/sam-vit-large"


@dataclass
class Mark:
    id: int
    center: tuple[float, float]       # full-image pixel well inside the part
    bbox: tuple[int, int, int, int]   # full-image x0, y0, x1, y1
    area_px: float                    # in full-image pixels
    score: float
    contour: np.ndarray               # (N, 2) full-image pixels

    def to_json(self):
        return {"id": self.id, "center": list(self.center), "bbox": list(self.bbox),
                "area_px": self.area_px, "score": self.score}


class Perception:
    def __init__(self, device: str = "cuda"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, pipeline

        self.torch, self.device = torch, device
        self.dino_proc = AutoProcessor.from_pretrained(DINO_MODEL)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(device).eval()
        self.sam = pipeline("mask-generation", model=SAM_MODEL, device=device)
        self.sam_proc = AutoProcessor.from_pretrained(SAM_MODEL)

    # ------------------------------------------------------------------ detection
    def detect(self, image_bgr: np.ndarray, prompt: str, threshold: float = 0.3) -> list[dict]:
        """Boxes matching a text prompt, best first. Several phrases: 'kitchen scale. digital scale.'"""
        from PIL import Image

        text = prompt.lower().strip()
        if not text.endswith("."):
            text += "."
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.dino_proc(images=img, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=threshold, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = [{"box": [float(v) for v in b], "score": float(s), "label": str(l)}
                for b, s, l in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels)]
        return sorted(dets, key=lambda d: -d["score"])

    def object_mask(self, image_bgr: np.ndarray, box) -> np.ndarray:
        """Boolean mask (full image) of the object inside `box`, from SAM with a box prompt."""
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.sam_proc(img, input_boxes=[[[float(v) for v in box]]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.sam.model(**inputs)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
        best = int(out.iou_scores[0, 0].argmax())
        return masks[best].numpy().astype(bool)

    def point_mask(self, image_bgr: np.ndarray, uv) -> np.ndarray | None:
        """Boolean mask (full image) of the thing under pixel `uv`, from SAM with a point prompt."""
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.sam_proc(img, input_points=[[[float(uv[0]), float(uv[1])]]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.sam.model(**inputs)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
        scores = out.iou_scores[0, 0].cpu().numpy()
        # SAM returns three nested candidates; the smallest plausible one is the key, not the panel
        order = np.argsort([m.sum().item() for m in masks])
        for i in order:
            if scores[i] > 0.5:
                return masks[i].numpy().astype(bool)
        return None

    # ------------------------------------------------------------------ parts
    def segment_parts(self, image_bgr: np.ndarray, box, margin: float = 0.06, long_side: int = 1024,
                      min_area_frac: float = 2e-4, max_area_frac: float = 0.35, max_marks: int = 45,
                      on_object_only: bool = True):
        """Candidate parts inside `box` (x0, y0, x1, y1). With on_object_only, parts that
        lie mostly off the object (table, neighbours) are dropped.

        -> (marks, crop_bgr, crop_origin (x0, y0), scale) where crop pixel = (full pixel - origin) * scale.
        The crop is enlarged so that its long side is `long_side`: small buttons get more pixels.
        """
        from PIL import Image

        H, W = image_bgr.shape[:2]
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        x1, y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
        crop = image_bgr[y0:y1, x0:x1]
        scale = long_side / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        out = self.sam(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), points_per_batch=64,
                       pred_iou_thresh=0.86, stability_score_thresh=0.90)
        masks = [np.asarray(m, bool) for m in out["masks"]]
        scores = [float(s) for s in out["scores"]]

        ch, cw = crop.shape[:2]
        obj = None
        if on_object_only:
            full = self.object_mask(image_bgr, box)[y0:y1, x0:x1].astype(np.uint8)
            obj = cv2.resize(full, (cw, ch), interpolation=cv2.INTER_NEAREST)
            obj = cv2.dilate(obj, np.ones((9, 9), np.uint8)).astype(bool)
        keep = []
        for m, s in sorted(zip(masks, scores), key=lambda t: -t[1]):
            frac = m.sum() / (ch * cw)
            if not (min_area_frac <= frac <= max_area_frac):
                continue
            if obj is not None and (m & obj).sum() < 0.7 * m.sum():
                continue
            if any((m & k).sum() / (m | k).sum() > 0.8 for k, _ in keep):
                continue
            keep.append((m, s))
        keep = keep[:max_marks]

        marks = []
        for m, s in keep:
            m8 = np.ascontiguousarray(m, dtype=np.uint8)
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
            blob = np.zeros(m8.shape, np.uint8)
            cv2.drawContours(blob, [cnt.reshape(-1, 1, 2)], -1, 1, -1)
            if blob.sum() > 1.3 * (blob & m8).sum():
                # A ring (a key's rim): its centre is the centre of the filled outline, the same
                # point in every camera, and the middle of the key itself.
                mom = cv2.moments(blob, binaryImage=True)
                cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
            else:
                # otherwise the point deepest inside the blob: a safe place to press
                dist = cv2.distanceTransform(blob & m8, cv2.DIST_L2, 5)
                cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
            bx, by, bw, bh = cv2.boundingRect(cnt.reshape(-1, 1, 2))
            marks.append(Mark(
                id=0,
                center=(cx / scale + x0, cy / scale + y0),
                bbox=(int(bx / scale + x0), int(by / scale + y0), int((bx + bw) / scale + x0), int((by + bh) / scale + y0)),
                area_px=float(m.sum() / scale ** 2), score=s, contour=cnt / scale + [x0, y0]))
        # number in reading order: rows of ~8 % of the crop height, then left to right
        row_h = 0.08 * (y1 - y0)
        marks.sort(key=lambda k: (int((k.center[1] - y0) / row_h), k.center[0]))
        for i, k in enumerate(marks, 1):
            k.id = i
        return marks, crop, (x0, y0), scale


def draw_marks(crop_bgr: np.ndarray, marks: list[Mark], origin, scale: float) -> np.ndarray:
    """Outline every mark and put its number next to it (not on it: buttons carry
    labels the language model has to read)."""
    img = crop_bgr.copy()
    h, w = img.shape[:2]
    rng = np.random.default_rng(7)
    font = cv2.FONT_HERSHEY_SIMPLEX
    taken = []
    for k in marks:
        color = tuple(int(c) for c in rng.integers(60, 256, 3))
        cnt = ((k.contour - origin) * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(img, [cnt], -1, color, 2)
        label = str(k.id)
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        bx0, by0 = (k.bbox[0] - origin[0]) * scale, (k.bbox[1] - origin[1]) * scale
        bx1, by1 = (k.bbox[2] - origin[0]) * scale, (k.bbox[3] - origin[1]) * scale
        big = (bx1 - bx0) > 4 * tw and (by1 - by0) > 4 * th
        cx, cy = (k.center[0] - origin[0]) * scale, (k.center[1] - origin[1]) * scale
        # big parts: number at the centre; small parts: just outside, first free spot
        spots = [(cx - tw / 2, cy + th / 2)] if big else [
            (bx0, by0 - 5), (bx1 + 4, by0 + th), (bx0 - tw - 4, by0 + th), (bx0, by1 + th + 5)]
        for x, y in spots + spots[:1]:
            x, y = int(np.clip(x, 2, w - tw - 2)), int(np.clip(y, th + 2, h - 3))
            box = (x - 2, y - th - 2, x + tw + 2, y + 3)
            if not any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
                break
        taken.append(box)
        cv2.rectangle(img, box[:2], box[2:], (0, 0, 0), -1)
        cv2.putText(img, label, (x, y), font, 0.6, color, 2, cv2.LINE_AA)
    return img
