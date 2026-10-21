"""The language-model side: read the manual, match marks to parts, plan, check the display.

PromptedVLM holds the five requests as backend-neutral content. ClaudeVLM sends them to the
Claude API (official SDK, credentials from the environment, the manual attached as a PDF);
LocalVLM sends them to a model served locally through an OpenAI-compatible API (the manual
attached as page images). ScriptedVLM answers from a prepared dict and is what the tests
and offline dry runs use. All return plain dicts that follow the schemas below.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

MODEL = os.environ.get("BYTHEBOOK_VLM_MODEL", "claude-opus-5-5")
# press / press_and_hold are done by the robot; operator_places / operator_removes are done by a
# human helper at the robot's request; read_display reports what the display shows.
OPERATIONS = ("press", "press_and_hold", "operator_places", "operator_removes", "read_display")

SYSTEM = (
    "You help a small robot arm operate a household appliance by following its user manual. "
    "The arm can only press parts straight down with one fingertip. Be literal and conservative: "
    "use only what the manual and the images show, and say so when something cannot be determined."
)


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


MANUAL_SCHEMA = _obj({
    "appliance": {"type": "string", "description": "What the appliance is, e.g. 'digital kitchen scale'."},
    "detection_prompt": {"type": "string", "description": "Two to four lowercase words naming the appliance for an "
                         "object detector, e.g. 'kitchen scale'."},
    "parts": {"type": "array", "items": _obj({
        "name": {"type": "string", "description": "The manual's own name for the part."},
        "kind": {"type": "string", "enum": ["button", "knob", "switch", "display", "platform", "other"]},
        "label": {"type": "string", "description": "Text or icon printed on or next to the part, empty if none."},
        "function": {"type": "string"},
        "location": {"type": "string", "description": "Where it is on the appliance according to the manual."},
    })},
    "notes": {"type": "string", "description": "Anything that matters for operating it: auto-off time, long presses, units."},
})

ALIGN_SCHEMA = _obj({
    "matches": {"type": "array", "items": _obj({
        "part": {"type": "string"},
        "mark": {"type": "integer", "description": "Number of the mark on that part, 0 if no mark is on it."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "reason": {"type": "string"},
    })},
})

PLAN_SCHEMA = _obj({
    "feasible": {"type": "boolean"},
    "reason": {"type": "string", "description": "Why the task cannot be done by pressing, if it cannot. Else empty."},
    "steps": {"type": "array", "items": _obj({
        "part": {"type": "string", "description": "Exactly one of the part names given."},
        "operation": {"type": "string", "enum": list(OPERATIONS)},
        "times": {"type": "integer"},
        "hold_seconds": {"type": "number", "description": "0 for a normal press."},
        "note": {"type": "string", "description": "For operator steps: what to place or remove and where. For "
                 "read_display: what to read. Empty for presses."},
        "expect": {"type": "string", "description": "The steady state of the display a few seconds after this step, "
                   "e.g. '0.00 g' or 'off'. Not transient start-up sequences."},
    })},
})

DISPLAY_SCHEMA = _obj({
    "display_visible": {"type": "boolean"},
    "on": {"type": "boolean"},
    "text": {"type": "string", "description": "Characters shown, exactly, empty if off or unreadable."},
    "unit": {"type": "string", "description": "Unit shown (g, oz, ml, lb:oz ...), empty if none."},
})

CHECK_SCHEMA = _obj({
    "achieved": {"type": "boolean"},
    "observed": {"type": "string", "description": "What the display shows after the step, in a few words, "
                 "e.g. '0.00 g', 'off', '12.5 ct'."},
    "changed": {"type": "boolean", "description": "Whether the display differs between the two images."},
})


def _jpeg_b64(img_bgr: np.ndarray, max_side: int = 1568) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.standard_b64encode(buf.tobytes()).decode()


def image_block(img_bgr: np.ndarray, max_side: int = 1568) -> dict:
    """Anthropic-format image block."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(img_bgr, max_side)}}


class PromptedVLM:
    """The five requests, written once as backend-neutral content: a list of ("text", str)
    and ("image", ndarray) items. Subclasses implement _ask(content, schema, with_manual)."""

    def _ask(self, content: list, schema: dict, with_manual: bool = True) -> dict:
        raise NotImplementedError

    def read_manual(self) -> dict:
        return self._ask([("text",
            "Read this user manual, including its diagrams. List every part of the appliance a user "
            "touches or reads (buttons, switches, the display, the weighing platform or similar), using "
            "the manual's own names, and fill in the fields.")], MANUAL_SCHEMA)

    def align(self, parts: list[dict], clean_crop: np.ndarray, marked_crop: np.ndarray, mark_ids: list[int],
              view: str = "") -> dict:
        return self._ask([
            ("text", f"Photo of the real appliance{' (' + view + ' camera)' if view else ''}:"),
            ("image", clean_crop),
            ("text", "The same photo with candidate regions outlined and numbered. The number sits "
                     "at the centre of a large region and next to a small one, in the outline's colour:"),
            ("image", marked_crop),
            ("text",
                f"Marks present: {mark_ids}.\nParts from the manual:\n{json.dumps(parts, ensure_ascii=False, indent=1)}\n\n"
                "For every part, give the number of the mark that outlines that part, using the manual's diagrams "
                "and the labels visible in the photo. Choose the mark that covers the part itself most tightly, "
                "not a larger region containing it. Answer 0 when no mark is on the part or you cannot tell: "
                "a wrong number makes the robot press the wrong place."),
        ], ALIGN_SCHEMA)

    def plan(self, parts: list[dict], task: str, state: str = "") -> dict:
        return self._ask([("text",
            f"Task: {task}\n"
            f"Current state of the appliance: {state or 'unknown, assume it is switched off'}\n"
            f"Parts: {json.dumps([p['name'] for p in parts], ensure_ascii=False)}\n\n"
            "Plan the task as a sequence of steps, following the manual where it covers the task and the "
            "parts' functions otherwise. Start from the current state: if the display is already on, the "
            "scale is already switched on and pressing ON/OFF would switch it off. One press of a key does "
            "what the manual says it does; never press the same key twice in a row unless the manual "
            "explicitly asks for it. Use as few steps as the task needs. The robot can only "
            "press keys (press, press_and_hold). Putting "
            "things on the weighing pan or taking them off is done by a human helper: use operator_places "
            "or operator_removes with the note saying what. When the task asks for a value, end with a "
            "read_display step. Set feasible to false only if the task needs something none of these cover. "
            "For each step, 'expect' is the steady display state a few seconds after the step, in a few "
            "words (for example '0.00 g', 'off', '0.00 ct'); never a start-up sequence.")], PLAN_SCHEMA)

    def read_display(self, crops: list[np.ndarray]) -> dict:
        content = [("text", "Close-up photos of the appliance's display, taken at the same moment from different cameras:")]
        content += [("image", c) for c in crops]
        content.append(("text", "Read the display. If it is dark or blank, it is off. Report only what is "
                                "actually visible: the characters, and the unit indicator if one is lit."))
        return self._ask(content, DISPLAY_SCHEMA, with_manual=False)

    def check_step(self, step: dict, before: list[np.ndarray], after: list[np.ndarray]) -> dict:
        """Read the display without telling the model what to expect, then compare in a
        separate text-only request: telling a model the expectation first invites it to see it."""
        seen = self.read_display(after)
        observed = (f"{seen['text']} {seen['unit']}".strip() if seen["display_visible"] else "display off or not visible")
        did = (f"{step['operation']} '{step['part']}' x{step['times']}" if step["operation"].startswith("press")
               else f"{step['operation']}: {step.get('note', '')}")
        verdict = self._ask([("text",
            f"A robot operated an appliance. Step done: {did}.\n"
            f"Expected display afterwards: {step['expect']}\n"
            f"Display actually read afterwards: {observed}\n"
            "Does the observed display satisfy the expectation? Minor formatting differences (0.00 vs 0.0, "
            "unit symbol spacing) do not matter; a different unit, a different value, or an off display when "
            "something should be shown does.")], CHECK_SCHEMA, with_manual=False)
        verdict["observed"] = observed
        return verdict


class ClaudeVLM(PromptedVLM):
    def __init__(self, manual_pdf: str | Path, model: str = MODEL):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        data = base64.standard_b64encode(Path(manual_pdf).read_bytes()).decode()
        # The manual leads every request and is cached, so the later calls reread it cheaply.
        self.manual_block = {"type": "document", "title": "User manual",
                             "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                             "cache_control": {"type": "ephemeral"}}
        self.usage = []

    def _ask(self, content: list, schema: dict, with_manual: bool = True) -> dict:
        blocks = [self.manual_block] if with_manual else []
        for kind, item in content:
            blocks.append({"type": "text", "text": item} if kind == "text" else image_block(item))
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": blocks}],
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
            # A declined request is retried server-side on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage.append(resp.usage.to_dict())
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"the model declined the request: {resp.stop_details}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("answer was cut off at max_tokens")
        return json.loads(next(b.text for b in resp.content if b.type == "text"))


def manual_pages(manual_pdf: Path, dpi: int = 110) -> list[np.ndarray]:
    """The manual rendered page by page, for models that cannot read PDFs. Cached next to the PDF."""
    import hashlib
    import subprocess

    manual_pdf = Path(manual_pdf)
    cache = manual_pdf.with_name(f".{manual_pdf.stem}_pages_{dpi}_{hashlib.sha256(manual_pdf.read_bytes()).hexdigest()[:8]}")
    if not cache.exists():
        cache.mkdir()
        subprocess.run(["pdftoppm", "-r", str(dpi), "-jpeg", "-q", str(manual_pdf), str(cache / "p")], check=True)
    return [cv2.imread(str(f)) for f in sorted(cache.glob("p-*.jpg"))]


class LocalVLM(PromptedVLM):
    """A vision-language model served locally through an OpenAI-compatible API (vLLM).
    The manual is sent as page images; answers are constrained to the JSON schema."""

    def __init__(self, manual_pdf: str | Path, model: str = "Qwen3-VL-32B", base_url: str = "http://127.0.0.1:8000/v1",
                 max_tokens: int = 4000):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="local")
        self.model, self.max_tokens = model, max_tokens
        self.pages = manual_pages(manual_pdf, dpi=80)        # ~600 tokens per page in a 10k context
        self.usage = []

    @staticmethod
    def _img(img):
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _jpeg_b64(img, 896)}}

    def _ask(self, content: list, schema: dict, with_manual: bool = True) -> dict:
        parts = []
        if with_manual:
            parts.append({"type": "text", "text": f"User manual, {len(self.pages)} pages:"})
            parts += [self._img(p) for p in self.pages]
        for kind, item in content:
            parts.append({"type": "text", "text": item} if kind == "text" else self._img(item))
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}],
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}},
        )
        if resp.usage:
            self.usage.append({"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens})
        return json.loads(resp.choices[0].message.content)


class ScriptedVLM:
    """Canned answers: {"manual": {...}, "align": {view: {...}} or {...}, "plan": {...},
    "display": {...}, "check": [{...}, ...] or {...}}."""

    def __init__(self, script: dict | str | Path):
        self.script = json.loads(Path(script).read_text()) if not isinstance(script, dict) else script
        self.calls = []
        self._check_i = 0

    def read_manual(self):
        self.calls.append("read_manual")
        return self.script["manual"]

    def align(self, parts, clean_crop, marked_crop, mark_ids, view=""):
        self.calls.append(f"align:{view}")
        a = self.script["align"]
        return a.get(view, a) if "matches" not in a else a

    def plan(self, parts, task, state=""):
        self.calls.append("plan")
        return self.script["plan"]

    def read_display(self, crops):
        self.calls.append("read_display")
        return self.script.get("display", {"display_visible": False, "on": False, "text": "", "unit": ""})

    def check_step(self, step, before, after):
        self.calls.append("check_step")
        c = self.script.get("check", {"achieved": True, "observed": "", "changed": True})
        if isinstance(c, list):
            c = c[min(self._check_i, len(c) - 1)]
            self._check_i += 1
        return c
