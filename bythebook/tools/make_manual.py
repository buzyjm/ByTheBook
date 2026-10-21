"""Build a user manual PDF for the appliance in the style of real appliance manuals
(and of CheckManual's generated ones): cover photo, overview and control-panel diagrams
with point-line part annotations, a key-function table, specifications, numbered
operating procedures, an error table and precautions.

    python -m bythebook.tools.make_manual appliances/leadzm_scale/leadzm_scale.json --out appliances/leadzm_scale/leadzm_scale_manual.pdf
    python -m bythebook.tools.make_manual ... --no-figures      # text-only variant for ablations

No LaTeX needed: pages are laid out with matplotlib on A4.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

A4 = (8.27, 11.69)
MARGIN = 0.8                       # inches
FONT = "DejaVu Sans"
BLUE = "#1f5fa8"


class Page:
    """A4 page with a cursor that flows downwards. Coordinates in inches from the top-left."""

    def __init__(self, pdf, title, page_no, footer):
        self.pdf, self.page_no = pdf, page_no
        self.fig = plt.figure(figsize=A4)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, A4[0])
        self.ax.set_ylim(A4[1], 0)
        self.ax.set_axis_off()
        self.y = MARGIN
        if title:
            self.ax.text(MARGIN, 0.45, title, fontsize=8.5, color="grey", family=FONT, va="center")
            self.ax.plot([MARGIN, A4[0] - MARGIN], [0.6, 0.6], color="grey", lw=0.5)
        self.ax.text(A4[0] / 2, A4[1] - 0.45, f"{footer}   |   {page_no}", fontsize=8, color="grey",
                     family=FONT, ha="center", va="center")

    @property
    def width(self):
        return A4[0] - 2 * MARGIN

    def close(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def heading(self, text, level=1):
        size = {0: 22, 1: 14, 2: 11.5}[level]
        self.y += {0: 0.3, 1: 0.28, 2: 0.18}[level]
        self.ax.text(MARGIN, self.y, text, fontsize=size, weight="bold", color=BLUE if level else "black",
                     family=FONT, va="top")
        self.y += size / 72 * 1.5 + 0.06
        if level == 1:
            self.ax.plot([MARGIN, A4[0] - MARGIN], [self.y - 0.04, self.y - 0.04], color=BLUE, lw=0.8)
            self.y += 0.08

    def paragraph(self, text, size=10, indent=0.0, bullet=None, width_chars=None):
        wrap = width_chars or int((self.width - indent) * 72 / (size * 0.52))
        lines = textwrap.wrap(text, wrap)
        for i, line in enumerate(lines):
            if bullet is not None and i == 0:
                self.ax.text(MARGIN + indent - 0.28, self.y, bullet, fontsize=size, family=FONT, va="top")
            self.ax.text(MARGIN + indent, self.y, line, fontsize=size, family=FONT, va="top")
            self.y += size / 72 * 1.45
        self.y += 0.06

    def numbered(self, items, size=10):
        for i, item in enumerate(items, 1):
            self.paragraph(item, size=size, indent=0.35, bullet=f"{i}.")

    def bullets(self, items, size=10):
        for item in items:
            self.paragraph(item, size=size, indent=0.35, bullet="\u2022")

    def table(self, rows, col_widths, header=None, size=9.5):
        x0 = MARGIN
        all_rows = ([header] if header else []) + rows
        for r, row in enumerate(all_rows):
            is_header = header is not None and r == 0
            wraps = [textwrap.wrap(str(c), int(w * 72 / (size * 0.55))) or [""] for c, w in zip(row, col_widths)]
            h = max(len(w) for w in wraps) * size / 72 * 1.4 + 0.1
            if is_header or r % 2 == (0 if header else 1):
                self.ax.add_patch(plt.Rectangle((x0, self.y), sum(col_widths), h, color="#dde7f3" if is_header else "#f4f4f4", lw=0))
            x = x0
            for lines, w in zip(wraps, col_widths):
                for k, line in enumerate(lines):
                    self.ax.text(x + 0.06, self.y + 0.06 + k * size / 72 * 1.4, line, fontsize=size, family=FONT,
                                 va="top", weight="bold" if is_header else "normal")
                x += w
            self.y += h
        self.ax.plot([x0, x0 + sum(col_widths)], [self.y, self.y], color="grey", lw=0.5)
        self.y += 0.15

    def image(self, path, width, annotations=None, caption=None, leader_color="#c8102e"):
        """Image scaled to `width` inches, kept at the current cursor. annotations: list of
        dicts with xy (pixel on the part), label_xy (pixel where the number box goes), id."""
        img = plt.imread(str(path))
        h_px, w_px = img.shape[:2]
        height = width * h_px / w_px
        x0 = MARGIN + (self.width - width) / 2
        ext = (x0, x0 + width, self.y + height, self.y)
        self.ax.imshow(img, extent=ext, aspect="auto", zorder=1)
        sx = width / w_px
        for a in annotations or []:
            px, py = a["xy"]
            lx, ly = a["label_xy"]
            X, Y = x0 + px * sx, self.y + py * sx
            LX, LY = x0 + lx * sx, self.y + ly * sx
            self.ax.plot([X, LX], [Y, LY], color=leader_color, lw=1.2, zorder=3)
            self.ax.plot([X], [Y], marker="o", ms=4, color=leader_color, zorder=3)
            self.ax.add_patch(FancyBboxPatch((LX - 0.13, LY - 0.11), 0.26, 0.22, boxstyle="round,pad=0.02",
                                             fc="white", ec=leader_color, lw=1.2, zorder=4))
            self.ax.text(LX, LY, str(a["id"]), fontsize=9, weight="bold", ha="center", va="center", zorder=5, family=FONT)
        self.y += height + 0.08
        if caption:
            self.ax.text(A4[0] / 2, self.y, caption, fontsize=9, style="italic", ha="center", va="top", family=FONT)
            self.y += 0.3


def build(spec_path: Path, out: Path, figures: bool = True):
    spec = json.loads(spec_path.read_text())
    root = spec_path.parent
    footer = f"{spec['title']} - {spec['subtitle']}"
    with PdfPages(str(out)) as pdf:
        # ------------------------------------------------------------ cover
        p = Page(pdf, None, 1, footer)
        p.y = 1.3
        p.ax.text(MARGIN, p.y, spec["title"], fontsize=26, weight="bold", color=BLUE, family=FONT, va="top")
        p.y += 0.6
        p.ax.text(MARGIN, p.y, spec["subtitle"], fontsize=18, family=FONT, va="top")
        p.y += 0.45
        p.ax.text(MARGIN, p.y, spec["model"], fontsize=11, color="grey", family=FONT, va="top")
        p.y += 0.8
        if figures:
            p.image(root / spec["photos"]["cover"], 5.2)
        p.y = max(p.y, 9.2)
        p.paragraph("Please read this manual carefully before use and keep it for future reference.", size=10)
        p.paragraph("The scale has been calibrated before leaving the factory. Do not start the calibration unless a "
                    "reference weight of the correct value is available.", size=10)
        p.close()

        # ------------------------------------------------------------ overview
        p = Page(pdf, "1  Overview", 2, footer)
        p.heading("1  Overview", 1)
        p.paragraph("This precision scale weighs up to 3000 g in steps of 0.01 g, converts between eleven units, "
                    "counts pieces and weighs in percent of a reference sample. Figure 1 shows the main parts; "
                    "Figure 2 shows the control panel.")
        if figures:
            p.image(root / spec["photos"]["overview"], 3.3, spec["overview_parts"], "Figure 1. Main parts.")
        p.table([[a["id"], a["name"]] for a in spec["overview_parts"]], [0.6, 4.0], header=["Ref.", "Part"])
        p.close()

        p = Page(pdf, "2  Control panel", 3, footer)
        p.heading("2  Control panel and keys", 1)
        if figures:
            p.image(root / spec["photos"]["panel"], 5.6, spec["panel_parts"], "Figure 2. Control panel.")
        p.table([[a["id"], a["label"] or a["name"], a["function"]] for a in spec["panel_parts"]], [0.5, 1.5, 4.6],
                header=["No.", "Key / part", "Function"])
        p.paragraph("The display shows the weight with the active unit on its right-hand side (g, ct, oz, ozt, lb, "
                    "TOL, dwt, mm, kg, pcs or %). A small circle in the upper left corner is the stability mark; "
                    "the battery symbol in the lower right corner shows the battery state.")
        p.close()

        # ------------------------------------------------------------ specs and basic use
        p = Page(pdf, "3  Specifications and basic operation", 4, footer)
        p.heading("3  Specifications", 1)
        p.table(spec["specs"], [2.4, 4.2])
        p.heading("4  Preparation", 1)
        p.bullets([
            "Place the scale on a stable, level surface away from draughts, vibration, direct sunlight and strong "
            "electromagnetic fields.",
            "Connect the power adapter, or insert batteries, and switch on. Let the scale warm up for a few minutes "
            "before precise measurements.",
            "Keep the pan clean and do not exceed the maximum capacity, container included.",
        ])
        p.heading("5  Switching on and off", 1)
        p.numbered([
            "Press the ON/OFF key. The display shows all segments (8.8.8.8.8), then the maximum weighing value "
            "(3000), the software version (u-1908) and a row of dashes (-----) while the sensor settles.",
            "When the display shows 0.00 g the scale is ready. The unit after switching on is always grams.",
            "To switch off, press the ON/OFF key again. The scale does not switch off by itself.",
        ])
        p.heading("6  Weighing", 1)
        p.numbered([
            "Make sure the display shows 0.00. If not, press the TARE/ZERO key with the pan empty.",
            "Place the object on the centre of the pan and read the weight once the stability mark is shown.",
        ])
        p.close()

        # ------------------------------------------------------------ units, counting, percent, calibration
        p = Page(pdf, "7  Tare, units, counting, percentage", 5, footer)
        p.heading("7  Tare: weighing in a container", 1)
        p.numbered([
            "Place the empty container on the pan. Its weight is displayed.",
            "Press the TARE/ZERO key. The display shows 0.00: the container weight is subtracted.",
            "Put the material into the container. The display shows the net weight.",
            "Pressing TARE/ZERO with the pan empty returns the display to 0.00.",
        ])
        p.heading("8  Changing the unit", 1)
        p.paragraph("Press the UNIT key to step through the units. Each press moves one step along the sequence "
                    "below and the unit indicator on the right of the display changes accordingly; after the "
                    "last unit the scale returns to grams.")
        p.paragraph("  \u2192  ".join(spec["units"]) + "  \u2192  g", size=10.5, indent=0.3)
        p.bullets([
            "TOL is the tola, dwt the pennyweight, mm the momme.",
            "The % step is the percentage mode (section 10). Without a reference sample the display shows "
            "Err SP for a moment and then %.",
            "The unit returns to grams every time the scale is switched on.",
        ])
        p.heading("9  Counting pieces (PCS)", 1)
        p.numbered([
            "Place the empty container, if one is used, on the pan and press TARE/ZERO so that the display shows 0.00.",
            "Press the PCS key. The display shows -CoU- briefly, then a flashing 10, and the unit indicator changes to pcs.",
            "Put 10 pieces of the parts to be counted on the pan and press the PCS key again. The display shows "
            "dashes while the average piece weight is determined, then 10.",
            "Add the remaining pieces. The display shows the number of pieces.",
            "To leave the counting mode press the PCS key once more. The display shows ----- and returns to grams. "
            "The UNIT key does not leave the counting mode.",
        ])
        p.paragraph("If the PCS key is pressed to confirm while the pan is empty, the display shows Err-SP and the scale "
                    "enters the counting mode without a reference; press PCS again to return to grams. The weight of "
                    "one piece must be larger than the resolution of the scale, 0.01 g.", size=9.5)
        p.heading("10  Percentage weighing", 1)
        p.numbered([
            "Place the reference sample on the pan and tare if a container is used.",
            "Press the UNIT key until % is shown. The sample is taken as 100 %.",
            "Replace the sample with the material to be compared; the display shows its weight in percent.",
        ])
        p.close()

        # ------------------------------------------------------------ errors and care
        p = Page(pdf, "11  Calibration, error messages and care", 6, footer)
        p.heading("11  Calibration (CAL)", 1)
        p.paragraph("Calibrate only with a 2000 g reference weight, after a warm-up of 30 minutes.")
        p.numbered([
            "With the pan empty, press and hold the CAL key for about 3 seconds until -CAL- is shown, then release.",
            "The display flashes the value of the reference weight (2000). Place the 2000 g weight on the pan.",
            "The display shows dashes, then the weight value. Remove the weight; the display shows dashes again, "
            "then 0.00. The calibration is complete.",
            "If the readings are still inaccurate, repeat the procedure.",
        ])
        p.heading("12  Error messages", 1)
        p.table(spec["errors"], [1.9, 2.3, 2.4], header=["Display", "Meaning", "What to do"])
        p.heading("13  Precautions", 1)
        p.bullets([
            "Never exceed 3000 g on the pan, container included. The overload warning is a row of dashes.",
            "Press the keys gently and one at a time. Do not press on the pan with a hard object.",
            "To remove the round pan, turn it clockwise and lift. Do not pull it straight up.",
            "Keep the scale dry. Clean it with a soft, slightly damp cloth; do not use solvents.",
            "Wait at least 3 seconds between switching off and on again.",
            "Do not press the CAL key without a reference weight; the factory calibration would be lost.",
            "Store the scale in its original packaging, with the batteries removed, when it is not used for a long time.",
        ])
        p.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    out = a.out or a.spec.with_name(a.spec.stem + ("_manual_text_only.pdf" if a.no_figures else "_manual.pdf"))
    build(a.spec, out, figures=not a.no_figures)
    print("wrote", out)


if __name__ == "__main__":
    main()
