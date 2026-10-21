"""Do the cameras still agree with their calibration?  Read-only, no arm involved.

    python -m bythebook.tools.check_rig

Needs a rig-calibrate board lying flat on the table, anywhere in view of at least two
cameras. Reports, at 640x480 and in the high-resolution mode:
  - reprojection error of triangulated board corners (cameras consistent with each other)
  - height of the corners above the table plane (table frame still on the table)
  - measured vs printed square size (metric scale)
"""
from __future__ import annotations

import json

from .. import rig
from ..cameras import grab_all
from ..targets import measure_board


def main():
    for hires in (False, True):
        cams = {r: c for r, c in rig.load_cameras(hires=hires).items() if c.T_table_cam is not None}
        if hires and not rig.HIRES_FILE.exists():
            print("no hires.json yet: run python -m bythebook.tools.probe_hires")
            continue
        images = grab_all(cams, average=3)
        _, report, _ = measure_board(cams, images)
        print(("high-resolution " if hires else "calibrated 640x480 ") + "mode")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
