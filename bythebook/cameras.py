"""Still capture from the rig cameras (MJPG, by-id paths, any supported mode)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import rig


def grab(cam, size=None, warm: int = 10, average: int = 1) -> np.ndarray:
    """One BGR frame. `cam` is a CameraModel or a role name; `size` (w, h) defaults to
    the mode the CameraModel describes (640x480 for a bare role name).

    The first frames after opening are dropped so auto exposure can settle. With
    average > 1 that many frames are averaged, which helps corner detection on a static scene.
    """
    if isinstance(cam, str):
        path, size = rig.camera_path(cam), size or (640, 480)
    else:
        path, size = cam.path, size or cam.size
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frames, deadline = [], time.time() + 10.0
        n = 0
        while len(frames) < average and time.time() < deadline:
            ok, f = cap.read()
            if not ok:
                continue
            n += 1
            if n > warm:
                frames.append(f)
        if not frames:
            raise RuntimeError(f"no frame from {path}")
        got = (frames[0].shape[1], frames[0].shape[0])
        if got != tuple(size):
            raise RuntimeError(f"{path}: asked for {size}, camera delivered {got}")
        if len(frames) == 1:
            return frames[0]
        return np.mean(np.stack(frames).astype(np.float32), axis=0).round().astype(np.uint8)
    finally:
        cap.release()


def grab_all(cams: dict, **kw) -> dict:
    """Frames from several cameras, one after another (they share one USB 2 bus)."""
    return {role: grab(cam, **kw) for role, cam in cams.items()}
