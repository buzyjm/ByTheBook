"""Record the rig cameras to video files while something happens (demos, debugging).

    with Recorder(["ego", "side_left"], out_dir) as rec:
        ... move the arm ...
    print(rec.files)

One thread per camera. All USB 2 cameras share one bus: three cameras record at 640x480
or 1280x720; a single camera can record at 1920x1080. While recording, a camera cannot be
opened a second time, so `latest` holds the newest frame of each camera for other uses.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from . import rig


class Recorder:
    def __init__(self, roles, out_dir: Path, size=(640, 480), fps: float = 30.0, tag: str = "demo"):
        self.roles, self.out_dir, self.size, self.fps, self.tag = list(roles), Path(out_dir), tuple(size), fps, tag
        self.files, self._threads, self._stop = {}, [], threading.Event()
        self.frames = {}
        self.latest = {}            # role -> most recent frame, for callers that need a picture while recording

    def _run(self, role, path):
        cap = cv2.VideoCapture(rig.camera_path(role), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        writer, n = None, 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self.latest[role] = frame.copy()
                cv2.putText(frame, time.strftime("%H:%M:%S"), (8, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
                n += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.frames[role] = n

    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for role in self.roles:
            path = self.out_dir / f"{self.tag}_{stamp}_{role}.mp4"
            self.files[role] = path
            t = threading.Thread(target=self._run, args=(role, path), daemon=True)
            t.start()
            self._threads.append(t)
            time.sleep(0.3)                     # cameras come up one at a time, as lerobot does it
        time.sleep(1.0)
        return self

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)
        return self.files

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
