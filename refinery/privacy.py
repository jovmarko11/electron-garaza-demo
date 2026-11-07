"""2a. Zastita podataka (Dusan): blur lica (+ hook za tekst/ekrane), bez zvuka.

Backendi za lica:
  - "mediapipe": MediaPipe FaceDetector (Apache-2.0), model blaze_face_short_range.tflite
  - "haar":      OpenCV Haar kaskada (ugradjena, radi offline, slabija)
  - "none":      iskljuceno (samo za testove)
Produkcija: EgoBlur (Meta) — isti interfejs detect(frame_rgb) -> list[box].
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h u pikselima


class FaceDetector:
    name = "none"

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> list[Box]:
        return []


class HaarFaces(FaceDetector):
    name = "opencv-haar-frontalface"

    def __init__(self):
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.clf = cv2.CascadeClassifier(str(path))
        if self.clf.empty():
            raise FileNotFoundError(f"Haar kaskada nije nadjena ({path}). Koristi --faces mediapipe "
                                    "ili opencv-python 4.x (sadrzi cv2/data/*.xml).")

    def detect(self, frame_bgr, ts_ms):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        found = self.clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        return [tuple(int(v) for v in b) for b in found]


class MediaPipeFaces(FaceDetector):
    name = "mediapipe-blazeface-short-range"

    def __init__(self, model_path: Path, min_conf: float = 0.4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opts = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_conf,
        )
        self.det = vision.FaceDetector.create_from_options(opts)

    def detect(self, frame_bgr, ts_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts_ms)
        out = []
        for d in res.detections:
            bb = d.bounding_box
            out.append((int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)))
        return out


def make_face_detector(kind: str, models_dir: Path) -> FaceDetector:
    if kind == "mediapipe":
        p = models_dir / "blaze_face_short_range.tflite"
        if not p.exists():
            raise FileNotFoundError(f"Nema modela {p}. Pokreni: bash scripts/download_models.sh")
        return MediaPipeFaces(p)
    if kind == "haar":
        return HaarFaces()
    return FaceDetector()


class Anonymizer:
    """Blur sa marginom i vremenskim 'drzanjem' (hold) da kratki promasaji detektora ne procure."""

    def __init__(self, detector: FaceDetector, margin: float = 0.25, hold_frames: int = 5):
        self.det = detector
        self.margin = margin
        self.hold = hold_frames
        self._recent: list[tuple[Box, int]] = []   # (box, preostalo frejmova)
        self.stats = {"frames_with_faces": 0, "face_boxes": 0, "held_boxes": 0}

    def _expand(self, b: Box, W: int, H: int) -> Box:
        x, y, w, h = b
        mx, my = int(w * self.margin), int(h * self.margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        return x0, y0, x1 - x0, y1 - y0

    def __call__(self, frame_bgr: np.ndarray, ts_ms: int) -> tuple[np.ndarray, int]:
        H, W = frame_bgr.shape[:2]
        boxes = [self._expand(b, W, H) for b in self.det.detect(frame_bgr, ts_ms)]
        if boxes:
            self.stats["frames_with_faces"] += 1
            self.stats["face_boxes"] += len(boxes)
        held = [(b, k - 1) for b, k in self._recent if k > 1]
        self.stats["held_boxes"] += len(held) if not boxes else 0
        self._recent = [(b, self.hold) for b in boxes] + held
        out = frame_bgr
        for (x, y, w, h), _ in self._recent:
            if w < 2 or h < 2:
                continue
            roi = out[y:y + h, x:x + w]
            k = max(3, (min(w, h) // 3) | 1)
            out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
        return out, len(boxes)
