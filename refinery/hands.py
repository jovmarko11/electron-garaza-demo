"""3D ruke (Marko).

Backendi:
  - "mediapipe": MediaPipe HandLandmarker (Apache-2.0, radi na Mac CPU). 21 tacka, 2D + 3D (metri, oko centra sake).
  - "fake":      sinteticke ruke, samo za testiranje pipeline-a bez modela.
Upgrade: WiLoR / HaMeR (MANO) preko istog interfejsa. Licenca: CC BY-NC-ND / MANO nekomercijalno.

Pozicija zgloba u koordinatama kamere je PROCENA: dubina iz prividne velicine sake (pinhole),
z = fx * L_palm / l_palm_px, L_palm = 0.09 m (zglob -> middle_mcp). Zato svaki frejm nosi confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PALM_LEN_M = 0.09


@dataclass
class HandObs:
    side: str               # "left" / "right" (iz perspektive nosioca kamere)
    score: float
    kp2d: np.ndarray        # (21, 2) pikseli
    kp3d: np.ndarray        # (21, 3) metri, lokalno (oko centra sake)


class HandBackend:
    name = "none"

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> list[HandObs]:
        return []


class MediaPipeHands(HandBackend):
    name = "mediapipe-hand-landmarker"

    def __init__(self, model_path: Path, min_conf: float = 0.4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.det = vision.HandLandmarker.create_from_options(opts)

    def detect(self, frame_bgr, ts_ms):
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.det.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts_ms)
        out = []
        for lm, wlm, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            kp2d = np.array([[p.x * W, p.y * H] for p in lm], dtype=np.float32)
            kp3d = np.array([[p.x, p.y, p.z] for p in wlm], dtype=np.float32)
            # MediaPipe pretpostavlja ogledalnu (selfie) sliku. Egocentric video NIJE ogledalni,
            # pa labelu obrcemo da bi "left" bila leva ruka nosioca kamere. (Proveriti vizuelno, CP1.)
            side = {"left": "right", "right": "left"}[hd[0].category_name.lower()]
            out.append(HandObs(side=side, score=float(hd[0].score), kp2d=kp2d, kp3d=kp3d))
        return out


class FakeHands(HandBackend):
    """Dve ruke koje se krecu po elipsi — za test bez modela."""
    name = "fake"

    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        rng = np.random.default_rng(0)
        self.template = (rng.normal(size=(21, 3)) * 0.02).astype(np.float32)
        self.template[0] = 0

    def detect(self, frame_bgr, ts_ms):
        t = ts_ms / 1000.0
        out = []
        for i, side in enumerate(("left", "right")):
            if (int(t) + i) % 7 == 0:          # povremeno nestane ruka
                continue
            cx = self.W * (0.35 + 0.3 * i) + 60 * np.cos(t + i)
            cy = self.H * 0.65 + 30 * np.sin(1.3 * t)
            kp2d = np.stack([cx + self.template[:, 0] * 2500, cy + self.template[:, 1] * 2500], 1)
            kp2d[9] = [cx, cy - 70]            # middle_mcp za procenu velicine sake
            kp2d[0] = [cx, cy]
            kp3d = self.template.copy()
            kp3d[4] = [0.03 + 0.02 * np.sin(2 * t), 0, 0]
            kp3d[8] = [-0.03, -0.02, 0]
            out.append(HandObs(side=side, score=0.9, kp2d=kp2d.astype(np.float32), kp3d=kp3d))
        return out


def make_hand_backend(kind: str, models_dir: Path, W: int, H: int) -> HandBackend:
    if kind == "mediapipe":
        p = models_dir / "hand_landmarker.task"
        if not p.exists():
            raise FileNotFoundError(f"Nema modela {p}. Pokreni: bash scripts/download_models.sh")
        return MediaPipeHands(p)
    if kind == "fake":
        return FakeHands(W, H)
    return HandBackend()


# ---------- geometrija: od opservacije do stanja ruke ----------

def _rotvec_from_basis(kp3d: np.ndarray) -> np.ndarray:
    """Orijentacija sake iz zgloba (0), index_mcp (5), pinky_mcp (17)."""
    w, i, p = kp3d[0], kp3d[5], kp3d[17]
    x = i - w
    x = x / (np.linalg.norm(x) + 1e-9)
    tmp = p - w
    z = np.cross(x, tmp)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1).astype(np.float64)
    rv, _ = cv2.Rodrigues(R)
    return rv.reshape(3).astype(np.float32)


def hand_state(obs: HandObs, intr: dict) -> dict:
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    palm_px = float(np.linalg.norm(obs.kp2d[9] - obs.kp2d[0]))
    z = fx * PALM_LEN_M / max(palm_px, 1.0)
    u, v = obs.kp2d[0]
    wrist_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float32)
    aperture = float(np.linalg.norm(obs.kp3d[4] - obs.kp3d[8]))   # palac -> kaziprst (m)
    return {
        "wrist_cam": wrist_cam,
        "wrist_rotvec": _rotvec_from_basis(obs.kp3d),
        "aperture": aperture,
        "kp3d": obs.kp3d,
        "palm_px": palm_px,
    }
