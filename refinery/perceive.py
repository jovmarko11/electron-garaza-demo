"""Jedan prolaz kroz video: anonimizacija -> metrike po frejmu -> ruke.

Izlaz: frames.parquet (jedan red po frejmu) + factory_edge/anonymized_raw.mp4 (mp4v, privremeno)
Metrike su "jeftine" namerno (lekcija Scale AI: jeftine CV metrike > VLM za ego QC).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .hands import HandBackend, hand_state
from .privacy import Anonymizer
from .schema import N_JOINTS, SIDES


def _sharpness(gray: np.ndarray) -> float:
    """Varijansa Laplasijana normalizovana kontrastom -> manje zavisi od osvetljenosti."""
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var() / (gray.astype(np.float32).var() + 1e-6))


def perceive(video: Path, run_dir: Path, manifest, anonymizer: Anonymizer, hands: HandBackend,
             max_seconds: float | None = None, progress=None) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video))
    fps, W, H = manifest.fps, manifest.width, manifest.height
    n_total = manifest.n_frames
    if max_seconds:
        n_total = min(n_total, int(max_seconds * fps))

    out_path = run_dir / "factory_edge" / "anonymized_raw.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rows = []
    prev_small = None
    margin = 0.02
    i = 0
    while i < n_total:
        ok, frame = cap.read()
        if not ok:
            break
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts_ms = int(round(i * 1000.0 / fps))      # monotono, potrebno MediaPipe VIDEO modu

        frame, n_faces = anonymizer(frame, ts_ms)
        writer.write(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (256, max(1, int(256 * H / W))))
        b, g, r = [float(c.mean()) for c in cv2.split(frame)]
        row = {
            "frame": i,
            "t": i / fps,
            "pts_ms": pts_ms,
            "faces": n_faces,
            "brightness": float(gray.mean()),
            "overexposed": float((gray > 250).mean()),
            "underexposed": float((gray < 5).mean()),
            "green_artifact": float(g - (r + b) / 2),
            "sharpness": _sharpness(small),
        }
        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_small, small, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            row["flow_dx"] = float(np.median(flow[..., 0]))
            row["flow_dy"] = float(np.median(flow[..., 1]))
            row["frame_diff"] = float(np.abs(small.astype(np.int16) - prev_small).mean())
        else:
            row["flow_dx"] = row["flow_dy"] = row["frame_diff"] = 0.0
        prev_small = small

        obs = {o.side: o for o in sorted(hands.detect(frame, ts_ms), key=lambda o: -o.score)}
        for side in SIDES:
            o = obs.get(side)
            if o is None:
                row[f"{side}_valid"] = 0.0
                row[f"{side}_score"] = 0.0
                row[f"{side}_truncated"] = 0.0
                row[f"{side}_kp2d"] = [np.nan] * (N_JOINTS * 2)
                row[f"{side}_kp3d"] = [np.nan] * (N_JOINTS * 3)
                row[f"{side}_wrist_cam"] = [np.nan] * 3
                row[f"{side}_wrist_rotvec"] = [np.nan] * 3
                row[f"{side}_aperture"] = np.nan
                continue
            st = hand_state(o, manifest.intrinsics)
            u, v = o.kp2d[:, 0] / W, o.kp2d[:, 1] / H
            truncated = bool(((u < margin) | (u > 1 - margin) | (v < margin) | (v > 1 - margin)).any())
            row[f"{side}_valid"] = 1.0
            row[f"{side}_score"] = o.score
            row[f"{side}_truncated"] = float(truncated)
            row[f"{side}_kp2d"] = o.kp2d.reshape(-1).tolist()
            row[f"{side}_kp3d"] = st["kp3d"].reshape(-1).tolist()
            row[f"{side}_wrist_cam"] = st["wrist_cam"].tolist()
            row[f"{side}_wrist_rotvec"] = st["wrist_rotvec"].tolist()
            row[f"{side}_aperture"] = st["aperture"]
        rows.append(row)
        i += 1
        if progress and i % 10 == 0:
            progress(i / max(n_total, 1))

    cap.release()
    writer.release()
    df = pd.DataFrame(rows)
    # Integritet: pts skokovi (dropovani frejmovi) i monotonost
    dt = df["pts_ms"].diff().fillna(1000.0 / fps)
    df["pts_gap"] = (dt > 1.5 * 1000.0 / fps).astype(float)
    df["pts_backwards"] = (dt < 0).astype(float)
    df.to_parquet(run_dir / "frames.parquet")
    manifest.privacy.update({
        "face_detector": anonymizer.det.name,
        **anonymizer.stats,
        "audio_removed": True,
        "text_screen_blur": "TODO (OCR hook)",
        "raw_video_location": "factory_edge/ (ne izlazi iz fabrike)",
    })
    manifest.models["hands"] = hands.name
    return df
