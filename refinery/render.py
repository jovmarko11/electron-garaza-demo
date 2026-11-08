"""Vizualizacija za dashboard: skelet ruku + QC traka + podzadatak, i H.264 transkod (browser)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .qc import GATE_LABELS, effective_pass
from .schema import N_JOINTS, SIDES, Segment

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
COLORS = {"left": (255, 160, 0), "right": (0, 200, 255)}


def to_h264(src: Path, dst: Path, max_seconds: float | None = None, strip_audio: bool = True) -> Path:
    if not shutil.which("ffmpeg"):
        shutil.copy2(src, dst)
        return dst
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"]
    cmd += ["-an"] if strip_audio else []
    subprocess.run(cmd + [str(dst)], check=True)
    return dst


def render_overlay(anon_video: Path, df: pd.DataFrame, segments: list[Segment], out: Path) -> Path:
    cap = cv2.VideoCapture(str(anon_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = out.with_suffix(".mp4v.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    seg_of = np.full(len(df), -1)
    for s in segments:
        seg_of[s.start_frame:s.end_frame] = s.index
    i = 0
    while i < len(df):
        ok, fr = cap.read()
        if not ok:
            break
        row = df.iloc[i]
        for side in SIDES:
            if row[f"{side}_valid"] <= 0:
                continue
            kp = np.array(row[f"{side}_kp2d"], float).reshape(N_JOINTS, 2).astype(int)
            for a, b in BONES:
                cv2.line(fr, tuple(kp[a]), tuple(kp[b]), COLORS[side], 2, cv2.LINE_AA)
            for p in kp:
                cv2.circle(fr, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
        si = seg_of[i]
        if si >= 0:
            s = segments[si]
            ok_seg = effective_pass(s)
            col = (60, 180, 60) if ok_seg else (40, 40, 220)
            cv2.rectangle(fr, (0, 0), (W, 34), col, -1)
            txt = "QC OK" if ok_seg else f"REJECTED: {GATE_LABELS.get(s.qc_reason, s.qc_reason)}"
            cv2.putText(fr, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if s.subtask:
                lab = s.subtask + (f"  [{s.outcome}]" if s.outcome else "")
                ocol = (40, 40, 220) if s.outcome == "error" else (255, 255, 255)
                cv2.rectangle(fr, (0, H - 36), (W, H), (0, 0, 0), -1)
                cv2.putText(fr, lab, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ocol, 2, cv2.LINE_AA)
        wr.write(fr)
        i += 1
    cap.release()
    wr.release()
    to_h264(tmp, out)
    tmp.unlink(missing_ok=True)
    return out
