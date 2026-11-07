"""Orkestracija: ingest -> privacy+perceive -> qc -> label -> package (+ overlay za dashboard)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import label as L
from .hands import make_hand_backend
from .ingest import ingest
from .package import bundle, export_canonical, export_lerobot, label_stats
from .perceive import perceive
from .privacy import Anonymizer, make_face_detector
from .qc import gate
from .render import render_overlay, to_h264
from .schema import load_manifest, load_segments, write_json

MODELS = Path(__file__).resolve().parent.parent / "models"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(video: Path, run_dir: Path, worker_id: str = "demo_worker", annotations: Path | None = None,
        faces: str = "mediapipe", hands: str = "mediapipe", captions: str = "template",
        hfov_deg: float = 64.0, intrinsics: Path | None = None, max_seconds: float | None = None,
        qc_cfg: dict | None = None, lerobot: bool = True) -> Path:
    t0 = time.time()
    log("1/5 ingest")
    m = ingest(video, run_dir, worker_id, hfov_deg, intrinsics)
    if max_seconds:
        m.n_frames = min(m.n_frames, int(max_seconds * m.fps))
        m.duration_s = m.n_frames / m.fps
    write_json(run_dir / "manifest.json", m)

    log("2/5 privacy + percepcija (ruke, metrike)")
    anon = Anonymizer(make_face_detector(faces, MODELS))
    hb = make_hand_backend(hands, MODELS, m.width, m.height)
    df = perceive(video, run_dir, m, anon, hb, max_seconds)
    m.n_frames = len(df)
    m.duration_s = len(df) / m.fps

    # Pregled "pre" (samo interno u fabrici) i anonimizovan H.264
    to_h264(video, run_dir / "factory_edge" / "raw_preview.mp4", max_seconds=m.duration_s)
    to_h264(run_dir / "factory_edge" / "anonymized_raw.mp4", run_dir / "anonymized.mp4")

    log("3/5 QC kapije")
    segments, summary = gate(df, m.fps, m.width, qc_cfg)
    m.qc_summary = summary
    write_json(run_dir / "segments.json", segments)
    write_json(run_dir / "manifest.json", m)

    finish(run_dir, df=df, segments=segments, annotations=annotations, captions=captions, lerobot=lerobot)
    log(f"gotovo za {time.time() - t0:.1f} s -> {run_dir}")
    return run_dir


def finish(run_dir: Path, df: pd.DataFrame | None = None, segments=None, annotations: Path | None = None,
           captions: str = "template", lerobot: bool = True, keep_existing_language: bool = False) -> None:
    """Labelovanje + pakovanje. Poziva se i iz dashboarda posle promene QC pragova / rucne trijaze."""
    m = load_manifest(run_dir)
    df = df if df is not None else pd.read_parquet(run_dir / "frames.parquet")
    segments = segments if segments is not None else load_segments(run_dir)
    anon_video = run_dir / "factory_edge" / "anonymized_raw.mp4"

    log("4/5 labelovanje")
    ann_path = annotations or (Path(m.labels["annotations"]) if m.labels.get("annotations") else None)
    L.assign_annotations(segments, L.load_annotations(ann_path))
    if not keep_existing_language:
        cap = L.make_captioner(captions)
        L.caption_segments(anon_video, df, segments, cap, run_dir)
        m.models["captions"] = cap.name
    arrays = L.state_action_arrays(df)
    episodes = L.episodes_from_segments(segments)
    m.labels.update(label_stats(arrays, episodes, segments))
    m.labels["annotations"] = str(ann_path) if ann_path else None
    write_json(run_dir / "segments.json", segments)
    (run_dir / "episodes.json").write_text(json.dumps(episodes, indent=2, ensure_ascii=False))

    render_overlay(anon_video, df, segments, run_dir / "overlay.mp4")

    log("5/5 pakovanje")
    export_canonical(run_dir, df, arrays, episodes, segments, run_dir / "anonymized.mp4")
    m.exports["lerobot"] = export_lerobot(run_dir, anon_video, arrays, episodes, m.fps) if lerobot \
        else {"status": "skipped", "reason": "--no-lerobot"}
    write_json(run_dir / "manifest.json", m)
    bundle(run_dir, m)
