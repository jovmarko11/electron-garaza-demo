"""CLI: python run_pipeline.py --video data/clip.mp4 --annotations data/clip_steps.csv --name demo1"""
import argparse
from pathlib import Path

from refinery.pipeline import run

p = argparse.ArgumentParser(description="Factory egocentric data refinery — demo pipeline")
p.add_argument("--video", required=True, type=Path)
p.add_argument("--name", default=None, help="ime run-a (folder u runs/)")
p.add_argument("--annotations", type=Path, default=None, help="CSV: start_s,end_s,step,outcome,error_type")
p.add_argument("--worker", default="demo_worker")
p.add_argument("--faces", choices=["mediapipe", "haar", "none"], default="mediapipe")
p.add_argument("--hands", choices=["mediapipe", "fake", "none"], default="mediapipe")
p.add_argument("--captions", choices=["template", "mlx"], default="template")
p.add_argument("--hfov", type=float, default=64.0, help="horizontalni FOV kamere (stepeni) ako nema kalibracije")
p.add_argument("--intrinsics", type=Path, default=None, help="JSON sa fx, fy, cx, cy")
p.add_argument("--max-seconds", type=float, default=None)
p.add_argument("--no-lerobot", action="store_true")
a = p.parse_args()

run(a.video, Path("runs") / (a.name or a.video.stem), worker_id=a.worker, annotations=a.annotations,
    faces=a.faces, hands=a.hands, captions=a.captions, hfov_deg=a.hfov, intrinsics=a.intrinsics,
    max_seconds=a.max_seconds, lerobot=not a.no_lerobot)
