"""1. Prijem i skladistenje (Dusan): hash, metapodaci, pseudonim radnika."""
from __future__ import annotations

import hashlib
import json
import math
import secrets
import shutil
import subprocess
from pathlib import Path

import cv2

from .schema import SCHEMA_VERSION, ClipManifest


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        return json.loads(out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}


def pseudonymize(worker_id: str, vault: Path) -> str:
    """Mapiranje pravi_ID -> pseudonim ostaje u fabrici (vault), ne ide kupcu."""
    vault.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(vault.read_text()) if vault.exists() else {}
    if worker_id not in table:
        table[worker_id] = f"worker_{secrets.token_hex(3)}"
        vault.write_text(json.dumps(table, indent=2))
    return table[worker_id]


def approx_intrinsics(width: int, height: int, hfov_deg: float) -> dict:
    fx = width / (2 * math.tan(math.radians(hfov_deg) / 2))
    return {"model": "pinhole_approx", "fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2,
            "hfov_deg": hfov_deg, "note": "Estimated from FOV; replace with device calibration."}


def ingest(video: Path, run_dir: Path, worker_id: str, hfov_deg: float,
           intrinsics_json: Path | None = None) -> ClipManifest:
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "factory_edge" / "raw"           # ovo NE napusta fabriku
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = raw_dir / video.name
    if not raw_copy.exists():
        shutil.copy2(video, raw_copy)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Ne mogu da otvorim video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    probe = ffprobe(video)
    vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    codec = vstreams[0].get("codec_name") if vstreams else None

    intr = json.loads(intrinsics_json.read_text()) if intrinsics_json else approx_intrinsics(w, h, hfov_deg)

    m = ClipManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_dir.name,
        source_path=str(video),
        source_sha256=sha256_file(video),
        worker_pseudonym=pseudonymize(worker_id, run_dir.parent / "_factory_vault" / "workers.json"),
        fps=float(fps), width=w, height=h, n_frames=n,
        duration_s=n / fps if fps else 0.0,
        codec=codec,
        intrinsics=intr,
    )
    m.privacy["audio_in_source"] = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    return m
