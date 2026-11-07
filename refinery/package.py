"""4. Pakovanje i slanje (Marko): kanonski format + LeRobot v3 + manifest + zip za kupca.

Sve sto ide kupcu je u runs/<run>/export/. factory_edge/ (sirov video, vault) se NIKAD ne pakuje.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .ingest import sha256_file
from .schema import (action_dex_names, action_grip_names, state_names, write_json)


def export_canonical(run_dir: Path, df: pd.DataFrame, arrays: dict, episodes: list[dict],
                     segments, processed_video: Path) -> Path:
    out = run_dir / "export" / "canonical"
    out.mkdir(parents=True, exist_ok=True)
    tbl = pd.DataFrame({
        "frame_index": df["frame"].to_numpy(),
        "timestamp": df["t"].to_numpy(),
        "observation.state": list(arrays["state"]),
        "action.dexterous": list(arrays["action_dex"]),
        "action.gripper": list(arrays["action_grip"]),
        "hand_valid": list(arrays["hand_valid"]),
        "action_valid": list(arrays["action_valid"]),
        "hand_score_left": df["left_score"].to_numpy(),
        "hand_score_right": df["right_score"].to_numpy(),
    })
    tbl.to_parquet(out / "frames.parquet")
    write_json(out / "episodes.json", episodes)
    write_json(out / "segments.json", segments)
    (out / "features.json").write_text(json.dumps({
        "observation.state": state_names(),
        "action.dexterous": action_dex_names(),
        "action.gripper": action_grip_names(),
        "frames": "Pozicije zgloba su u koordinatama kamere (procena dubine), kp3d lokalno oko sake (m).",
    }, indent=2))
    shutil.copy2(processed_video, out / "ego_anonymized.mp4")
    return out


def export_lerobot(run_dir: Path, anon_video: Path, arrays: dict, episodes: list[dict], fps: float,
                   width: int = 640) -> dict:
    """Pravi LeRobot v3 dataset preko zvanicnog API-ja (lerobot>=0.4: create/add_frame/save_episode/finalize)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": f"lerobot nije instaliran ({e.__class__.__name__})"}
    if not episodes:
        return {"status": "skipped", "reason": "nema epizoda koje su prosle QC"}
    cap = cv2.VideoCapture(str(anon_video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = width
    h = int(round(H * w / W / 2) * 2)
    root = run_dir / "export" / "lerobot"
    if root.exists():
        shutil.rmtree(root)
    S, D, G = arrays["state"].shape[1], arrays["action_dex"].shape[1], arrays["action_grip"].shape[1]
    features = {
        "observation.images.ego": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (S,), "names": state_names()},
        "action": {"dtype": "float32", "shape": (D,), "names": action_dex_names()},
        "action.gripper": {"dtype": "float32", "shape": (G,), "names": action_grip_names()},
        "observation.hand_valid": {"dtype": "float32", "shape": (2,), "names": ["left", "right"]},
    }
    ds = LeRobotDataset.create(repo_id=f"refinery/{run_dir.name}", fps=int(round(fps)), features=features,
                               root=root, robot_type="human_hand_egocentric", use_videos=True)
    for ep in episodes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ep["start_frame"])
        for f in range(ep["start_frame"], ep["end_frame"]):
            ok, fr = cap.read()
            if not ok:
                break
            img = cv2.cvtColor(cv2.resize(fr, (w, h)), cv2.COLOR_BGR2RGB)
            ds.add_frame({
                "observation.images.ego": img,
                "observation.state": arrays["state"][f],
                "action": arrays["action_dex"][f],
                "action.gripper": arrays["action_grip"][f],
                "observation.hand_valid": arrays["hand_valid"][f],
                "task": ep["task"],
            })
        ds.save_episode()
    ds.finalize()
    cap.release()
    return {"status": "ok", "root": str(root), "codebase_version": "v3.0", "n_episodes": len(episodes)}


def bundle(run_dir: Path, manifest) -> Path:
    exp = run_dir / "export"
    files = [p for p in exp.rglob("*") if p.is_file() and not p.name.startswith("delivery_")
             and p != exp / "manifest.json"]
    manifest.exports["files"] = {str(p.relative_to(run_dir)): sha256_file(p) for p in files}
    write_json(run_dir / "manifest.json", manifest)
    shutil.copy2(run_dir / "manifest.json", exp / "manifest.json")
    zpath = exp / f"delivery_{run_dir.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + [exp / "manifest.json"]:
            z.write(p, p.relative_to(exp))
    manifest.exports["delivery_zip"] = str(zpath.relative_to(run_dir))
    manifest.exports["delivery_sha256"] = sha256_file(zpath)
    write_json(run_dir / "manifest.json", manifest)
    return zpath


def label_stats(arrays: dict, episodes: list[dict], segments) -> dict:
    hv = arrays["hand_valid"]
    return {
        "frames_any_hand_pct": float(100 * (hv.max(1) > 0).mean()) if len(hv) else 0.0,
        "frames_both_hands_pct": float(100 * (hv.min(1) > 0).mean()) if len(hv) else 0.0,
        "n_episodes": len(episodes),
        "episode_seconds": float(sum(e["end_s"] - e["start_s"] for e in episodes)),
        "segments_with_language": sum(1 for s in segments if s.subtask),
        "segments_with_outcome": sum(1 for s in segments if s.outcome),
        "segments_with_error": sum(1 for s in segments if s.outcome == "error"),
        "action_layouts": ["dexterous_21kp (138D)", "gripper_wrist6d_aperture (14D)"],
        "state_dim": int(arrays["state"].shape[1]),
        "nan_free": bool(np.isfinite(arrays["state"]).all()),
    }
