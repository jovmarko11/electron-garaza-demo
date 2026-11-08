"""Zajednicki ugovor izmedju modula (Dusan / Zoran / Marko).

Svaki modul cita i pise iste strukture:
  runs/<run>/manifest.json   -> ClipManifest (sve sto se zna o klipu)
  runs/<run>/frames.parquet  -> jedan red po frejmu (QC metrike + ruke)
  runs/<run>/segments.json   -> lista Segment (2 s prozori, QC + labele)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1.0"
SEGMENT_SECONDS = 2.0

# MediaPipe / OpenPose konvencija, 21 tacka po ruci
HAND_JOINTS = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
N_JOINTS = len(HAND_JOINTS)
SIDES = ("left", "right")

# Stanje po ruci: valid(1) + wrist_cam_xyz(3) + wrist_rotvec(3) + aperture(1) + kp3d(63) = 71
PER_HAND_STATE = 1 + 3 + 3 + 1 + N_JOINTS * 3
# Dexterous akcija po ruci: d_wrist_xyz(3) + d_wrist_rotvec(3) + next_kp3d(63) = 69
PER_HAND_ACTION_DEX = 3 + 3 + N_JOINTS * 3
# Gripper akcija po ruci: d_wrist_xyz(3) + d_wrist_rotvec(3) + aperture(1) = 7
PER_HAND_ACTION_GRIP = 3 + 3 + 1


def state_names() -> list[str]:
    names = []
    for s in SIDES:
        names += [f"{s}.valid", f"{s}.wrist_x", f"{s}.wrist_y", f"{s}.wrist_z",
                  f"{s}.wrist_rx", f"{s}.wrist_ry", f"{s}.wrist_rz", f"{s}.aperture"]
        names += [f"{s}.{j}.{a}" for j in HAND_JOINTS for a in "xyz"]
    return names


def action_dex_names() -> list[str]:
    names = []
    for s in SIDES:
        names += [f"{s}.d_wrist_x", f"{s}.d_wrist_y", f"{s}.d_wrist_z",
                  f"{s}.d_wrist_rx", f"{s}.d_wrist_ry", f"{s}.d_wrist_rz"]
        names += [f"{s}.next.{j}.{a}" for j in HAND_JOINTS for a in "xyz"]
    return names


def action_grip_names() -> list[str]:
    names = []
    for s in SIDES:
        names += [f"{s}.d_wrist_x", f"{s}.d_wrist_y", f"{s}.d_wrist_z",
                  f"{s}.d_wrist_rx", f"{s}.d_wrist_ry", f"{s}.d_wrist_rz", f"{s}.aperture"]
    return names


@dataclass
class Segment:
    index: int
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int                      # ekskluzivno
    qc_pass: bool = True
    qc_reason: str | None = None        # prvi razlog odbijanja (redosled kapija)
    qc_flags: dict[str, bool] = field(default_factory=dict)   # sve kapije: True = palo
    qc_metrics: dict[str, float] = field(default_factory=dict)
    borderline: bool = False            # za rucnu trijazu
    human_decision: str | None = None   # "approve" / "reject" iz dashboarda
    subtask: str | None = None          # jezicki opis koraka
    subtask_source: str | None = None   # "annotation" / "vlm" / "template"
    outcome: str | None = None          # "ok" / "error" / "recovery" / None
    error_type: str | None = None


@dataclass
class ClipManifest:
    schema_version: str
    run_id: str
    source_path: str
    source_sha256: str
    worker_pseudonym: str
    fps: float
    width: int
    height: int
    n_frames: int
    duration_s: float
    codec: str | None = None
    intrinsics: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    qc_summary: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    exports: dict[str, Any] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    rights: dict[str, Any] = field(default_factory=lambda: {
        "consent_form_id": None,
        "works_council_approval": None,
        "commercial_use": None,
        "license": "TBD",
        "note": "For demo: public dataset (IndustReal, Apache-2.0) + team recording with consent.",
    })


def write_json(path: Path, obj: Any) -> None:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    elif isinstance(obj, list) and obj and hasattr(obj[0], "__dataclass_fields__"):
        obj = [asdict(o) for o in obj]
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def load_manifest(run_dir: Path) -> ClipManifest:
    return ClipManifest(**json.loads((run_dir / "manifest.json").read_text()))


def load_segments(run_dir: Path) -> list[Segment]:
    return [Segment(**d) for d in json.loads((run_dir / "segments.json").read_text())]
