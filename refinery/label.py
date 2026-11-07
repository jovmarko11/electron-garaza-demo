"""3. Labelovanje (Marko): stanje/akcije ruku, podzadaci (jezik), ishod/greske, epizode.

Anotacije ishoda (simulirani QC fabrike / MES) ulaze kao CSV:
    start_s,end_s,step,outcome,error_type
    0.0,12.4,attach front wheel,ok,
    12.4,20.1,insert bolt,error,omission
outcome: ok | error | recovery.  Za IndustReal: scripts/industreal_to_csv.py (prilagoditi stvarnim fajlovima).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .qc import effective_pass
from .schema import (N_JOINTS, PER_HAND_ACTION_DEX, PER_HAND_ACTION_GRIP, PER_HAND_STATE, SIDES,
                     Segment)

VERBS = ["pick", "place", "insert", "screw", "unscrew", "push", "pull", "align", "hold",
         "rotate", "attach", "detach", "inspect", "transfer", "press"]


# ---------------- ishod / anotacije ----------------

def load_annotations(csv_path: Path | None) -> pd.DataFrame | None:
    if not csv_path:
        return None
    a = pd.read_csv(csv_path)
    need = {"start_s", "end_s", "step", "outcome"}
    if not need.issubset(a.columns):
        raise ValueError(f"Anotacije moraju imati kolone {need}, a imaju {set(a.columns)}")
    if "error_type" not in a.columns:
        a["error_type"] = None
    return a


def assign_annotations(segments: list[Segment], ann: pd.DataFrame | None) -> None:
    if ann is None:
        return
    for s in segments:
        ov = (np.minimum(ann["end_s"].to_numpy(), s.end_s)
              - np.maximum(ann["start_s"].to_numpy(), s.start_s)).clip(min=0)
        if ov.max() <= 0:
            continue
        r = ann.iloc[int(np.argmax(ov))]
        s.subtask, s.subtask_source = str(r["step"]), "annotation"
        s.outcome = str(r["outcome"])
        s.error_type = None if pd.isna(r["error_type"]) else str(r["error_type"])


# ---------------- jezik: VLM na kolazu isecaka oko ruku ----------------

def _hand_crop(frame: np.ndarray, row: pd.Series, pad: float = 0.6) -> np.ndarray:
    H, W = frame.shape[:2]
    pts = []
    for side in SIDES:
        if row[f"{side}_valid"] > 0:
            pts.append(np.array(row[f"{side}_kp2d"], dtype=float).reshape(N_JOINTS, 2))
    if not pts:
        return cv2.resize(frame, (320, 240))
    p = np.concatenate(pts)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 - pad * w)), int(min(W, x1 + pad * w))
    y0, y1 = int(max(0, y0 - pad * h)), int(min(H, y1 + pad * h))
    crop = frame[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]
    return cv2.resize(crop, (320, 240))


def build_collage(video: Path, df: pd.DataFrame, seg: Segment, out_path: Path) -> Path:
    """Pre / tokom / posle (lekcija Scale AI: kolaz oko ruku > ceo video)."""
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for f in np.linspace(seg.start_frame, seg.end_frame - 1, 3).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if ok:
            tiles.append(_hand_crop(fr, df.iloc[int(f)]))
    cap.release()
    if not tiles:
        return out_path
    cv2.imwrite(str(out_path), np.hstack(tiles))
    return out_path


class Captioner:
    name = "template"

    def caption(self, collage: Path, seg: Segment, df: pd.DataFrame) -> str:
        sl = df.iloc[seg.start_frame:seg.end_frame]
        l, r = sl["left_valid"].mean() > 0.5, sl["right_valid"].mean() > 0.5
        hands = "both hands" if (l and r) else ("left hand" if l else "right hand")
        return f"manipulate object with {hands}"


class MLXCaptioner(Captioner):
    """Qwen2.5-VL preko mlx-vlm (Apple Silicon)."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)
        self.name = f"mlx-vlm:{model_id}"

    def caption(self, collage, seg, df):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = (
            "The image shows three crops (before, during, after) of a worker's hands in a first-person "
            "factory video. Describe the single manipulation step in the middle crop. Answer with ONE short "
            "lowercase phrase in the form '<verb> <object>' or '<verb> <object> to/into/onto <destination>'. "
            "Add a destination only if a clear place or container is visible (a tray, box, table, fixture); "
            "never use hands or a person as destination. Do not use brackets, quotes or punctuation. "
            f"Use one verb from: {', '.join(VERBS)}."
        )
        formatted = apply_chat_template(self.processor, self.config, prompt, num_images=1)
        out = generate(self.model, self.processor, formatted, image=[str(collage)], max_tokens=24, verbose=False)
        text = getattr(out, "text", out)
        return str(text).strip().split("\n")[0].strip(" .'\"").lower()


def make_captioner(kind: str) -> Captioner:
    return MLXCaptioner() if kind == "mlx" else Captioner()


def caption_segments(video: Path, df: pd.DataFrame, segments: list[Segment], cap: Captioner,
                     run_dir: Path, progress=None) -> None:
    cdir = run_dir / "collages"
    cdir.mkdir(exist_ok=True)
    todo = [s for s in segments if effective_pass(s) and s.subtask_source != "annotation"]
    for k, s in enumerate(todo):
        col = build_collage(video, df, s, cdir / f"seg_{s.index:04d}.jpg")
        s.subtask = cap.caption(col, s, df)
        s.subtask_source = "vlm" if isinstance(cap, MLXCaptioner) else "template"
        if progress:
            progress((k + 1) / len(todo))


# ---------------- stanje / akcije ----------------

def _rotvec_rel(r_now: np.ndarray, r_next: np.ndarray) -> np.ndarray:
    R0, _ = cv2.Rodrigues(r_now.astype(np.float64).reshape(3, 1))
    R1, _ = cv2.Rodrigues(r_next.astype(np.float64).reshape(3, 1))
    rv, _ = cv2.Rodrigues(R1 @ R0.T)
    return rv.reshape(3)


def state_action_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(df)
    state = np.zeros((n, PER_HAND_STATE * 2), np.float32)
    a_dex = np.zeros((n, PER_HAND_ACTION_DEX * 2), np.float32)
    a_grip = np.zeros((n, PER_HAND_ACTION_GRIP * 2), np.float32)
    valid = np.zeros((n, 2), np.float32)
    for h, side in enumerate(SIDES):
        v = df[f"{side}_valid"].to_numpy() > 0
        wc = np.array([x if v[i] else [0, 0, 0] for i, x in enumerate(df[f"{side}_wrist_cam"])], np.float32)
        rv = np.array([x if v[i] else [0, 0, 0] for i, x in enumerate(df[f"{side}_wrist_rotvec"])], np.float32)
        kp = np.array([x if v[i] else [0] * (N_JOINTS * 3) for i, x in enumerate(df[f"{side}_kp3d"])], np.float32)
        ap = np.nan_to_num(df[f"{side}_aperture"].to_numpy(np.float32))
        valid[:, h] = v
        o = h * PER_HAND_STATE
        state[:, o] = v
        state[:, o + 1:o + 4] = wc
        state[:, o + 4:o + 7] = rv
        state[:, o + 7] = ap
        state[:, o + 8:o + PER_HAND_STATE] = kp
        od, og = h * PER_HAND_ACTION_DEX, h * PER_HAND_ACTION_GRIP
        for i in range(n - 1):
            if not (v[i] and v[i + 1]):
                continue
            dxyz = wc[i + 1] - wc[i]
            drot = _rotvec_rel(rv[i], rv[i + 1])
            a_dex[i, od:od + 3] = dxyz
            a_dex[i, od + 3:od + 6] = drot
            a_dex[i, od + 6:od + PER_HAND_ACTION_DEX] = kp[i + 1]
            a_grip[i, og:og + 3] = dxyz
            a_grip[i, og + 3:og + 6] = drot
            a_grip[i, og + 6] = ap[i + 1]
    # akcija je validna samo ako je ruka vidljiva i u sledecem frejmu
    action_valid = np.zeros((n, 2), np.float32)
    action_valid[:-1] = valid[:-1] * valid[1:]
    return {"state": state, "action_dex": a_dex, "action_grip": a_grip,
            "hand_valid": valid, "action_valid": action_valid}


def episodes_from_segments(segments: list[Segment], min_segments: int = 2) -> list[dict]:
    """Epizoda = neprekinut niz prihvacenih segmenata (min 4 s, kao ACE-Ego-0)."""
    eps, cur = [], []
    for s in segments + [None]:
        if s is not None and effective_pass(s):
            cur.append(s)
            continue
        if len(cur) >= min_segments:
            steps = [c.subtask for c in cur if c.subtask]
            dedup = [x for i, x in enumerate(steps) if i == 0 or x != steps[i - 1]]
            outcomes = [c.outcome for c in cur if c.outcome]
            eps.append({
                "episode_index": len(eps),
                "start_frame": cur[0].start_frame, "end_frame": cur[-1].end_frame,
                "start_s": cur[0].start_s, "end_s": cur[-1].end_s,
                "segments": [c.index for c in cur],
                "task": "; then ".join(dedup[:6]) if dedup else "manipulation task",
                "subtasks": [{"start_s": c.start_s, "end_s": c.end_s, "text": c.subtask,
                              "outcome": c.outcome, "error_type": c.error_type} for c in cur],
                "success": None if not outcomes else all(o != "error" for o in outcomes),
                "first_error_s": next((c.start_s for c in cur if c.outcome == "error"), None),
            })
        cur = []
    return eps
