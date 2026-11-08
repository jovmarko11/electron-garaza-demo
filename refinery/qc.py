"""2b. QC kapije (Zoran). Cista funkcija nad frames.parquet -> moze da se pokrece uzivo iz dashboarda.

Redosled kapija je bitan: segment dobija PRVI razlog odbijanja, a u qc_flags se cuvaju sve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SEGMENT_SECONDS, Segment

GATES = [
    ("integritet", "Integrity (dropped / out-of-order frames)"),
    ("ekspozicija", "Exposure (black / overexposed / green artifact)"),
    ("zamucenje", "Blur"),
    ("ruke", "Hands not visible / truncated"),
    ("kamera", "Camera shake"),
    ("aktivnost", "Idle (no activity)"),
]
GATE_LABELS = dict(GATES)

DEFAULT_CFG = {
    "max_drop_frac": 0.10,
    "max_bad_exposure_frac": 0.30,
    "blur_rel_to_median": 0.40,      # frejm je mutan ako je sharpness < 0.4 * medijana klipa
    "max_blur_frac": 0.40,
    "min_hand_frac": 0.40,
    "max_trunc_frac": 0.50,
    "max_jerk": 0.012,               # udeo sirine slike po frejmu (druga razlika globalnog toka)
    "min_activity": 0.02,            # brzina zgloba, udeo sirine slike u sekundi
    "borderline_band": 0.15,
}


def _frame_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["drop"] = (df["pts_gap"] + df["pts_backwards"]) > 0
    f["bad_exposure"] = ((df["overexposed"] > 0.25) | (df["underexposed"] > 0.25)
                         | (df["brightness"] < 15) | (df["green_artifact"] > 40))
    med = float(np.nanmedian(df["sharpness"])) if len(df) else 0.0
    f["blurry"] = df["sharpness"] < cfg["blur_rel_to_median"] * med
    f["hand"] = (df["left_valid"] > 0) | (df["right_valid"] > 0)
    f["trunc"] = ((df["left_valid"] > 0) & (df["left_truncated"] > 0)) | \
                 ((df["right_valid"] > 0) & (df["right_truncated"] > 0))
    ddx, ddy = df["flow_dx"].diff().fillna(0), df["flow_dy"].diff().fillna(0)
    f["jerk"] = np.sqrt(ddx ** 2 + ddy ** 2) / 256.0
    return f


def _wrist_speed(df: pd.DataFrame, W: int, fps: float) -> pd.Series:
    sp = []
    for side in ("left", "right"):
        xy = np.array([k[:2] if isinstance(k, (list, np.ndarray)) else [np.nan, np.nan]
                       for k in df[f"{side}_kp2d"]], dtype=float)
        d = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=1) / W * fps
        sp.append(pd.Series(d, index=df.index))
    return pd.concat(sp, axis=1).max(axis=1, skipna=True).fillna(0.0)


def gate(df: pd.DataFrame, fps: float, W: int, cfg: dict | None = None) -> tuple[list[Segment], dict]:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    ff = _frame_flags(df, cfg)
    speed = _wrist_speed(df, W, fps)
    seg_len = max(1, int(round(SEGMENT_SECONDS * fps)))
    segments: list[Segment] = []
    for si, s in enumerate(range(0, len(df), seg_len)):
        e = min(len(df), s + seg_len)
        if e - s < seg_len // 2:          # odbaci kratki ostatak na kraju
            break
        sl = slice(s, e)
        fsl = ff.iloc[sl]
        hand_frames = fsl["hand"].sum()
        m = {
            "drop_frac": float(fsl["drop"].mean()),
            "bad_exposure_frac": float(fsl["bad_exposure"].mean()),
            "blur_frac": float(fsl["blurry"].mean()),
            "hand_frac": float(fsl["hand"].mean()),
            "trunc_frac": float(fsl["trunc"].sum() / hand_frames) if hand_frames else 1.0,
            "jerk": float(np.median(fsl["jerk"])),
            "activity": float(speed.iloc[sl].mean()),
            "faces": int(df["faces"].iloc[sl].sum()),
        }
        flags = {
            "integritet": m["drop_frac"] > cfg["max_drop_frac"],
            "ekspozicija": m["bad_exposure_frac"] > cfg["max_bad_exposure_frac"],
            "zamucenje": m["blur_frac"] > cfg["max_blur_frac"],
            "ruke": (m["hand_frac"] < cfg["min_hand_frac"]) or (m["trunc_frac"] > cfg["max_trunc_frac"]),
            "kamera": m["jerk"] > cfg["max_jerk"],
            "aktivnost": m["activity"] < cfg["min_activity"],
        }
        reason = next((g for g, _ in GATES if flags[g]), None)
        band = cfg["borderline_band"]
        near = [
            abs(m["blur_frac"] - cfg["max_blur_frac"]) < band * cfg["max_blur_frac"],
            abs(m["hand_frac"] - cfg["min_hand_frac"]) < band * cfg["min_hand_frac"],
            abs(m["jerk"] - cfg["max_jerk"]) < band * cfg["max_jerk"],
            abs(m["activity"] - cfg["min_activity"]) < band * cfg["min_activity"],
        ]
        segments.append(Segment(
            index=si, start_s=s / fps, end_s=e / fps, start_frame=s, end_frame=e,
            qc_pass=reason is None, qc_reason=reason, qc_flags=flags, qc_metrics=m,
            borderline=any(near),
        ))
    return segments, summarize(segments, cfg)


def summarize(segments: list[Segment], cfg: dict) -> dict:
    total = sum(s.end_s - s.start_s for s in segments)
    funnel = [{"kapija": "input", "sekundi": total}]
    alive = list(segments)
    for g, label in GATES:
        alive = [s for s in alive if not s.qc_flags.get(g)]
        funnel.append({"kapija": label, "sekundi": sum(s.end_s - s.start_s for s in alive)})
    reasons: dict[str, float] = {}
    for s in segments:
        if s.qc_reason:
            reasons[GATE_LABELS[s.qc_reason]] = reasons.get(GATE_LABELS[s.qc_reason], 0) + (s.end_s - s.start_s)
    passed = sum(s.end_s - s.start_s for s in segments if effective_pass(s))
    return {
        "input_seconds": total,
        "usable_seconds": passed,
        "usable_pct": 100.0 * passed / total if total else 0.0,
        "n_segments": len(segments),
        "n_borderline": sum(s.borderline for s in segments),
        "funnel": funnel,
        "rejected_seconds_by_reason": reasons,
        "config": cfg,
        "method": "Automatic gates = triage; borderline segments go to manual review.",
    }


def effective_pass(s: Segment) -> bool:
    if s.human_decision == "approve":
        return True
    if s.human_decision == "reject":
        return False
    return s.qc_pass
