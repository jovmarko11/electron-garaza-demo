"""IndustReal -> nas format: isecak videa (H.264) + CSV ishoda (start_s,end_s,step,outcome,error_type).

Stvarni format IndustReal (proveren na train_p4, 10 fps):
  data/industreal/all_rgb_videos/<rec>.mp4          RGB 1280x720, 10 fps (mpeg4)
  data/industreal/<split>/<rec>/AR_labels.csv       rec, action_id, action_name, start.jpg, end.jpg   (fine akcije)
  data/industreal/<split>/<rec>/PSR_labels_with_errors.csv   frame.jpg, step_id, step_name   (dogadjaj: korak zavrsen)
       "Incorrectly ..." = greska u izvodjenju koraka (simulira fabricki QC / MES).
  hands.csv (HL2 pracenje ruku) i pose.csv (poza glave) -> za kasnije poredjenje tacnosti.

Labele:
  step     = fina akcija iz AR_labels ("fit_wheel" -> "fit wheel")
  outcome  = "error" za akcije u prozoru izmedju prethodnog PSR dogadjaja i PSR greske; inace "ok"
  error_type = naziv PSR greske (npr. "incorrectly installed rear rear chassis pin")

Primer (snimak sa greskom u 79.8 s):
  python scripts/industreal_to_csv.py 27_main_3_1 --start 40 --end 120 --out ir_01
"""
import argparse
import glob
import subprocess
from pathlib import Path

import pandas as pd

FPS = 10.0
ROOT = Path("data/industreal")


def frame_s(name: str) -> float:
    return int(str(name).split(".")[0]) / FPS


def find_rec_dir(rec: str) -> Path:
    hits = [Path(p) for p in glob.glob(str(ROOT / "*" / rec)) if Path(p).is_dir()]
    if not hits:
        raise SystemExit(f"Nema foldera za snimak {rec} u {ROOT}/*/")
    return hits[0]


def build_labels(rec_dir: Path) -> pd.DataFrame:
    ar = pd.read_csv(rec_dir / "AR_labels.csv", header=None, names=["rec", "aid", "action", "start", "end"])
    ar["start_s"], ar["end_s"] = ar["start"].map(frame_s), ar["end"].map(frame_s)
    ar["step"] = ar["action"].str.replace("_", " ")
    psr = pd.read_csv(rec_dir / "PSR_labels_with_errors.csv", header=None, names=["frame", "sid", "name"])
    psr["t"] = psr["frame"].map(frame_s)
    times = sorted(psr["t"].unique())
    ar["outcome"], ar["error_type"] = "ok", None
    for _, e in psr[psr["name"].str.lower().str.startswith("incorrect")].iterrows():
        prev = max([t for t in times if t < e["t"]], default=0.0)
        m = (ar["end_s"] > prev) & (ar["start_s"] <= e["t"])
        ar.loc[m, "outcome"] = "error"
        ar.loc[m, "error_type"] = e["name"].lower()
    return ar[["start_s", "end_s", "step", "outcome", "error_type"]].sort_values("start_s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rec", help="npr. 27_main_3_1")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--out", default=None, help="ime izlaza u data/ (bez ekstenzije)")
    a = p.parse_args()
    out = a.out or a.rec
    rec_dir = find_rec_dir(a.rec)
    lab = build_labels(rec_dir)
    end = a.end if a.end is not None else lab["end_s"].max()
    lab = lab[(lab["end_s"] > a.start) & (lab["start_s"] < end)].copy()
    lab["start_s"] = (lab["start_s"].clip(lower=a.start) - a.start).round(2)
    lab["end_s"] = (lab["end_s"].clip(upper=end) - a.start).round(2)
    csv = Path("data") / f"{out}_steps.csv"
    lab.to_csv(csv, index=False)

    src = ROOT / "all_rgb_videos" / f"{a.rec}.mp4"
    mp4 = Path("data") / f"{out}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(a.start), "-to", str(end), "-i", str(src),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", str(mp4)],
                   check=True)
    n_err = int((lab["outcome"] == "error").sum())
    print(f"-> {mp4} ({end - a.start:.1f} s), {csv}: {len(lab)} akcija, {n_err} sa greskom")


if __name__ == "__main__":
    main()
