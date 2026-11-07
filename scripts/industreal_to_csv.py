"""IndustReal -> nas CSV ishoda (start_s,end_s,step,outcome,error_type) + MP4 iz frejmova.

!! Format IndustReal fajlova NISAM mogao da proverim. Pre pokretanja:
   1) skini jedan snimak (data.4tu.nl, DOI 10.4121/b008dd74-020d-4ea4-a8ba-7bb60769d224) i pogledaj README,
   2) posalji `find <snimak> -maxdepth 2 | head -50` i prve redove fajlova sa anotacijama,
   3) popuni COLUMN_MAP ispod (ili javi pa cemo zajedno).

Upotreba:
  python scripts/industreal_to_csv.py frames <folder_sa_rgb_frejmovima> data/industreal_01.mp4 --fps 10
  python scripts/industreal_to_csv.py ann <fajl_anotacija> data/industreal_01_steps.csv --fps 10
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

# Popuniti prema stvarnom fajlu (ime kolone u IndustReal -> nase polje)
COLUMN_MAP = {
    "start": "start_frame",     # pocetni frejm koraka
    "end": "end_frame",         # krajnji frejm koraka
    "step": "step_name",        # naziv koraka / akcije
    "error": "error_label",     # npr. prazno = ok, inace tip greske
}


def frames_to_mp4(folder: Path, out: Path, fps: float) -> None:
    exts = sorted({p.suffix for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")})
    if not exts:
        raise SystemExit(f"Nema slika u {folder}")
    pattern = str(folder / f"*{exts[0]}")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-pattern_type", "glob", "-i", pattern,
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)], check=True)


def ann_to_csv(src: Path, out: Path, fps: float) -> None:
    a = pd.read_csv(src)
    c = COLUMN_MAP
    missing = [v for v in c.values() if v not in a.columns]
    if missing:
        raise SystemExit(f"Kolone {missing} ne postoje. Dostupne: {list(a.columns)} -> popuni COLUMN_MAP")
    err = a[c["error"]].fillna("").astype(str).str.strip()
    pd.DataFrame({
        "start_s": a[c["start"]] / fps,
        "end_s": a[c["end"]] / fps,
        "step": a[c["step"]].astype(str),
        "outcome": ["ok" if e in ("", "0", "none", "correct") else "error" for e in err.str.lower()],
        "error_type": [None if e in ("", "0", "none", "correct") else e for e in err],
    }).to_csv(out, index=False)
    print(f"-> {out}")


p = argparse.ArgumentParser()
p.add_argument("mode", choices=["frames", "ann"])
p.add_argument("src", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--fps", type=float, default=10.0)
a = p.parse_args()
frames_to_mp4(a.src, a.out, a.fps) if a.mode == "frames" else ann_to_csv(a.src, a.out, a.fps)
