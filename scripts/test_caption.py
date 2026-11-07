"""Izolovan test VLM opisivača (CP4): opisuje do 3 kolaža i meri vreme po kolažu.

Upotreba: python scripts/test_caption.py [runs/webcam/collages] [--n 3]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refinery.label import MLXCaptioner, VERBS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="runs/webcam/collages")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    files = sorted(Path(a.folder).glob("seg_*.jpg"))[:a.n]
    if not files:
        sys.exit(f"nema kolaza u {a.folder}")

    t0 = time.time()
    cap = MLXCaptioner()
    print(f"model: {cap.name} (ucitavanje {time.time() - t0:.1f} s)")

    times = []
    for f in files:
        t = time.time()
        text = cap.caption(f, None, None)  # seg i df se u MLXCaptioner ne koriste
        dt = time.time() - t
        times.append(dt)
        ok = text.split(" ")[0] in VERBS
        print(f"{f.name}: '{text}'  [{dt:.1f} s]  glagol sa liste: {'da' if ok else 'NE'}")
    print(f"prosek po kolazu: {sum(times) / len(times):.1f} s (prvi poziv ukljucuje zagrevanje)")


if __name__ == "__main__":
    main()
