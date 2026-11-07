"""Provera LeRobot v3 izvoza (CP5): ucitava paket zvanicnim API-jem, kao sto bi to uradio kupac.

Upotreba: python scripts/verify_lerobot.py [ime_runa]   (podrazumevano: webcam)
"""
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

name = sys.argv[1] if len(sys.argv) > 1 else "webcam"
root = Path("runs") / name / "export" / "lerobot"
if not root.exists():
    sys.exit(f"nema izvoza u {root} (pokreni pipeline bez --no-lerobot)")

ds = LeRobotDataset(f"refinery/{name}", root=root)
print(f"epizoda: {ds.num_episodes} | frejmova: {ds.num_frames} | fps: {ds.fps}")

x = ds[0]
for k, v in x.items():
    print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else v!r}")

ok = ds.num_episodes > 0 and "task" in x and "observation.state" in x and "action" in x
print("OK" if ok else "PROBLEM: nedostaju obavezna polja")
sys.exit(0 if ok else 1)
