# Data Refinery — hackathon demo

Fabrički egocentric video → **anonimizovan, proveren (QC), labelovan** paket za treniranje robota (LeRobot v3).

```
ingest → privacy + percepcija → QC kapije → labele → paket  →  Streamlit dashboard
(Dušan)   (Dušan / Marko)       (Zoran)     (Marko)   (Marko)    (Dušan)
```

## Brzi start (Mac, Apple Silicon)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-extra.txt          # LeRobot izvoz + mlx-vlm (opciono, povlači torch)
bash scripts/download_models.sh                # MediaPipe modeli za ruke i lica

# test bez modela (sintetičke ruke), da proveriš da sve radi:
python run_pipeline.py --video data/synth_h264.mp4 --annotations data/synth_steps.csv \
       --faces none --hands fake --name synth

# pravi snimak:
python run_pipeline.py --video data/clip.mp4 --annotations data/clip_steps.csv --name demo1 \
       --hfov 64 --max-seconds 90            # --captions mlx za VLM jezik

streamlit run app.py
```

Ne treba treniranje, sve je inferencija. Na M-čipu MediaPipe radi na CPU-u. Qwen2.5-VL-7B (4-bit) preko `mlx-vlm` traži oko 6 GB memorije. Ako je sporo ili nema dovoljno RAM-a: `--captions template`, ili 3B model u `refinery/label.py`.

## Moduli i vlasnici

| Fajl | Šta radi | Vlasnik |
|---|---|---|
| `refinery/schema.py` | **ugovor**: manifest, segment, rasporedi stanja i akcija | svi (menjati zajedno!) |
| `refinery/ingest.py` | sha256, ffprobe, pseudonim radnika (vault ostaje u `runs/_factory_vault`) | Dušan |
| `refinery/privacy.py` | blur lica (MediaPipe / Haar; EgoBlur kao upgrade), hold 5 frejmova, bez zvuka | Dušan |
| `refinery/perceive.py` | jedan prolaz: metrike po frejmu + ruke → `frames.parquet` | Marko / Zoran |
| `refinery/qc.py` | 6 kapija na segmentima od 2 s, razlozi, levak, granični slučajevi | **Zoran** |
| `refinery/hands.py` | 21 tačka, zglob u koordinatama kamere (procena dubine), rotacija, otvor šake | **Marko** |
| `refinery/label.py` | ishod iz CSV-a, VLM jezik na kolažu oko ruku, state/action, epizode | **Marko** |
| `refinery/package.py` | kanonski izvoz + LeRobot v3 + zip + sha256 svih fajlova | Marko |
| `refinery/render.py` | overlay (skelet + QC traka + podzadatak), H.264 | Dušan |
| `app.py` | dashboard: pipeline, kvalitet, ručna trijaža, labele, isporuka | **Dušan** |

## Izlaz (`runs/<ime>/`)
- `factory_edge/`: sirovi video, anonimizovan međukorak, pregled. **Nikad se ne pakuje.**
- `frames.parquet`, `segments.json`, `episodes.json`, `manifest.json`, `overlay.mp4`
- `export/canonical/`, `export/lerobot/` (v3.0), `export/delivery_<ime>.zip`

## Reprezentacija (nezavisna od robota)
- `observation.state` (142): po ruci valid, zglob xyz (m, kamera), rotvec, otvor šake, 21×3 tačke
- `action` (138, dexterous): Δzglob, Δrotacija, sledećih 21×3 tačaka
- `action.gripper` (14): Δzglob, Δrotacija, otvor šake
- `observation.hand_valid` (2), confidence po ruci u kanonskom izvozu

## Poštene napomene za pitch
- Dubina zgloba je **procena** iz veličine šake (pinhole). Bez kalibracije koristi se `--hfov`. Produkcija: stereo ili SLAM, kalibracija po uređaju.
- MediaPipe je Apache-2.0. WiLoR i MANO (precizniji) su samo za nekomercijalnu upotrebu.
- QC je trijaža: automatske kapije + ručna odluka za granične segmente.
