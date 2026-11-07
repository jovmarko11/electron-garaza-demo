# Data Refinery: hackathon demo (pitch 8. 11. u 18:00)

Fabrički egocentric video → anonimizovan, QC-proveren, labelovan paket za treniranje robota (LeRobot v3).
Plan rada i checkpointi: `../PLAN_IZRADE.md`. Kada korisnik napiše "CPx", pročitaj taj odeljak plana i drži se njegovog kriterijuma "Gotovo kad".

## Komande
- Sintetički test: `python run_pipeline.py --video data/synth_h264.mp4 --annotations data/synth_steps.csv --faces none --hands fake --name synth --no-lerobot`
- Pravi snimak: `python run_pipeline.py --video data/<clip>.mp4 [--annotations data/<clip>_steps.csv] --name <ime> --faces mediapipe --hands mediapipe --max-seconds 60`
- Dashboard: `streamlit run app.py`
- Python: `.venv/bin/python` (3.12). Modeli su u `models/` (MediaPipe .task / .tflite).

## Arhitektura
`ingest → privacy + perceive → qc → label → package`, orkestracija u `refinery/pipeline.py`.
- `refinery/schema.py` je **ugovor između modula** (Segment, ClipManifest, dimenzije stanja i akcija). Ne menjaj ga bez eksplicitnog odobrenja: menja izlaz svih modula.
- `qc.gate()` je čista funkcija nad `frames.parquet`; dashboard je poziva uživo.
- Izlaz: `runs/<ime>/`. Folder `factory_edge/` (sirovi video, vault pseudonima) NE SME u izvoz ni u zip.

## Pravila
- Mašina: MacBook Air M2, **8 GB RAM**. Bez velikih modela: VLM najviše `Qwen2.5-VL-3B-Instruct-4bit` (mlx-vlm). Za testove `--max-seconds 60`.
- Licence: default modeli moraju biti komercijalno dozvoljeni (MediaPipe, Apache-2.0). WiLoR / MANO samo kao opcioni backend, uz napomenu.
- Pre izmene koja dira više od jednog modula: kratak plan, pa čekaj potvrdu.
- Posle izmene pokreni sintetički test (mora i dalje da prolazi: Upotrebljivo 60 %, 2 epizode), pa ciljani test za CP.
- Ne briši `runs/` ni podatke u `data/` bez pitanja.
- Komentari i UI tekstovi na srpskom (latinica). Labele podzadataka (jezik za kupca) na engleskom.
- Poštene napomene ostaju u kodu: dubina zgloba je PROCENA, QC je trijaža + ručna odluka.

## Poznate zamke
- MediaPipe handedness pretpostavlja selfie (ogledalnu) sliku, zato `hands.py` obrće labelu. Proveriti vizuelno u `overlay.mp4`.
- I `opencv-python` i `opencv-contrib-python` su instalirani. Ako `import cv2` pravi probleme, ostaviti samo contrib (traži ga MediaPipe).
- Streamlit: koristiti `width="stretch"`, ne `use_container_width`.
- LeRobot (>=0.4, ovde 0.7) API: `LeRobotDataset.create(...)` → `add_frame({... "task": str})` → `save_episode()` → `finalize()`.
