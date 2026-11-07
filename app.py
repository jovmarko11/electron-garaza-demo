"""Dashboard (Dusan): streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from refinery.pipeline import finish, run
from refinery.qc import DEFAULT_CFG, GATE_LABELS, effective_pass, gate, summarize
from refinery.schema import load_manifest, load_segments, write_json

RUNS = Path("runs")
st.set_page_config(page_title="Data Refinery — fabricki egocentric podaci", layout="wide")


# ---------------- pomocne ----------------

def list_runs() -> list[Path]:
    return sorted([p for p in RUNS.glob("*") if (p / "manifest.json").exists()],
                  key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def frames_df(run_dir: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "frames.parquet")


def thumb(video: Path, frame_idx: int, width: int = 360) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    h = int(fr.shape[0] * width / fr.shape[1])
    return cv2.cvtColor(cv2.resize(fr, (width, h)), cv2.COLOR_BGR2RGB)


def regate(run_dir: Path, cfg: dict, lerobot: bool) -> None:
    """Nove QC granice -> novi segmenti; cuva rucne odluke i jezik po indeksu segmenta."""
    m = load_manifest(run_dir)
    old = {s.index: s for s in load_segments(run_dir)}
    df = pd.read_parquet(run_dir / "frames.parquet")
    segs, _ = gate(df, m.fps, m.width, cfg)
    for s in segs:
        o = old.get(s.index)
        if o:
            s.human_decision = o.human_decision
            if o.subtask_source in ("vlm", "template"):
                s.subtask, s.subtask_source = o.subtask, o.subtask_source
    m.qc_summary = summarize(segs, {**DEFAULT_CFG, **cfg})
    write_json(run_dir / "manifest.json", m)
    finish(run_dir, df=df, segments=segs, lerobot=lerobot, keep_existing_language=True)


def refresh_summary(run_dir: Path) -> None:
    m = load_manifest(run_dir)
    segs = load_segments(run_dir)
    m.qc_summary = summarize(segs, m.qc_summary.get("config", DEFAULT_CFG))
    write_json(run_dir / "manifest.json", m)


# ---------------- sidebar ----------------

st.sidebar.title("Data Refinery")
st.sidebar.caption("Fabricki egocentric video → anonimizovani, provereni, labelovani podaci za robote")

with st.sidebar.expander("1. Prijem novog snimka", expanded=not list_runs()):
    up = st.file_uploader("Video (mp4/mov)", type=["mp4", "mov", "avi"])
    ann = st.file_uploader("Anonacije ishoda (CSV, opciono)", type=["csv"])
    c1, c2 = st.columns(2)
    faces = c1.selectbox("Lica", ["mediapipe", "haar", "none"])
    hands = c2.selectbox("Ruke", ["mediapipe", "fake", "none"])
    captions = st.selectbox("Jezik", ["template", "mlx"], help="mlx = Qwen2.5-VL lokalno na Macu")
    max_s = st.number_input("Max sekundi", 5, 600, 60)
    if up and st.button("Pokreni pipeline", type="primary"):
        Path("data").mkdir(exist_ok=True)
        vpath = Path("data") / up.name
        vpath.write_bytes(up.getbuffer())
        apath = None
        if ann:
            apath = Path("data") / ann.name
            apath.write_bytes(ann.getbuffer())
        with st.spinner("Obrada... (privacy → QC → labele → paket)"):
            run(vpath, RUNS / vpath.stem, annotations=apath, faces=faces, hands=hands,
                captions=captions, max_seconds=float(max_s))
        st.rerun()

runs = list_runs()
if not runs:
    st.info("Nema obradjenih snimaka. Ubaci video levo ili pokreni `python run_pipeline.py --video ...`.")
    st.stop()

run_dir = st.sidebar.selectbox("Snimak", runs, format_func=lambda p: p.name)
m = load_manifest(run_dir)
segs = load_segments(run_dir)
df = frames_df(str(run_dir), (run_dir / "frames.parquet").stat().st_mtime)
cfg0 = {**DEFAULT_CFG, **m.qc_summary.get("config", {})}

with st.sidebar.expander("2. QC pragovi"):
    cfg = dict(cfg0)
    cfg["max_blur_frac"] = st.slider("Max udeo mutnih frejmova", 0.0, 1.0, float(cfg0["max_blur_frac"]), 0.05)
    cfg["blur_rel_to_median"] = st.slider("Mutno ako je ostrina < x · medijana", 0.1, 0.9,
                                          float(cfg0["blur_rel_to_median"]), 0.05)
    cfg["min_hand_frac"] = st.slider("Min udeo frejmova sa rukom", 0.0, 1.0, float(cfg0["min_hand_frac"]), 0.05)
    cfg["max_trunc_frac"] = st.slider("Max udeo odsecenih ruku", 0.0, 1.0, float(cfg0["max_trunc_frac"]), 0.05)
    cfg["max_jerk"] = st.slider("Max trzaj kamere", 0.001, 0.05, float(cfg0["max_jerk"]), 0.001, format="%.3f")
    cfg["min_activity"] = st.slider("Min aktivnost ruku", 0.0, 0.2, float(cfg0["min_activity"]), 0.005)
    re_lerobot = st.checkbox("Ponovo izvezi i LeRobot", value=False)
    if st.button("Primeni pragove"):
        with st.spinner("Ponovno ocenjivanje..."):
            regate(run_dir, cfg, re_lerobot)
        st.rerun()

# ---------------- KPI ----------------

q, lab = m.qc_summary, m.labels
st.title(f"Snimak: {m.run_id}")
k = st.columns(6)
k[0].metric("Ulaz", f"{q['input_seconds']:.0f} s")
k[1].metric("Upotrebljivo", f"{q['usable_pct']:.0f} %", f"{q['usable_seconds']:.0f} s")
k[2].metric("Za rucnu proveru", q["n_borderline"])
k[3].metric("Zamucenih lica", m.privacy.get("face_boxes", 0))
k[4].metric("Frejmova sa rukom", f"{lab.get('frames_any_hand_pct', 0):.0f} %")
k[5].metric("Epizoda za kupca", lab.get("n_episodes", 0), f"{lab.get('episode_seconds', 0):.0f} s")

t1, t2, t3, t4, t5 = st.tabs(["Pipeline", "Kvalitet", "Rucna trijaza", "Labele", "Isporuka"])

# ---------------- 1. Pipeline ----------------
with t1:
    a, b = st.columns(2)
    a.subheader("Sirovo — ostaje u fabrici")
    raw = run_dir / "factory_edge" / "raw_preview.mp4"
    if raw.exists():
        a.video(str(raw))
    b.subheader("Obradjeno — ide kupcu")
    b.video(str(run_dir / "overlay.mp4"))
    st.caption("Plava/narandzasta = desna/leva ruka (21 tacka). Gornja traka = QC odluka, donja = podzadatak i ishod.")
    st.markdown(
        f"**Privacy:** detektor lica `{m.privacy.get('face_detector')}`, frejmova sa licima "
        f"{m.privacy.get('frames_with_faces', 0)}, zvuk uklonjen: {m.privacy.get('audio_removed')}, "
        f"radnik → `{m.worker_pseudonym}` (mapiranje ostaje u fabrici)."
    )

# ---------------- 2. Kvalitet ----------------
with t2:
    c1, c2 = st.columns([3, 2])
    fun = pd.DataFrame(q["funnel"])
    fig = go.Figure(go.Funnel(y=fun["kapija"], x=fun["sekundi"], textinfo="value+percent initial"))
    fig.update_layout(title="Levak kvaliteta (sekunde koje prolaze svaku kapiju)", height=420,
                      margin=dict(l=10, r=10, t=50, b=10))
    c1.plotly_chart(fig, width="stretch")
    rr = q.get("rejected_seconds_by_reason", {})
    if rr:
        r = pd.DataFrame({"razlog": list(rr), "sekundi": list(rr.values())})
        c2.plotly_chart(px.bar(r, x="sekundi", y="razlog", orientation="h", title="Odbijeno po razlogu"),
                        width="stretch")
    else:
        c2.success("Nijedan segment nije odbijen.")

    sd = pd.DataFrame([{
        "segment": s.index, "start": s.start_s, "trajanje": s.end_s - s.start_s,
        "status": ("OK" if effective_pass(s) else "odbijeno") + (" (rucno)" if s.human_decision else ""),
        "razlog": GATE_LABELS.get(s.qc_reason, "") if s.qc_reason else "",
    } for s in segs])
    fig = px.bar(sd, x="trajanje", y=["timeline"] * len(sd), base="start", orientation="h", color="status",
                 hover_data=["segment", "razlog"],
                 color_discrete_map={"OK": "#3a3", "odbijeno": "#d33", "OK (rucno)": "#7c7", "odbijeno (rucno)": "#f77"})
    fig.update_layout(title="Vremenska osa segmenata (2 s)", height=200, xaxis_title="sekunde", yaxis_title="",
                      barmode="overlay", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    met = df[["t", "sharpness", "brightness", "frame_diff"]].copy()
    met["ruke"] = df["left_valid"] + df["right_valid"]
    sel = st.multiselect("Metrike po frejmu", ["sharpness", "brightness", "frame_diff", "ruke"],
                         default=["sharpness", "ruke"])
    if sel:
        st.plotly_chart(px.line(met, x="t", y=sel), width="stretch")
    with st.expander("Metrike po segmentu"):
        st.dataframe(pd.DataFrame([{"segment": s.index, "pass": s.qc_pass, "razlog": s.qc_reason,
                                    **{kk: round(v, 4) for kk, v in s.qc_metrics.items()}} for s in segs]))

# ---------------- 3. Trijaza ----------------
with t3:
    st.caption("Model rangira, covek odlucuje granicne slucajeve (kao Scale AI QC). Odluke ulaze u manifest.")
    cand = [s for s in segs if s.borderline or not s.qc_pass]
    if not cand:
        st.success("Nema segmenata za rucnu proveru.")
    for s in cand[:12]:
        c1, c2 = st.columns([1, 2])
        img = thumb(run_dir / "overlay.mp4", (s.start_frame + s.end_frame) // 2)
        if img is not None:
            c1.image(img)
        c2.markdown(f"**Segment {s.index}** ({s.start_s:.1f}–{s.end_s:.1f} s) — auto: "
                    f"{'OK' if s.qc_pass else 'odbijeno: ' + GATE_LABELS.get(s.qc_reason, '')}"
                    f"{' · granicni' if s.borderline else ''}")
        c2.json({kk: round(v, 3) for kk, v in s.qc_metrics.items()}, expanded=False)
        opts = ["auto", "approve", "reject"]
        cur = s.human_decision or "auto"
        choice = c2.radio("Odluka", opts, index=opts.index(cur), horizontal=True, key=f"dec_{s.index}")
        new = None if choice == "auto" else choice
        if new != s.human_decision:
            s.human_decision = new
            write_json(run_dir / "segments.json", segs)
            refresh_summary(run_dir)
            st.rerun()
    if st.button("Ponovo spakuj sa rucnim odlukama"):
        with st.spinner("Pakovanje..."):
            finish(run_dir, lerobot=True, keep_existing_language=True)
        st.rerun()

# ---------------- 4. Labele ----------------
with t4:
    ld = pd.DataFrame([{"start": s.start_s, "trajanje": s.end_s - s.start_s, "podzadatak": s.subtask or "",
                        "ishod": s.outcome or "nepoznato", "izvor": s.subtask_source or "",
                        "greska": s.error_type or ""} for s in segs])
    fig = px.bar(ld, x="trajanje", y="podzadatak", base="start", orientation="h", color="ishod",
                 hover_data=["izvor", "greska"],
                 color_discrete_map={"ok": "#3a3", "error": "#d33", "recovery": "#e90", "nepoznato": "#999"})
    fig.update_layout(title="Podzadaci i ishod (labele koje drugi nemaju)", xaxis_title="sekunde", height=360)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    side = c1.radio("Ruka", ["right", "left"], horizontal=True)
    wc = np.array([x if v > 0 else [np.nan] * 3 for x, v in zip(df[f"{side}_wrist_cam"], df[f"{side}_valid"])],
                  float)
    tr = pd.DataFrame({"t": df["t"], "x": wc[:, 0], "y": wc[:, 1], "z (procena)": wc[:, 2]})
    c1.plotly_chart(px.line(tr, x="t", y=["x", "y", "z (procena)"], title="Zglob u koordinatama kamere (m)"),
                    width="stretch")
    ap = pd.DataFrame({"t": df["t"], "otvor sake (m)": df[f"{side}_aperture"], "score": df[f"{side}_score"]})
    c2.plotly_chart(px.line(ap, x="t", y=["otvor sake (m)", "score"], title="Hvat (gripper adapter) + pouzdanost"),
                    width="stretch")
    eps_path = run_dir / "episodes.json"
    if eps_path.exists():
        eps = json.loads(eps_path.read_text())
        st.subheader("Epizode za kupca")
        st.dataframe(pd.DataFrame([{k2: e[k2] for k2 in ("episode_index", "start_s", "end_s", "task", "success",
                                                         "first_error_s")} for e in eps]))

# ---------------- 5. Isporuka ----------------
with t5:
    c1, c2 = st.columns(2)
    lr = m.exports.get("lerobot", {})
    c1.markdown(f"**LeRobot v3:** `{lr.get('status')}` {lr.get('reason', '')}")
    info = Path(lr.get("root", "")) / "meta" / "info.json" if lr.get("root") else None
    if info and info.exists():
        c1.json(json.loads(info.read_text()), expanded=False)
    c1.markdown("**Rasporedi akcija:** " + ", ".join(lab.get("action_layouts", [])))
    zp = run_dir / m.exports.get("delivery_zip", "missing")
    if zp.exists():
        c1.download_button("Preuzmi paket za kupca (.zip)", zp.read_bytes(), file_name=zp.name, type="primary")
        c1.code(f"sha256 {m.exports.get('delivery_sha256')}")
    c2.markdown("**Manifest (provenance, QC, privacy, prava)**")
    c2.json(json.loads((run_dir / "manifest.json").read_text()), expanded=False)
