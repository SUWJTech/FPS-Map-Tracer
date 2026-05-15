from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from pandas.errors import EmptyDataError

st.set_page_config(page_title="FPS Trajectory Analyzer", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.0rem; padding-bottom: 1.0rem;}
div[data-testid="stImage"] img {
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
</style>
""",
    unsafe_allow_html=True,
)
st.title("FPS Trajectory Analyzer")

project_root = Path(__file__).resolve().parent
output_root = project_root / "outputs"
upload_root = project_root / "uploads"
output_root.mkdir(parents=True, exist_ok=True)
upload_root.mkdir(parents=True, exist_ok=True)

config_path = project_root / "configs" / "default.yaml"
map_path = project_root / "configs" / "map.jpg"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def build_runtime_config(video_path: Path, run_name: str) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["video_path"] = str(video_path)
    cfg["output_dir"] = f"outputs/{run_name}"
    runtime_cfg = project_root / "configs" / "_runtime_webui.yaml"
    runtime_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_cfg


def run_cmd_stream(cmd: list[str], title: str) -> int:
    st.write(f"### {title}")
    log_box = st.empty()
    full_log = ""
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        full_log += line
        log_box.text_area(f"{title} log", full_log, height=220)
    ret = proc.wait()
    log_box.text_area(f"{title} log", full_log + f"\n[exit_code={ret}]\n", height=220)
    return int(ret)


st.sidebar.header("Input / Output")
compat_mode = st.sidebar.checkbox("Compatibility mode (strictly use configs/default.yaml + outputs)", value=False)
uploaded = st.sidebar.file_uploader("Upload video (one per run)", type=["mp4", "mov", "mkv", "avi"])
if uploaded is not None:
    uploaded_path = upload_root / uploaded.name
    uploaded_path.write_bytes(uploaded.getbuffer())
    st.session_state["video_path"] = str(uploaded_path)
    st.sidebar.success(f"Uploaded: {uploaded.name}")

default_video = st.session_state.get("video_path", str(project_root / "video1.mov"))
video_path_text = st.sidebar.text_input("Video path", value=default_video)

run_candidates = sorted([p.name for p in output_root.iterdir() if p.is_dir()])
default_run = Path(video_path_text).stem if video_path_text else "run"
run_name = st.sidebar.text_input("Output run name", value=default_run)
if not run_name.strip():
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

output_dir = output_root / run_name
output_dir.mkdir(parents=True, exist_ok=True)
runtime_cfg = build_runtime_config(Path(video_path_text), run_name)

if compat_mode:
    # Fully align with your verified CLI commands (uploaded video path will be ignored)
    runtime_cfg = config_path
    output_dir = output_root
    if uploaded is not None:
        st.sidebar.warning("Compatibility mode is ON: uploaded video is ignored; using configs/default.yaml video_path")

st.sidebar.caption(f"Current run output: {output_dir}")
st.sidebar.caption(f"Runtime config: {runtime_cfg}")
try:
    _cfg_preview = yaml.safe_load(Path(runtime_cfg).read_text(encoding="utf-8"))
    st.sidebar.caption(f"Effective video_path: {_cfg_preview.get('video_path', '')}")
except Exception:
    pass

st.sidebar.header("Pipeline")
mapped_csv_path = output_dir / "mapped_trajectory.csv"
time_slider_max = 100
_df = safe_read_csv(mapped_csv_path)
if len(_df) > 0 and "frame_id" in _df.columns:
    time_slider_max = int(_df["frame_id"].max())

frame_start = st.sidebar.slider("Timeline frame start", min_value=0, max_value=max(0, time_slider_max), value=0)
frame_limit = st.sidebar.slider(
    "Timeline frame end",
    min_value=max(1, frame_start + 1),
    max_value=max(1, time_slider_max),
    value=max(1, time_slider_max),
)

team_options = ["all"]
traj_json = output_dir / "trajectory.json"
if traj_json.exists():
    try:
        payload = json.loads(traj_json.read_text(encoding="utf-8"))
        tracks = payload.get("tracks", {})
        dyn = sorted({str(v.get("final_team", "unknown")) for v in tracks.values()})
        team_options.extend([t for t in dyn if t and t != "all"])
    except Exception:
        pass

team_filter = st.sidebar.selectbox("3) Team filter", team_options)

if st.sidebar.button("🚀 One-click Full Pipeline (1→2→4)"):
    extract_cmd = ["python", "-u", "-m", "src.pipeline", "--config", str(runtime_cfg)]
    map_cmd = [
        "python", "map_mapper.py",
        "--config", str(runtime_cfg),
        "--trajectory_csv", str(output_dir / "trajectory.csv"),
        "--map_image", str(map_path),
        "--matrix", str(project_root / "configs" / "homography_matrix.json"),
        "--trajectory_json", str(output_dir / "trajectory.json"),
        "--output_csv", str(output_dir / "mapped_trajectory.csv"),
        "--debug_match_png", str(output_dir / "registration_matches.png"),
        "--force_reregister",
    ]
    render_cmd = [
        "python", "render_map.py",
        "--mapped_csv", str(output_dir / "mapped_trajectory.csv"),
        "--map_image", str(map_path),
        "--output_dir", str(output_dir),
        "--team", team_filter,
        "--events_csv", str(output_dir / "events.csv"),
        "--events_json", str(output_dir / "events.json"),
        "--trajectory_json", str(output_dir / "trajectory.json"),
        # one-click default to full timeline to match your manual CLI quality
        "--frame_start", "0",
        "--frame_end", "-1",
    ]

    with st.spinner("Running full pipeline..."):
        c1 = run_cmd_stream(extract_cmd, "Step 1/3: Trajectory Extraction")
        if c1 == 0:
            c2 = run_cmd_stream(map_cmd, "Step 2/3: Map Registration + Mapping")
        else:
            c2 = 1
        if c1 == 0 and c2 == 0:
            c3 = run_cmd_stream(render_cmd, "Step 3/3: Render Map Visuals")
        else:
            c3 = 1

    if c1 == 0 and c2 == 0 and c3 == 0:
        st.success("✅ Full pipeline completed successfully. Scroll down to view outputs.")
    else:
        st.error(f"❌ Full pipeline failed. exit_codes=({c1}, {c2}, {c3})")

if st.sidebar.button("1) Run Trajectory Extraction"):
    cmd = ["python", "-u", "-m", "src.pipeline", "--config", str(runtime_cfg)]
    with st.spinner("Running extraction..."):
        p = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    st.text_area("Extraction log", (p.stdout or "") + "\n" + (p.stderr or ""), height=180)

if st.sidebar.button("2) Run Map Registration + Mapping"):
    cmd = [
        "python", "map_mapper.py",
        "--config", str(runtime_cfg),
        "--trajectory_csv", str(output_dir / "trajectory.csv"),
        "--map_image", str(map_path),
        "--matrix", str(project_root / "configs" / "homography_matrix.json"),
        "--trajectory_json", str(output_dir / "trajectory.json"),
        "--output_csv", str(output_dir / "mapped_trajectory.csv"),
        "--debug_match_png", str(output_dir / "registration_matches.png"),
        "--force_reregister",
    ]
    with st.spinner("Running registration..."):
        p = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    st.text_area("Registration log", (p.stdout or "") + "\n" + (p.stderr or ""), height=180)

if st.sidebar.button("4) Render Map Visuals"):
    cmd = [
        "python", "render_map.py",
        "--mapped_csv", str(output_dir / "mapped_trajectory.csv"),
        "--map_image", str(map_path),
        "--output_dir", str(output_dir),
        "--team", team_filter,
        "--events_csv", str(output_dir / "events.csv"),
        "--events_json", str(output_dir / "events.json"),
        "--trajectory_json", str(output_dir / "trajectory.json"),
        "--frame_start", str(frame_start),
        "--frame_end", str(frame_limit),
    ]
    with st.spinner("Rendering map visuals..."):
        p = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    st.text_area("Render log", (p.stdout or "") + "\n" + (p.stderr or ""), height=180)

st.sidebar.header("Display")
show_heatmap = st.sidebar.checkbox("Show heatmaps", value=True)
show_events = st.sidebar.checkbox("Show events table", value=True)
show_event_markers = st.sidebar.checkbox("Show event markers image", value=True)
st.sidebar.caption(f"Team: {team_filter} | Frames: {frame_start}-{frame_limit}")

col1, col2 = st.columns(2)
traj_png = output_dir / "final_trajectory.png"
team_png = output_dir / f"{team_filter}_team_routes.png"
match_png = output_dir / "registration_matches.png"

if traj_png.exists():
    col1.subheader("final_trajectory.png")
    col1.image(str(traj_png), width="stretch")
if team_png.exists():
    col2.subheader(team_png.name)
    col2.image(str(team_png), width="stretch")

if show_heatmap:
    st.subheader("Heatmaps")
    c3, c4 = st.columns(2)
    prefix = team_filter if team_filter != "all" else "all"
    for p, col in [
        (output_dir / f"{prefix}_heatmap.png", c3),
        (output_dir / f"{prefix}_density_heatmap.png", c4),
        (output_dir / f"{prefix}_speed_heatmap.png", c3),
        (output_dir / f"{prefix}_stay_heatmap.png", c4),
    ]:
        fallback = output_dir / p.name.replace(f"{prefix}_", "")
        target = p if p.exists() else fallback
        if target.exists():
            col.markdown(f"**{target.name}**")
            col.image(str(target), width="stretch")

if match_png.exists():
    st.subheader("registration_matches.png")
    st.image(str(match_png), width="stretch")

if show_event_markers:
    ev_img = output_dir / (f"{team_filter}_event_markers.png" if team_filter != "all" else "all_event_markers.png")
    if not ev_img.exists():
        ev_img = output_dir / "event_markers.png"
    if ev_img.exists():
        st.subheader(ev_img.name)
        st.image(str(ev_img), width="stretch")

st.subheader("CSVs")
for name in ["trajectory.csv", "mapped_trajectory.csv", "events.csv"]:
    path = output_dir / name
    if not path.exists():
        continue
    st.write(name)
    dfv = safe_read_csv(path)
    if len(dfv) == 0:
        st.info(f"{name} is empty")
        continue
    if "frame_id" in dfv.columns:
        dfv = dfv[(dfv["frame_id"] >= frame_start) & (dfv["frame_id"] <= frame_limit)]
    st.dataframe(dfv.head(1000))

