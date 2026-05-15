from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.signal import savgol_filter

from src.advanced_heatmap import AdvancedHeatmap
from src.event_detector import EventDetector
from src.professional_renderer import ProfessionalRenderer
from src.speed_heatmap import SpeedHeatmapGenerator

TEAM_COLORS = {
    "all": (0, 255, 255),
}


def _team_color(team: str) -> tuple[int, int, int]:
    if team in TEAM_COLORS:
        return TEAM_COLORS[team]
    seed = abs(hash(team)) % 255
    return (int((seed * 37) % 255), int((seed * 67) % 255), int((seed * 97) % 255))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render mapped trajectories on global map")
    parser.add_argument("--mapped_csv", default="outputs/mapped_trajectory.csv")
    parser.add_argument("--map_image", default="configs/map.jpg")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--team", default="all")
    parser.add_argument("--events_csv", default="outputs/events.csv")
    parser.add_argument("--events_json", default="outputs/events.json")
    parser.add_argument("--trajectory_json", default="outputs/trajectory.json")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=-1)
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def load_final_team_from_json(trajectory_json: str) -> Dict[int, str]:
    p = Path(trajectory_json)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    tracks = payload.get("tracks", {})
    out: Dict[int, str] = {}
    for k, v in tracks.items():
        try:
            tid = int(v.get("track_id", int(k)))
        except Exception:
            continue
        out[tid] = str(v.get("final_team", "unknown")).lower()
    return out


def estimate_track_team(df: pd.DataFrame, final_team_map: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    team_by_track: Dict[int, str] = {}
    if final_team_map:
        for tid in df["track_id"].unique():
            team_by_track[int(tid)] = final_team_map.get(int(tid), "unknown")
        return team_by_track

    if "final_team" in df.columns:
        for tid, group in df.groupby("track_id"):
            team_by_track[int(tid)] = str(group["final_team"].iloc[0]).lower()
        return team_by_track

    if "team_color" not in df.columns:
        for tid in df["track_id"].unique():
            team_by_track[int(tid)] = "unknown"
        return team_by_track

    for tid, group in df.groupby("track_id"):
        votes = group["team_color"].astype(str).str.lower().value_counts(normalize=True)
        if len(votes) == 0:
            team_by_track[int(tid)] = "unknown"
        else:
            team_by_track[int(tid)] = str(votes.index[0])
    return team_by_track


def _smooth_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 7:
        return pts
    win = min(len(pts) if len(pts) % 2 == 1 else len(pts) - 1, 11)
    if win < 5:
        return pts
    x = savgol_filter(pts[:, 0], window_length=win, polyorder=2, mode="interp")
    y = savgol_filter(pts[:, 1], window_length=win, polyorder=2, mode="interp")
    return np.stack([x, y], axis=1)


def render_routes(map_img: np.ndarray, df: pd.DataFrame, team_filter: str, team_by_track: Dict[int, str]) -> np.ndarray:
    grouped_tracks: Dict[int, np.ndarray] = {}
    for tid, group in df.groupby("track_id"):
        grouped_tracks[int(tid)] = group[["map_x", "map_y"]].to_numpy(dtype=np.float32)
    return ProfessionalRenderer(max_jump=40.0).render_routes(map_img, grouped_tracks, team_by_track, team_filter=team_filter)


def render_heatmap(map_img: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    h, w = map_img.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(df) == 0:
        return map_img.copy()

    for _, row in df.iterrows():
        try:
            x = int(round(float(row["map_x"])))
            y = int(round(float(row["map_y"])))
        except Exception:
            continue
        if 0 <= x < w and 0 <= y < h:
            density[y, x] += 1.0

    if float(density.max()) <= 0.0:
        return map_img.copy()

    # Stronger hotspot visibility: blur + percentile normalization
    heat = cv2.GaussianBlur(density, (45, 45), 0)
    nz = heat[heat > 0]
    if nz.size == 0:
        return map_img.copy()

    # robust normalization to avoid a single extreme point flattening the map
    p99 = float(np.percentile(nz, 99.0))
    scale = max(p99, 1e-6)
    heat_norm = np.clip(heat / scale, 0.0, 1.0)
    heat_u8 = np.clip(heat_norm * 255.0, 0, 255).astype(np.uint8)
    color_heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

    # Blend only hotspot area so background map remains clear
    out = map_img.copy()
    mask = (heat_u8 > 8).astype(np.uint8)
    if int(mask.sum()) == 0:
        return out

    alpha = 0.62
    idx = mask.astype(bool)
    out[idx] = cv2.addWeighted(map_img[idx], 1.0 - alpha, color_heat[idx], alpha, 0)
    return out


def render_event_markers(map_img: np.ndarray, events_csv: str, mapped_df: pd.DataFrame) -> np.ndarray:
    canvas = map_img.copy()
    path = Path(events_csv)
    if not path.exists():
        return canvas
    events = safe_read_csv(path)
    if len(events) == 0:
        return canvas

    style_by_type = {
        "direction_change": ((0, 255, 255), cv2.MARKER_CROSS),
        "stop_loot": ((255, 0, 0), cv2.MARKER_SQUARE),
        "high_speed_rush": ((0, 0, 255), cv2.MARKER_STAR),
        "possible_fire": ((0, 165, 255), cv2.MARKER_DIAMOND),
        "possible_death": ((128, 128, 128), cv2.MARKER_TILTED_CROSS),
    }

    for _, e in events.iterrows():
        tid = int(e["track_id"])
        fid = int(e["frame_id"])
        et = str(e.get("event_type", "")).strip()
        rows = mapped_df[(mapped_df["track_id"] == tid) & (mapped_df["frame_id"] == fid)]
        if len(rows) == 0:
            continue
        x = int(round(float(rows.iloc[0]["map_x"])))
        y = int(round(float(rows.iloc[0]["map_y"])))
        col, marker = style_by_type.get(et, ((0, 165, 255), cv2.MARKER_CROSS))
        cv2.drawMarker(canvas, (x, y), col, markerType=marker, markerSize=12, thickness=2)

    # legend panel (bottom-left)
    legend_items = [
        ("direction_change", (0, 255, 255), cv2.MARKER_CROSS),
        ("stop_loot", (255, 0, 0), cv2.MARKER_SQUARE),
        ("high_speed_rush", (0, 0, 255), cv2.MARKER_STAR),
        ("possible_fire", (0, 165, 255), cv2.MARKER_DIAMOND),
        ("possible_death", (128, 128, 128), cv2.MARKER_TILTED_CROSS),
    ]
    x0 = 16
    y0 = canvas.shape[0] - 176
    cv2.rectangle(canvas, (x0 - 12, y0 - 26), (x0 + 390, y0 + 160), (0, 0, 0), -1)
    cv2.putText(canvas, "Event Legend", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (name, col, mk) in enumerate(legend_items):
        yy = y0 + 30 + i * 24
        cv2.drawMarker(canvas, (x0 + 12, yy - 6), col, markerType=mk, markerSize=15, thickness=3)
        cv2.putText(canvas, name, (x0 + 34, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)

    return canvas


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_img = cv2.imread(args.map_image)
    if map_img is None:
        raise FileNotFoundError(f"Cannot open map image: {args.map_image}")

    df = safe_read_csv(Path(args.mapped_csv))
    if len(df) == 0:
        print(f"[render_map] mapped csv empty: {args.mapped_csv}")
        return
    if args.frame_end >= 0:
        df = df[(df["frame_id"] >= int(args.frame_start)) & (df["frame_id"] <= int(args.frame_end))]
    else:
        df = df[df["frame_id"] >= int(args.frame_start)]
    final_team_map = load_final_team_from_json(args.trajectory_json)
    team_by_track = estimate_track_team(df, final_team_map=final_team_map)

    # events (offline)
    tracks_payload: Dict[str, list] = {}
    for tid, group in df.groupby("track_id"):
        tracks_payload[str(int(tid))] = [
            {
                "frame_id": int(r["frame_id"]),
                "timestamp": float(r["timestamp"]),
                "x": float(r["map_x"]),
                "y": float(r["map_y"]),
            }
            for _, r in group.iterrows()
        ]
    EventDetector().export_events_from_tracks(tracks_payload, args.events_csv, fps=30.0, output_json=args.events_json)

    # apply requested team filter to outputs using final_team
    if args.team != "all":
        keep_ids = [tid for tid, tm in team_by_track.items() if tm == args.team]
        df_render = df[df["track_id"].isin(keep_ids)]
    else:
        df_render = df

    final_traj = render_routes(map_img, df_render, "all" if args.team == "all" else args.team, team_by_track)
    cv2.imwrite(str(out_dir / "final_trajectory.png"), final_traj)

    for team in sorted(set(team_by_track.values())) + ["all"]:
        if team not in TEAM_COLORS:
            TEAM_COLORS[team] = _team_color(team)
        img = render_routes(map_img, df, team, team_by_track)
        cv2.imwrite(str(out_dir / f"{team}_team_routes.png"), img)

    selected_tag = args.team if args.team != "all" else "all"

    # overall hotspot heatmap should represent all players in selected timeline, not team-filtered subset
    heatmap = render_heatmap(map_img, df if args.team == "all" else df_render)
    cv2.imwrite(str(out_dir / "heatmap.png"), heatmap)
    cv2.imwrite(str(out_dir / f"{selected_tag}_heatmap.png"), heatmap)

    events_img = render_event_markers(map_img, args.events_csv, df_render if args.team != "all" else df)
    cv2.imwrite(str(out_dir / "event_markers.png"), events_img)
    cv2.imwrite(str(out_dir / f"{selected_tag}_event_markers.png"), events_img)

    # timeline-filtered snapshot for consistent slider behavior in auxiliary heatmaps
    filtered_csv = out_dir / "_mapped_filtered_for_render.csv"
    (df if args.team == "all" else df_render).to_csv(filtered_csv, index=False)

    # density/speed/stay overlays on big map
    AdvancedHeatmap().generate(str(filtered_csv), str(out_dir), map_image=args.map_image)

    # specialized speed layer (blue->yellow->red) with map overlay
    SpeedHeatmapGenerator().generate(
        mapped_csv=str(filtered_csv),
        density_out=str(out_dir / "density_heatmap.png"),
        speed_out=str(out_dir / "speed_heatmap.png"),
        map_image=args.map_image,
    )

    # duplicate team-tagged artifacts for UI/team filtering consistency
    Path(out_dir / f"{selected_tag}_density_heatmap.png").write_bytes(Path(out_dir / "density_heatmap.png").read_bytes())
    Path(out_dir / f"{selected_tag}_speed_heatmap.png").write_bytes(Path(out_dir / "speed_heatmap.png").read_bytes())
    Path(out_dir / f"{selected_tag}_stay_heatmap.png").write_bytes(Path(out_dir / "stay_heatmap.png").read_bytes())


if __name__ == "__main__":
    main()
