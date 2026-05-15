from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from src.event_detector import EventDetector
from src.heatmap_generator import HeatmapGenerator
from src.homography_mapper import HomographyMapper


COLOR_MAP = {
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "purple": (255, 0, 255),
    "all": (0, 255, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline trajectory renderer (stage-2)")
    parser.add_argument("--input_json", default="outputs/trajectory.json")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--team", default="all", choices=["all", "white", "red", "green", "purple"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def load_tracks(path: str) -> Dict[str, List[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("tracks", {})


def render_trajectories(tracks: Dict[str, List[dict]], width: int, height: int, team_filter: str) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for _, samples in tracks.items():
        if len(samples) < 2:
            continue
        team = str(samples[0].get("team_color", "unknown")).lower()
        if team_filter != "all" and team != team_filter:
            continue
        color = COLOR_MAP.get(team, COLOR_MAP["all"])
        pts = np.array([[int(s["x"]), int(s["y"])] for s in samples], dtype=np.int32)
        overlay = canvas.copy()
        for i in range(1, len(pts)):
            cv2.line(overlay, tuple(pts[i - 1]), tuple(pts[i]), color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    return canvas


def save_png(image: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, image)


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args.input_json)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_img = render_trajectories(tracks, args.width, args.height, args.team)
    save_png(traj_img, str(out_dir / "final_trajectory.png"))

    filtered_img = render_trajectories(tracks, args.width, args.height, args.team)
    save_png(filtered_img, str(out_dir / "filtered_tracks.png"))

    HeatmapGenerator().generate_from_tracks(tracks, (args.height, args.width, 3), str(out_dir / "heatmap.png"))

    # also generate optional event/mapped outputs in offline stage
    EventDetector().export_events_from_tracks(tracks, str(out_dir / "events.csv"), fps=30.0)
    traj_csv = out_dir / "trajectory.csv"
    if traj_csv.exists():
        HomographyMapper().export_identity_mapped_csv(str(traj_csv), str(out_dir / "mapped_trajectory.csv"))


if __name__ == "__main__":
    main()
