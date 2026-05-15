from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from .detector import make_detector
    from .preprocess import VideoReader
    from .tracker import ByteTrackLite
    from .trajectory_manager import TrajectoryManager
except ImportError:  # pragma: no cover
    from detector import make_detector
    from preprocess import VideoReader
    from tracker import ByteTrackLite
    from trajectory_manager import TrajectoryManager

from utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FPS map trajectory extraction (stage-1)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str, base_dir: Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    search_roots = [base_dir.parent, base_dir, Path.cwd()]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str((base_dir.parent / candidate).resolve())


def resolve_output_paths(output_dir: str) -> Dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return {
        "trajectory_csv": str(base / "trajectory.csv"),
        "trajectory_json": str(base / "trajectory.json"),
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(args.config)

    log_cfg = cfg.get("logging", {})
    log_file = resolve_path(log_cfg.get("file", "logs/runtime.log"), config_dir)
    logger = setup_logger(log_file, log_cfg.get("level", "INFO"))

    output_dir_cfg = cfg.get("output_dir", "outputs")
    output_dir = str((config_dir.parent / output_dir_cfg).resolve()) if not Path(output_dir_cfg).is_absolute() else output_dir_cfg
    output_paths = resolve_output_paths(output_dir)

    video_path = resolve_path(cfg["video_path"], config_dir)
    model_path = resolve_path(cfg["model_path"], config_dir)

    reader = VideoReader(video_path, roi=cfg.get("roi", {}), debug=False)
    first_frame = reader.read()
    if first_frame is None:
        logger.error("Video is empty: %s", cfg["video_path"])
        return

    detector_cfg = dict(cfg.get("detector", {}))
    detector_cfg["model_path"] = model_path
    detector = make_detector(detector_cfg)

    tracker_cfg = cfg.get("tracker", {})
    tracker = ByteTrackLite(
        max_distance=tracker_cfg.get("max_distance", 35),
        max_lost=tracker_cfg.get("max_lost", 12),
        min_hits=tracker_cfg.get("min_hits", 1),
        high_conf_threshold=tracker_cfg.get("high_conf_threshold", 0.0),
        low_conf_threshold=tracker_cfg.get("low_conf_threshold", 0.0),
        process_noise=tracker_cfg.get("process_noise", 1e-2),
        measurement_noise=tracker_cfg.get("measurement_noise", 1e-1),
        position_weight=tracker_cfg.get("position_weight", 1.0),
        motion_weight=tracker_cfg.get("motion_weight", 0.0),
        trajectory_length=tracker_cfg.get("trajectory_length", 1500),
        prediction_append_limit=tracker_cfg.get("prediction_append_limit", 6),
        use_kalman=tracker_cfg.get("use_kalman", True),
        use_motion_prediction=tracker_cfg.get("use_motion_prediction", False),
        max_speed=tracker_cfg.get("max_speed", 120),
        keep_history_len=tracker_cfg.get("keep_history_len", 1000),
        smoothing_beta=cfg.get("visualizer", {}).get("smoothing_beta", 0.85),
        birth_frames=tracker_cfg.get("birth_frames", 3),
        birth_border_margin=tracker_cfg.get("birth_border_margin", 60.0),
        enable_reid=False,
        occlusion_freeze_updates=False,
    )

    trajectory_manager = TrajectoryManager(team_min_confidence=0.55, n_team_clusters=6)

    total_detections = 0
    total_visible_tracks = 0

    frame_data = first_frame
    frame_count = 0
    while frame_data is not None:
        frame_count += 1
        frame = frame_data["frame"]
        frame_id = int(frame_data["frame_id"])
        timestamp = float(frame_data["timestamp"])

        minimap_cfg = cfg.get("minimap", {})
        if minimap_cfg and minimap_cfg.get("minimap_roi"):
            roi = minimap_cfg["minimap_roi"]
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"])
            crop = frame[y : y + h, x : x + w]
            detections = detector.detect(crop)
            for d in detections:
                bx1, by1, bx2, by2 = [int(v) for v in d["bbox"]]
                d["bbox"] = [bx1 + x, by1 + y, bx2 + x, by2 + y]
        else:
            detections = detector.detect(frame)

        print(f"Tracker input detections: {len(detections)}")
        total_detections += len(detections)
        for d in detections:
            d["frame_id"] = frame_id

        tracks = tracker.update(detections, frame=frame)
        total_visible_tracks += len(tracks)
        trajectory_manager.update_from_tracks(tracks, frame_id=frame_id, timestamp=timestamp, frame=frame)

        if frame_count % 100 == 0:
            logger.info("Processed %d frames", frame_count)

        frame_data = reader.read()

    reader.release()

    trajectory_manager.export_csv(output_paths["trajectory_csv"])
    trajectory_manager.export_json(output_paths["trajectory_json"])

    logger.info("Trajectory CSV saved to %s", output_paths["trajectory_csv"])
    logger.info("Trajectory JSON saved to %s", output_paths["trajectory_json"])
    logger.info(
        "Pipeline stats | frames=%d detections=%d visible_tracks=%d samples=%d created_tracks=%d blocked_births=%d",
        frame_count,
        total_detections,
        total_visible_tracks,
        len(trajectory_manager.samples),
        int(getattr(tracker, "stats", {}).get("created_tracks", 0)),
        int(getattr(tracker, "stats", {}).get("blocked_births", 0)),
    )

if __name__ == "__main__":
    main()