from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class TrajectorySample:
    frame_id: int
    timestamp: float
    track_id: int
    x: float
    y: float
    predicted: bool
    confidence: float
    vx: float
    vy: float
    team_color: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "timestamp": float(self.timestamp),
            "track_id": int(self.track_id),
            "x": float(self.x),
            "y": float(self.y),
            "predicted": bool(self.predicted),
            "confidence": float(self.confidence),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "team_color": self.team_color,
        }


class TrajectoryManager:
    """Collect track-level time series and export structured results."""

    def __init__(self, team_min_confidence: float = 0.55, n_team_clusters: int = 6) -> None:
        self.samples: List[TrajectorySample] = []
        self._track_history: Dict[int, List[TrajectorySample]] = {}
        self._color_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._track_color_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.team_min_confidence = float(team_min_confidence)
        self.n_team_clusters = int(max(2, n_team_clusters))

    @staticmethod
    def _extract_color_feature(frame: np.ndarray, bbox: Sequence[int]) -> Optional[Dict[str, Any]]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        h, w = roi.shape[:2]
        cx1, cy1 = int(0.2 * w), int(0.2 * h)
        cx2, cy2 = int(0.8 * w), int(0.8 * h)
        core = roi[cy1:cy2, cx1:cx2] if (cx2 > cx1 and cy2 > cy1) else roi

        mean_bgr = np.mean(core.reshape(-1, 3), axis=0).astype(np.float32)
        mean_rgb = mean_bgr[::-1]

        lab = cv2.cvtColor(core, cv2.COLOR_BGR2LAB)
        lab_mean = np.mean(lab.reshape(-1, 3), axis=0).astype(np.float32)
        dom_lab = np.percentile(lab.reshape(-1, 3), 75, axis=0).astype(np.float32)

        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).astype(np.float32)
        hist = cv2.normalize(hist, None).flatten()

        # dominant color by largest channel in RGB
        dom_idx = int(np.argmax(mean_rgb))
        dominant = ["red", "green", "blue"][dom_idx]

        return {
            "mean_rgb": mean_rgb.tolist(),
            "lab_mean": lab_mean.tolist(),
            "dom_lab": dom_lab.tolist(),
            "hsv_hist": hist.tolist(),
            "dominant": dominant,
        }

    def add_track_sample(
        self,
        track_id: int,
        frame_id: int,
        timestamp: float,
        center: Sequence[float],
        velocity: Sequence[float],
        team_color: Optional[str] = None,
        predicted: bool = False,
        confidence: float = 1.0,
        color_feature: Optional[Dict[str, Any]] = None,
    ) -> None:
        sample = TrajectorySample(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            track_id=int(track_id),
            x=float(center[0]),
            y=float(center[1]),
            predicted=bool(predicted),
            confidence=float(confidence),
            vx=float(velocity[0]),
            vy=float(velocity[1]),
            team_color=str(team_color or "unknown"),
        )
        self.samples.append(sample)
        self._track_history.setdefault(sample.track_id, []).append(sample)
        c = str(sample.team_color or "unknown").lower()
        if c in {"white", "red", "green", "purple", "blue"}:
            self._color_votes[sample.track_id][c] += 1
        if color_feature is not None and not bool(predicted):
            self._track_color_history[sample.track_id].append(color_feature)

    def update_from_tracks(self, tracks: List[Dict[str, Any]], frame_id: int, timestamp: float, frame: Optional[np.ndarray] = None) -> None:
        for track in tracks:
            center = track.get("smoothed_center") or track.get("center")
            if center is None:
                continue
            color_feature = None
            if frame is not None and track.get("bbox") is not None:
                color_feature = self._extract_color_feature(frame, track.get("bbox", [0, 0, 0, 0]))

            self.add_track_sample(
                track_id=track["track_id"],
                frame_id=frame_id,
                timestamp=timestamp,
                center=center,
                velocity=track.get("velocity", (0.0, 0.0)),
                team_color=track.get("stable_team") or track.get("team_color"),
                predicted=bool(track.get("lost_frames", 0) > 0),
                confidence=float(track.get("confidence", 1.0)),
                color_feature=color_feature,
            )

    def get_track_history(self, track_id: int) -> List[Dict[str, Any]]:
        return [sample.to_dict() for sample in self._track_history.get(int(track_id), [])]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame([sample.to_dict() for sample in self.samples])
        if len(df) == 0:
            if "final_team" not in df.columns:
                df["final_team"] = []
            if "team" not in df.columns:
                df["team"] = []
            return df

        team_map = self._cluster_teams()
        df["final_team"] = df["track_id"].astype(int).map(lambda x: team_map.get(int(x), "unknown"))
        # alias for downstream compatibility
        df["team"] = df["final_team"]
        return df

    def export_csv(self, path: str) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(output_path, index=False)
        return str(output_path)

    def _track_avg_appearance(self, track_id: int) -> Optional[np.ndarray]:
        feats = self._track_color_history.get(track_id, [])
        if len(feats) == 0:
            return None
        lab_mean = np.mean(np.asarray([f["lab_mean"] for f in feats], dtype=np.float32), axis=0)
        dom_lab = np.mean(np.asarray([f["dom_lab"] for f in feats], dtype=np.float32), axis=0)
        hsv_hist = np.mean(np.asarray([f["hsv_hist"] for f in feats], dtype=np.float32), axis=0)
        return np.concatenate([lab_mean, dom_lab, hsv_hist], axis=0).astype(np.float32)

    def _cluster_teams(self) -> Dict[int, str]:
        track_ids: List[int] = []
        vectors: List[np.ndarray] = []
        for tid in self._track_history.keys():
            v = self._track_avg_appearance(tid)
            if v is None:
                continue
            track_ids.append(int(tid))
            vectors.append(v)

        if len(vectors) == 0:
            return {int(tid): "unknown" for tid in self._track_history.keys()}

        X = np.vstack(vectors).astype(np.float32)
        k = int(max(2, min(self.n_team_clusters, len(track_ids))))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
        _compactness, labels, _centers = cv2.kmeans(X, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)

        out: Dict[int, str] = {}
        for tid, lb in zip(track_ids, labels.reshape(-1).tolist()):
            out[int(tid)] = f"team_{int(lb)}"

        for tid in self._track_history.keys():
            out.setdefault(int(tid), "unknown")
        return out

    def export_json(self, path: str) -> str:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        team_map = self._cluster_teams()
        payload = {
            "team_clusters": sorted(set(team_map.values())),
            "tracks": {
                str(track_id): {
                    "track_id": int(track_id),
                    "final_team": team_map.get(int(track_id), "unknown"),
                    "color_votes": dict(self._color_votes.get(track_id, {})),
                    "color_history_len": len(self._track_color_history.get(track_id, [])),
                    "trajectory": [sample.to_dict() for sample in samples],
                    "statistics": {
                        "num_points": len(samples),
                    },
                }
                for track_id, samples in self._track_history.items()
            },
            "samples": [sample.to_dict() for sample in self.samples],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output_path)