from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .kalman_filter import KalmanFilter2D
    from .occlusion_manager import OcclusionManager, TrackMemory, OccludedTrack, TrackState
except ImportError:  # pragma: no cover
    from kalman_filter import KalmanFilter2D
    from occlusion_manager import OcclusionManager, TrackMemory, OccludedTrack, TrackState


@dataclass
class Track:
    track_id: int
    bbox: List[int]
    center: List[float]
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    age: int = 0
    hits: int = 0
    lost_frames: int = 0
    trajectory: List[List[float]] = field(default_factory=list)

    # internal runtime fields
    score: float = 0.0
    confidence: float = 1.0
    predicted_center: List[float] = field(default_factory=lambda: [0.0, 0.0])
    predicted_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    use_kalman: bool = True
    last_observed_center: List[float] = field(default_factory=lambda: [0.0, 0.0])
    # persistent full history (permanent)
    history: Deque[List[float]] = field(default_factory=lambda: deque(maxlen=1000))
    # smoothed center for visualization
    smoothed_center: List[float] = field(default_factory=lambda: [0.0, 0.0])
    # occlusion management
    occlusion_memory: Optional[TrackMemory] = field(default_factory=TrackMemory)
    track_state: str = "ACTIVE"
    frames_in_state: int = 0
    appearance_histogram: Optional[Any] = None

    def initialize_filter(self, process_noise: float, measurement_noise: float) -> None:
        if self.use_kalman:
            self.kalman = KalmanFilter2D(process_noise=process_noise, measurement_noise=measurement_noise)
            self.kalman.initialize(self.center, self.velocity)
        else:
            self.kalman = None
        self.predicted_center = [float(self.center[0]), float(self.center[1])]
        self.predicted_velocity = [float(self.velocity[0]), float(self.velocity[1])]
        self.last_observed_center = [float(self.center[0]), float(self.center[1])]
        # initialize history and smoothed center
        self.history.clear()
        self.history.append([float(self.center[0]), float(self.center[1])])
        self.smoothed_center = [float(self.center[0]), float(self.center[1])]
        self.smooth_history = deque(maxlen=self.history.maxlen)
        self.smooth_history.append([float(self.smoothed_center[0]), float(self.smoothed_center[1])])

    def predict(self) -> None:
        if self.kalman is not None:
            state = self.kalman.predict()
            self.predicted_center = [state.x, state.y]
            self.predicted_velocity = [state.vx, state.vy]
            # do NOT overwrite self.center/self.velocity here; keep last observed
        else:
            self.predicted_center = [float(self.center[0]), float(self.center[1])]
            self.predicted_velocity = [float(self.velocity[0]), float(self.velocity[1])]
        self.age += 1
        # append predicted to history (will be distinguished by lost_frames > 0)
        self.history.append([float(self.predicted_center[0]), float(self.predicted_center[1])])
        # update smoothed center towards predicted to avoid jumps while missing
        b = 1.0 - getattr(self, "smoothed_center", [0.85])[0] if False else None
        # Use tracker-provided smoothing_beta if set externally; fallback to 0.85
        smoothing_beta = getattr(self, "smoothed_beta", 0.85)
        self.smoothed_center[0] = smoothing_beta * self.smoothed_center[0] + (1 - smoothing_beta) * float(self.predicted_center[0])
        self.smoothed_center[1] = smoothing_beta * self.smoothed_center[1] + (1 - smoothing_beta) * float(self.predicted_center[1])
        if hasattr(self, "smooth_history"):
            self.smooth_history.append([float(self.smoothed_center[0]), float(self.smoothed_center[1])])

    def update_from_detection(self, detection: Dict[str, Any]) -> None:
        # walkable area check: if tracker has mask and detection center is outside, reject update
        try:
            tracker_ref = getattr(self, "_tracker_ref", None)
            if tracker_ref is not None and not tracker_ref._is_walkable(detection.get("center", [0.0, 0.0])):
                # treat as missed detection
                self.mark_missed()
                return
        except Exception:
            pass
        if self.kalman is not None:
            state = self.kalman.update(detection["center"])
            self.center = [state.x, state.y]
            self.velocity = [state.vx, state.vy]
            self.predicted_center = [state.x, state.y]
            self.predicted_velocity = [state.vx, state.vy]
        else:
            new_center = [float(detection["center"][0]), float(detection["center"][1])]
            self.velocity = [new_center[0] - self.center[0], new_center[1] - self.center[1]]
            self.center = new_center
            self.predicted_center = new_center
            self.predicted_velocity = [float(self.velocity[0]), float(self.velocity[1])]
        self.bbox = [int(v) for v in detection["bbox"]]
        self.score = float(detection.get("conf", self.score))
        # update fused confidence (simple EMA of detection confidence and previous)
        det_conf = float(detection.get("conf", self.confidence))
        self.confidence = 0.7 * float(getattr(self, "confidence", 1.0)) + 0.3 * det_conf
        self.hits += 1
        self.lost_frames = 0
        self.last_observed_center = [float(self.center[0]), float(self.center[1])]
        self.trajectory.append([float(self.center[0]), float(self.center[1])])
        self.history.append([float(self.center[0]), float(self.center[1])])
        # update appearance histogram for reid
        if "appearance_histogram" in detection and detection["appearance_histogram"] is not None:
            self.appearance_histogram = detection["appearance_histogram"]
            if self.occlusion_memory is not None:
                self.occlusion_memory.appearance_histogram = self.appearance_histogram
        # update smoothed center towards observed (stronger trust)
        smoothing_beta = getattr(self, "smoothed_beta", 0.85)
        self.smoothed_center[0] = smoothing_beta * self.smoothed_center[0] + (1 - smoothing_beta) * float(self.center[0])
        self.smoothed_center[1] = smoothing_beta * self.smoothed_center[1] + (1 - smoothing_beta) * float(self.center[1])
        if hasattr(self, "smooth_history"):
            self.smooth_history.append([float(self.smoothed_center[0]), float(self.smoothed_center[1])])

    def mark_missed(self) -> None:
        self.lost_frames += 1

    def append_prediction_to_trajectory(self) -> None:
        # append smoothed predicted center to trajectory to avoid jumps
        point = [float(self.smoothed_center[0]), float(self.smoothed_center[1])]
        # respect walkable mask if present (tracker will host mask)
        try:
            tracker_ref = getattr(self, "_tracker_ref", None)
            if tracker_ref is not None and not tracker_ref._is_walkable(point):
                # do not append points outside walkable area
                return
        except Exception:
            pass
        self.trajectory.append(point)
        self.history.append(point)

    def to_dict(self, min_hits: int) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "center": self.center,
            "velocity": self.velocity,
            "age": self.age,
            "hits": self.hits,
            "lost_frames": self.lost_frames,
            "trajectory": self.trajectory,
            "predicted_center": self.predicted_center,
            "predicted_velocity": self.predicted_velocity,
            "history": list(self.history),
            "raw_history": list(self.history),
            "smooth_history": list(getattr(self, "smooth_history", [])),
            "smoothed_center": list(self.smoothed_center),
            "track_state": getattr(self, "track_state", "ACTIVE"),
            "confidence": float(getattr(self, "confidence", 1.0)),
            "appearance_histogram": self.appearance_histogram,
            "is_confirmed": self.hits >= min_hits,
        }


class ByteTrackStyleTracker:
    """Motion-based ByteTrack-style tracker with decoupled color estimation.

    Key points:
    - Matching uses Kalman predicted center (not last raw center)
    - Cost combines position + motion consistency
    - Distance gating prevents long-range ID jumps
    - Tracks survive short detector drops via prediction persistence
    - Color estimation is DECOUPLED via separate TeamEstimator layer
    - No color information in tracking logic
    """

    def __init__(
        self,
        max_distance: float = 60.0,
        max_lost: int = 30,
        min_hits: int = 3,
        high_conf_threshold: float = 0.5,
        low_conf_threshold: float = 0.1,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        position_weight: float = 0.7,
        motion_weight: float = 0.3,
        trajectory_length: int = 200,
        prediction_append_limit: int = 8,
        use_kalman: bool = True,
        use_motion_prediction: bool = True,
        max_speed: float = 120.0,
        keep_history_len: int = 1000,
        smoothing_beta: float = 0.85,
        # occlusion manager params
        max_missing_frames: int = 45,
        temp_occluded_protect: int = 10,
        iou_threshold: float = 0.3,
        reid_position_weight: float = 0.5,
        reid_velocity_weight: float = 0.2,
        reid_direction_weight: float = 0.2,
        reid_appearance_weight: float = 0.1,
        birth_frames: int = 3,
        enable_reid: bool = False,
        occlusion_freeze_updates: bool = True,
        birth_border_margin: float = 60.0,
        enable_birth_border_constraint: bool = False,
    ) -> None:
        self.max_distance = float(max_distance)
        self.max_lost = int(max_lost)
        self.min_hits = int(min_hits)
        self.high_conf_threshold = float(high_conf_threshold)
        self.low_conf_threshold = float(low_conf_threshold)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.position_weight = float(position_weight)
        self.motion_weight = float(motion_weight)
        self.trajectory_length = int(trajectory_length)
        self.prediction_append_limit = int(prediction_append_limit)
        self.use_kalman = bool(use_kalman)
        self.use_motion_prediction = bool(use_motion_prediction)

        self.tracks: List[Track] = []
        # archived tracks (removed due to max_lost) kept for visualization/history
        self.archived_tracks: List[Dict[str, Any]] = []
        self._next_track_id = 1
        self.max_speed = float(max_speed)
        self.keep_history_len = int(keep_history_len)
        self.smoothing_beta = float(smoothing_beta)
        # walkable mask (optional) - numpy uint8 mask where walkable pixels are >0
        self.walkable_mask: Optional[np.ndarray] = None
        # track birth delay (detection voting)
        self.birth_frames = max(1, int(birth_frames))
        self.birth_border_margin = float(birth_border_margin)
        self.enable_birth_border_constraint = bool(enable_birth_border_constraint)
        self.stats: Dict[str, int] = {"created_tracks": 0, "blocked_births": 0}
        self._pending_spawns: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.occlusion_freeze_updates = bool(occlusion_freeze_updates)
        self.frame_index = 0
        
        # occlusion manager
        self.occlusion_manager = OcclusionManager(
            max_missing_frames=max_missing_frames,
            temp_occluded_protect=temp_occluded_protect,
            iou_threshold=iou_threshold,
            distance_threshold=60.0,  # reuse max_distance
            reid_position_weight=reid_position_weight,
            reid_velocity_weight=reid_velocity_weight,
            reid_direction_weight=reid_direction_weight,
            reid_appearance_weight=reid_appearance_weight,
        )

    @staticmethod
    def _center_distance(a: Sequence[float], b: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    def _compute_motion_term(self, track: Track, detection_center: Sequence[float]) -> float:
        det_center = np.asarray(detection_center, dtype=float)
        last_obs = np.asarray(track.last_observed_center, dtype=float)
        pred_vel = np.asarray(track.predicted_velocity, dtype=float)
        observed_displacement = det_center - last_obs
        return float(np.linalg.norm(observed_displacement - pred_vel))

    def _is_walkable(self, point: Sequence[float]) -> bool:
        if self.walkable_mask is None:
            return True
        x = int(round(point[0]))
        y = int(round(point[1]))
        h, w = self.walkable_mask.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        return bool(self.walkable_mask[y, x] > 0)

    def _pair_cost(self, track: Track, detection: Dict[str, Any]) -> float:
        pos_dist = self._center_distance(track.predicted_center, detection["center"])

        # Motion gating: stop impossible associations to reduce ID switch.
        if pos_dist > self.max_distance:
            return float("inf")

        if self.use_motion_prediction:
            motion_term = self._compute_motion_term(track, detection["center"])
            return self.position_weight * pos_dist + self.motion_weight * motion_term
        return pos_dist

    def _multi_feature_cost(self, track: Track, detection: Dict[str, Any]) -> float:
        """
        Enhanced cost function with multi-feature matching:
        cost = 0.5*position + 0.2*velocity + 0.2*direction + 0.1*appearance
        """
        costs = {}
        
        # 1. Position (0.5 weight)
        pos_dist = self._center_distance(track.predicted_center, detection["center"])
        if pos_dist > self.max_distance:
            return float("inf")
        costs["position"] = min(pos_dist / 100.0, 1.0) * 0.5
        
        # 2. Velocity (0.2 weight)
        detection_vel = detection.get("velocity", [0.0, 0.0])
        track_speed = float(np.linalg.norm(track.velocity))
        det_speed = float(np.linalg.norm(detection_vel))
        speed_diff = abs(track_speed - det_speed)
        costs["velocity"] = min(speed_diff / 50.0, 1.0) * 0.2

        # Speed jump rejection: if speed difference unrealistic, reject association
        speed_jump_threshold = max(50.0, float(self.max_speed) * 1.5)
        if speed_diff > speed_jump_threshold:
            return float("inf")
        
        # 3. Direction (0.2 weight)
        if track_speed > 0.1:
            track_dir = (np.asarray(track.velocity) / track_speed).tolist()
            det_dir = detection_vel
            det_speed_norm = np.linalg.norm(det_dir)
            if det_speed_norm > 0.1:
                det_dir_norm = (np.asarray(det_dir) / det_speed_norm).tolist()
                dir_sim = float(np.dot(track_dir, det_dir_norm))
                dir_sim = max(0.0, min(1.0, dir_sim))
                costs["direction"] = (1.0 - dir_sim) * 0.2
            else:
                costs["direction"] = 0.2
        else:
            costs["direction"] = 0.1  # neutral
        
        # 4. Appearance (0.1 weight) - HSV histogram similarity
        app_cost = 0.0
        try:
            det_hist = detection.get("appearance_histogram")
            trk_hist = track.appearance_histogram
            if det_hist is not None and trk_hist is not None:
                # correlation similarity in [-1,1] -> clamp to [0,1]
                sim = float(cv2.compareHist(np.asarray(trk_hist, dtype=float), np.asarray(det_hist, dtype=float), cv2.HISTCMP_CORREL))
                sim = max(0.0, min(1.0, sim))
                app_cost = (1.0 - sim) * 0.2
        except Exception:
            app_cost = 0.0
        costs["appearance"] = app_cost
        
        total = sum(costs.values())
        return float(total)

    def _build_cost_matrix(self, tracks: List[Track], detections: List[Dict[str, Any]]) -> np.ndarray:
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=float)
        cost = np.full((len(tracks), len(detections)), fill_value=float("inf"), dtype=float)
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                cost[i, j] = self._multi_feature_cost(track, detection)
        return cost

    def _associate(
        self,
        tracks: List[Track],
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(tracks))), []

        cost = self._build_cost_matrix(tracks, detections)
        if not np.isfinite(cost).any():
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Replace inf for Hungarian solve and reject those pairs post-hoc.
        safe_cost = cost.copy()
        safe_cost[~np.isfinite(safe_cost)] = 1e9

        row_ind, col_ind = linear_sum_assignment(safe_cost)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if not np.isfinite(cost[r, c]):
                continue
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    def _split_detections(self, detections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        high, low = [], []
        for det in detections:
            conf = float(det.get("conf", 1.0))
            if conf >= self.high_conf_threshold:
                high.append(det)
            elif conf >= self.low_conf_threshold:
                low.append(det)
        return high, low

    def _new_track(self, detection: Dict[str, Any]) -> None:
        trk = Track(
            track_id=self._next_track_id,
            bbox=[int(v) for v in detection["bbox"]],
            center=[float(v) for v in detection["center"]],
            velocity=[0.0, 0.0],
            age=1,
            hits=1,
            lost_frames=0,
            trajectory=[],
            score=float(detection.get("conf", 1.0)),
            use_kalman=self.use_kalman,
        )
        trk.initialize_filter(self.process_noise, self.measurement_noise)
        trk.trajectory.append([float(trk.center[0]), float(trk.center[1])])
        # set history deque size from tracker config
        trk.history = deque(maxlen=self.keep_history_len)
        trk.history.append([float(trk.center[0]), float(trk.center[1])])
        # smoothing param from tracker
        trk.smoothed_beta = float(self.smoothing_beta)
        
        # Initialize occlusion memory (no color dependency)
        trk.occlusion_memory = TrackMemory(stable_color="unknown")
        trk.track_state = "ACTIVE"
        trk.frames_in_state = 0
        # seed appearance histogram if available
        if "appearance_histogram" in detection and detection["appearance_histogram"] is not None:
            trk.appearance_histogram = detection["appearance_histogram"]
            trk.occlusion_memory.appearance_histogram = trk.appearance_histogram

        trk.bbox = [int(v) for v in detection["bbox"]]

        # give track a backref to tracker for walkable checks
        trk._tracker_ref = self

        self.tracks.append(trk)
        self._next_track_id += 1
        self.stats["created_tracks"] += 1

    def _trim_trajectory(self, track: Track) -> None:
        if self.trajectory_length <= 0:
            return
        if len(track.trajectory) > self.trajectory_length:
            track.trajectory = track.trajectory[-self.trajectory_length :]

    def _prune(self) -> None:
        remaining: List[Track] = []
        for t in self.tracks:
            if t.lost_frames <= self.max_lost:
                remaining.append(t)
            else:
                # archive track for history/visualization
                info = t.to_dict(self.min_hits)
                info["archived"] = True
                # include full history
                info["history"] = list(t.history)
                self.archived_tracks.append(info)
                
                # also save to occlusion manager for potential ReID recovery
                if t.occlusion_memory:
                    t.occlusion_memory.update_from_track(info)
                    occ_trk = self.occlusion_manager.archive_track(info, t.occlusion_memory)
                    self.occlusion_manager.occluded_tracks[t.track_id] = occ_trk
        
        self.tracks = remaining
        self.occlusion_manager.cleanup_old_occluded()

    def _near_border(self, center: Sequence[float], frame_shape: Optional[Sequence[int]]) -> bool:
        if frame_shape is None:
            return True
        h, w = int(frame_shape[0]), int(frame_shape[1])
        x, y = float(center[0]), float(center[1])
        m = self.birth_border_margin
        return x <= m or y <= m or x >= (w - m) or y >= (h - m)

    def update(self, detections: List[Dict[str, Any]], frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Simple high-recall MOT: center-distance matching + Kalman smoothing."""
        self.frame_index += 1
        max_move_per_frame = 40.0
        for det in detections:
            bx = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [float(v) for v in bx]
            det["center"] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        for track in self.tracks:
            track.predict()

        matched_tracks = set()
        matched_dets = set()

        pairs: List[Tuple[float, int, int]] = []
        for ti, trk in enumerate(self.tracks):
            for di, det in enumerate(detections):
                dist = self._center_distance(trk.predicted_center, det["center"])
                if dist <= min(self.max_distance, max_move_per_frame):
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda x: x[0])

        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            trk = self.tracks[ti]
            det = detections[di]
            trk.update_from_detection(det)
            self._trim_trajectory(trk)
            matched_tracks.add(ti)
            matched_dets.add(di)

        for ti, trk in enumerate(self.tracks):
            if ti in matched_tracks:
                continue
            trk.mark_missed()
            if trk.lost_frames <= self.max_lost:
                trk.append_prediction_to_trajectory()
            self._trim_trajectory(trk)

        # Birth cache: detection must persist for birth_frames.
        expired_keys = []
        for key, info in self._pending_spawns.items():
            if self.frame_index - int(info.get("last_frame", 0)) > self.birth_frames:
                expired_keys.append(key)
        for key in expired_keys:
            self._pending_spawns.pop(key, None)

        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            cx, cy = det["center"]
            key = (int(round(cx / 4.0)), int(round(cy / 4.0)))
            entry = self._pending_spawns.get(key)
            if entry is None:
                self._pending_spawns[key] = {"count": 1, "last_frame": self.frame_index, "det": det}
            else:
                if int(entry.get("last_frame", 0)) == self.frame_index - 1:
                    entry["count"] = int(entry.get("count", 0)) + 1
                else:
                    entry["count"] = 1
                entry["last_frame"] = self.frame_index
                entry["det"] = det

            if int(self._pending_spawns[key]["count"]) >= self.birth_frames:
                if (not self.enable_birth_border_constraint) or self._near_border(self._pending_spawns[key]["det"]["center"], None if frame is None else frame.shape[:2]):
                    self._new_track(self._pending_spawns[key]["det"])
                else:
                    self.stats["blocked_births"] += 1
                self._pending_spawns.pop(key, None)

        self._prune()

        visible_tracks: List[Dict[str, Any]] = []
        for trk in self.tracks:
            if trk.hits >= self.min_hits:
                visible_tracks.append(trk.to_dict(self.min_hits))
        return visible_tracks


ByteTrackLite = ByteTrackStyleTracker
