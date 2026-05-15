"""
Occlusion Manager: Handles track state management and ReID recovery under occlusion.

Features:
- Track state machine (ACTIVE, OCCLUDED, LOST, REMOVED)
- IOU and distance-based occlusion detection
- Multi-feature ReID matching (position, velocity, direction, appearance)
- Track memory (stable_color, avg_speed, direction, HSV histogram)
- Trajectory smoothing (raw_history + smooth_history)
- Hungarian matching with weighted cost function
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(Enum):
    """Track lifecycle states."""
    ACTIVE = 1
    OCCLUDED = 2
    LOST = 3
    REMOVED = 4


@dataclass
class TrackMemory:
    """Persistent track memory for ReID across occlusion."""
    
    stable_color: str = "unknown"
    avg_speed: float = 0.0
    direction: List[float] = field(default_factory=lambda: [0.0, 0.0])  # unit vector
    appearance_histogram: Optional[np.ndarray] = None
    color_hist: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    
    # Appearance features (if available)
    appearance_feat: Optional[np.ndarray] = None
    
    def update_from_track(self, track_state: Dict[str, Any]) -> None:
        """Update memory from track state."""
        self.stable_color = track_state.get("stable_team", track_state.get("stable_color", "unknown"))
        
        # Update avg speed (EMA)
        vel = track_state.get("velocity", [0.0, 0.0])
        speed = float(np.linalg.norm(vel))
        if self.avg_speed == 0.0:
            self.avg_speed = speed
        else:
            self.avg_speed = 0.7 * self.avg_speed + 0.3 * speed
        
        # Update direction (unit vector of velocity)
        if speed > 0.1:  # only if moving significantly
            vel_arr = np.asarray(vel, dtype=float)
            self.direction = (vel_arr / speed).tolist()
    
    def compute_appearance_histogram(self, frame: np.ndarray, bbox: Sequence[int]) -> None:
        """Compute Lab histogram from bbox region for ReID appearance."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            region = frame[y1:y2, x1:x2]
            if region.size > 0:
                lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                self.appearance_histogram = hist


@dataclass
class OccludedTrack:
    """Temporary record for occluded/lost track for potential ReID."""
    
    track_id: int
    state: TrackState
    last_center: List[float]
    last_bbox: List[int]
    memory: TrackMemory
    
    prediction_center: List[float] = field(default_factory=lambda: [0.0, 0.0])
    prediction_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    
    # Occlusion timeline
    occlusion_start_frame: int = 0
    last_update_frame: int = 0
    
    # Smoothed trajectory
    raw_history: Deque[List[float]] = field(default_factory=lambda: deque(maxlen=300))
    smooth_history: Deque[List[float]] = field(default_factory=lambda: deque(maxlen=300))
    
    smoothing_beta: float = 0.8


class OcclusionManager:
    """Manages track states and ReID under occlusion."""
    
    def __init__(
        self,
        max_missing_frames: int = 45,
        temp_occluded_protect: int = 10,
        iou_threshold: float = 0.3,
        distance_threshold: float = 50.0,
        reid_position_weight: float = 0.5,
        reid_velocity_weight: float = 0.2,
        reid_direction_weight: float = 0.2,
        reid_appearance_weight: float = 0.1,
    ) -> None:
        """
        Args:
            max_missing_frames: Max frames to keep track before removal (30-60 typical)
            temp_occluded_protect: Frames to protect track in occluded state before lost
            iou_threshold: IOU threshold for occlusion detection
            distance_threshold: Distance threshold for occlusion detection
            reid_*_weight: Cost weights for multi-feature ReID matching
        """
        self.max_missing_frames = int(max_missing_frames)
        self.temp_occluded_protect = int(temp_occluded_protect)
        self.iou_threshold = float(iou_threshold)
        self.distance_threshold = float(distance_threshold)
        
        # ReID cost weights
        self.reid_position_weight = float(reid_position_weight)
        self.reid_velocity_weight = float(reid_velocity_weight)
        self.reid_direction_weight = float(reid_direction_weight)
        self.reid_appearance_weight = float(reid_appearance_weight)
        
        # Occluded/archived tracks (for recovery)
        self.occluded_tracks: Dict[int, OccludedTrack] = {}
        self._frame_counter = 0
    
    def update_frame_counter(self) -> None:
        """Increment frame counter."""
        self._frame_counter += 1
    
    @staticmethod
    def _iou(box1: Sequence[int], box2: Sequence[int]) -> float:
        """Compute IOU between two bboxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        return float(inter_area) / float(union_area)
    
    @staticmethod
    def _center_distance(a: Sequence[float], b: Sequence[float]) -> float:
        """Euclidean distance between two points."""
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    
    @staticmethod
    def _direction_similarity(dir1: Sequence[float], dir2: Sequence[float]) -> float:
        """Cosine similarity between two direction vectors (0-1, higher is more similar)."""
        d1 = np.asarray(dir1, dtype=float)
        d2 = np.asarray(dir2, dtype=float)
        norm1 = np.linalg.norm(d1)
        norm2 = np.linalg.norm(d2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.5  # unknown direction, neutral
        
        sim = float(np.dot(d1, d2)) / (norm1 * norm2)
        return max(0.0, min(1.0, sim))  # clamp to [0, 1]
    
    def detect_occlusion(self, tracks_dicts: List[Dict[str, Any]]) -> Dict[int, bool]:
        """
        Detect if each track is occluded based on IOU with other tracks.
        
        Returns:
            Dict[track_id] -> is_occluded (bool)
        """
        occluded = {}
        n = len(tracks_dicts)
        
        for i, trk_i in enumerate(tracks_dicts):
            tid_i = trk_i.get("track_id", -1)
            bbox_i = trk_i.get("bbox", [0, 0, 0, 0])
            
            is_occluded_i = False
            for j, trk_j in enumerate(tracks_dicts):
                if i == j:
                    continue
                bbox_j = trk_j.get("bbox", [0, 0, 0, 0])
                
                # If significant IOU or very close, mark as occluded
                iou = self._iou(bbox_i, bbox_j)
                if iou > self.iou_threshold:
                    is_occluded_i = True
                    break
            
            if tid_i >= 0:
                occluded[tid_i] = is_occluded_i
        
        return occluded
    
    def update_occlusion_state(
        self,
        active_tracks: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
    ) -> Dict[int, TrackState]:
        """
        Update track states based on occlusion detection.
        
        Returns:
            Dict[track_id] -> TrackState
        """
        states: Dict[int, TrackState] = {}
        occluded = self.detect_occlusion(active_tracks)
        
        for trk in active_tracks:
            tid = trk.get("track_id", -1)
            if tid < 0:
                continue
            
            lost_frames = trk.get("lost_frames", 0)
            
            if occluded.get(tid, False):
                if lost_frames <= self.temp_occluded_protect:
                    states[tid] = TrackState.OCCLUDED
                else:
                    states[tid] = TrackState.LOST
            elif lost_frames == 0:
                states[tid] = TrackState.ACTIVE
            else:
                states[tid] = TrackState.LOST
        
        return states
    
    def reid_cost(
        self,
        occluded_track: OccludedTrack,
        detection: Dict[str, Any],
    ) -> float:
        """
        Compute multi-feature ReID cost between occluded track and detection.
        
        Cost combines:
        - Position distance (0.5 weight)
        - Velocity magnitude difference (0.2 weight)
        - Direction similarity (0.2 weight, inverted to cost)
        - Appearance HSV similarity (0.1 weight)
        
        Lower cost = better match.
        """
        costs = []
        
        # 1. Position cost
        pos_dist = self._center_distance(occluded_track.last_center, detection["center"])
        pos_cost = min(pos_dist / 100.0, 1.0)  # normalize to [0, 1]
        costs.append(("position", self.reid_position_weight * pos_cost))
        
        # 2. Velocity cost
        detection_vel = detection.get("velocity", [0.0, 0.0])
        det_speed = float(np.linalg.norm(detection_vel))
        speed_diff = abs(occluded_track.memory.avg_speed - det_speed)
        vel_cost = min(speed_diff / 50.0, 1.0)
        costs.append(("velocity", self.reid_velocity_weight * vel_cost))
        
        # 3. Direction cost (inverted similarity)
        detection_dir = detection.get("velocity", [0.0, 0.0])
        dir_sim = self._direction_similarity(occluded_track.memory.direction, detection_dir)
        dir_cost = 1.0 - dir_sim
        costs.append(("direction", self.reid_direction_weight * dir_cost))
        
        # 4. Appearance cost (Lab histogram)
        app_cost = 0.0
        if occluded_track.memory.appearance_histogram is not None and "appearance_histogram" in detection:
            det_hist = detection.get("appearance_histogram")
            if det_hist is not None:
                # Chi-square distance between histograms
                chi2 = cv2.compareHist(
                    occluded_track.memory.appearance_histogram,
                    det_hist,
                    cv2.HISTCMP_CHISQR,
                )
                app_cost = min(chi2 / 10.0, 1.0)  # normalize
        costs.append(("appearance", self.reid_appearance_weight * app_cost))
        
        total_cost = sum(w for _, w in costs)
        return float(total_cost)
    
    def reid_matching(
        self,
        occluded_list: List[OccludedTrack],
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        ReID matching between occluded tracks and new detections.
        
        Returns:
            (matches: [(occluded_idx, det_idx), ...], unmatched_occluded, unmatched_dets)
        """
        if not occluded_list or not detections:
            return [], list(range(len(occluded_list))), list(range(len(detections)))
        
        # Build cost matrix
        cost_matrix = np.full(
            (len(occluded_list), len(detections)),
            fill_value=float("inf"),
            dtype=float,
        )
        
        for i, occ_trk in enumerate(occluded_list):
            for j, det in enumerate(detections):
                cost = self.reid_cost(occ_trk, det)
                cost_matrix[i, j] = cost
        
        # Hungarian assignment (with cost threshold)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches: List[Tuple[int, int]] = []
        unmatched_occ = set(range(len(occluded_list)))
        unmatched_det = set(range(len(detections)))
        
        cost_threshold = 1.0  # reject matches with cost > threshold
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < cost_threshold:
                matches.append((r, c))
                unmatched_occ.discard(r)
                unmatched_det.discard(c)
        
        return matches, sorted(unmatched_occ), sorted(unmatched_det)
    
    def smooth_path(
        self,
        raw_center: List[float],
        smooth_center: List[float],
        beta: float = 0.8,
    ) -> List[float]:
        """
        EMA smoothing: smooth_t = beta * smooth_{t-1} + (1-beta) * raw_t
        """
        smoothed = [
            beta * smooth_center[0] + (1 - beta) * raw_center[0],
            beta * smooth_center[1] + (1 - beta) * raw_center[1],
        ]
        return smoothed
    
    def archive_track(
        self,
        track: Dict[str, Any],
        memory: TrackMemory,
    ) -> OccludedTrack:
        """Convert active track to occluded track record."""
        occ_trk = OccludedTrack(
            track_id=track.get("track_id", -1),
            state=TrackState.OCCLUDED,
            last_center=list(track.get("center", [0.0, 0.0])),
            last_bbox=list(track.get("bbox", [0, 0, 0, 0])),
            memory=memory,
            prediction_center=list(track.get("predicted_center", [0.0, 0.0])),
            prediction_velocity=list(track.get("predicted_velocity", [0.0, 0.0])),
            occlusion_start_frame=self._frame_counter,
            last_update_frame=self._frame_counter,
        )
        
        # Initialize history
        if "history" in track:
            occ_trk.raw_history = deque(track["history"], maxlen=300)
        occ_trk.raw_history.append(occ_trk.last_center)
        occ_trk.smooth_history.append(occ_trk.last_center)
        
        return occ_trk
    
    def cleanup_old_occluded(self) -> None:
        """Remove occluded tracks that have been missing too long."""
        to_remove = []
        for tid in list(self.occluded_tracks.keys()):
            occ_trk = self.occluded_tracks[tid]
            frames_missing = self._frame_counter - occ_trk.last_update_frame
            if frames_missing > self.max_missing_frames:
                to_remove.append(tid)
        
        for tid in to_remove:
            del self.occluded_tracks[tid]
