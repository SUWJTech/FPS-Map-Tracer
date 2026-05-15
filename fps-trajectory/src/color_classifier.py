from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


TEAM_CLASSES = ("white", "green", "purple", "red", "yellow", "blue")


def normalize_color_name(name: str) -> str:
    key = str(name).lower().strip()
    if key.startswith("red"):
        return "red"
    if key.startswith("green"):
        return "green"
    if key.startswith("white"):
        return "white"
    if key.startswith("purple") or key.startswith("violet"):
        return "purple"
    if key.startswith("yellow"):
        return "yellow"
    if key.startswith("blue") or key.startswith("cyan"):
        return "blue"
    return key or "unknown"


def ensure_bbox(frame: np.ndarray, bbox: Sequence[int]) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


class ColorVotingEngine:
    """Track-level voting helper used by TeamEstimator."""

    def __init__(
        self,
        voting_window: int = 30,
        color_lock_frames: int = 10,
        color_change_threshold: float = 0.95,
    ) -> None:
        self.voting_window = max(3, int(voting_window))
        self.color_lock_frames = max(1, int(color_lock_frames))
        self.color_change_threshold = float(color_change_threshold)

    def vote(self, history: Sequence[str]) -> str:
        valid = [normalize_color_name(c) for c in history if normalize_color_name(c) != "unknown"]
        if not valid:
            return "unknown"
        return Counter(valid).most_common(1)[0][0]

    def update(
        self,
        history: Deque[str],
        observed_color: str,
        stable_color: str,
        color_locked: bool,
        consistent_frames: int,
        conflict_frames: int,
    ) -> Tuple[str, bool, int, int, float]:
        observed = normalize_color_name(observed_color)
        if observed != "unknown":
            history.append(observed)

        voted = self.vote(history)
        valid = [normalize_color_name(c) for c in history if normalize_color_name(c) != "unknown"]
        valid_count = len(valid)

        conflict_ratio = 0.0
        if valid_count > 0 and stable_color != "unknown":
            conflict_count = sum(1 for c in valid if c != stable_color)
            conflict_ratio = conflict_count / float(valid_count)

        if not color_locked:
            if voted != "unknown":
                stable_color = voted
            if observed != "unknown" and observed == stable_color:
                consistent_frames += 1
            else:
                consistent_frames = 0
            conflict_frames = 0
            if consistent_frames >= self.color_lock_frames and stable_color != "unknown":
                color_locked = True
        else:
            if observed != "unknown" and observed != stable_color:
                conflict_frames += 1
            else:
                conflict_frames = 0

            if (
                voted != "unknown"
                and voted != stable_color
                and conflict_ratio >= self.color_change_threshold
                and conflict_frames >= max(2, self.voting_window)
            ):
                stable_color = voted
                color_locked = False
                consistent_frames = 1 if observed == stable_color else 0
                conflict_frames = 0

        return stable_color, color_locked, consistent_frames, conflict_frames, conflict_ratio


@dataclass(frozen=True)
class LabReference:
    name: str
    bgr: Tuple[int, int, int]


class ColorClassifier:
    """Lab histogram classifier on marker-center ROI only (no full-bbox statistics)."""

    def __init__(
        self,
        reference_colors: Optional[Dict[str, Sequence[int]]] = None,
        roi_radius: int = 5,
        unknown_threshold: float = 0.40,
        margin_threshold: float = 0.10,
        white_brightness_threshold: float = 170.0,
        white_boost: float = 0.10,
        adaptive_alpha: float = 0.00,
        hist_bins_l: int = 8,
        hist_bins_ab: int = 8,
    ) -> None:
        self.roi_radius = max(2, int(roi_radius))
        self.unknown_threshold = float(unknown_threshold)
        self.margin_threshold = float(margin_threshold)
        self.white_brightness_threshold = float(white_brightness_threshold)
        self.white_boost = float(white_boost)
        self.adaptive_alpha = float(adaptive_alpha)
        self.hist_bins_l = max(4, int(hist_bins_l))
        self.hist_bins_ab = max(4, int(hist_bins_ab))

        default_refs = {
            "white": (245, 245, 245),
            "green": (40, 180, 40),
            "purple": (190, 70, 190),
            "red": (60, 60, 220),
            "yellow": (40, 220, 220),
            "blue": (220, 140, 40),
        }
        merged = default_refs.copy()
        if reference_colors:
            for k, v in reference_colors.items():
                name = normalize_color_name(k)
                if name in TEAM_CLASSES and len(v) >= 3:
                    merged[name] = (int(v[0]), int(v[1]), int(v[2]))

        self.reference_lab_means = {name: self._bgr_to_lab_mean(tuple(merged[name])) for name in TEAM_CLASSES}
        self.references = [LabReference(name=n, bgr=tuple(merged[n])) for n in TEAM_CLASSES]
        self.reference_hists = {ref.name: self._build_reference_hist(ref.bgr) for ref in self.references}

    @staticmethod
    def _bgr_to_lab_mean(bgr: Tuple[int, int, int]) -> np.ndarray:
        patch = np.full((6, 6, 3), bgr, dtype=np.uint8)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        return np.mean(lab.reshape(-1, 3), axis=0).astype(np.float32)

    def _build_reference_hist(self, bgr: Tuple[int, int, int]) -> np.ndarray:
        patch = np.full((12, 12, 3), bgr, dtype=np.uint8)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], None, [self.hist_bins_l, self.hist_bins_ab, self.hist_bins_ab], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist.astype(np.float32)

    def _extract_center_roi(self, frame: np.ndarray, bbox: Sequence[int]) -> Tuple[np.ndarray, Tuple[int, int]]:
        x1, y1, x2, y2 = ensure_bbox(frame, bbox)
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))
        r = self.roi_radius

        rx1 = max(0, cx - r)
        ry1 = max(0, cy - r)
        rx2 = min(frame.shape[1], cx + r)
        ry2 = min(frame.shape[0], cy + r)
        roi = frame[ry1:ry2, rx1:rx2]
        return roi, (cx, cy)

    def _adaptive_update_reference(self, color: str, roi_lab: np.ndarray, roi_hist: np.ndarray) -> None:
        if color not in self.reference_hists or roi_lab.size == 0:
            return
        obs_mean = np.mean(roi_lab.reshape(-1, 3), axis=0).astype(np.float32)
        self.reference_lab_means[color] = (1.0 - self.adaptive_alpha) * self.reference_lab_means[color] + self.adaptive_alpha * obs_mean
        ref_hist = self.reference_hists[color]
        blended = (1.0 - self.adaptive_alpha) * ref_hist + self.adaptive_alpha * roi_hist
        self.reference_hists[color] = cv2.normalize(blended, blended, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

    def _resolve_green_yellow_blue_purple(self, lab_mean: np.ndarray, top1: str, top2: str, score_gap: float) -> str:
        a = float(lab_mean[1])
        b = float(lab_mean[2])
        candidates = {top1, top2}
        if "yellow" in candidates and "green" in candidates and score_gap <= 0.08:
            return "yellow" if b >= 155 else "green"
        if "blue" in candidates and "purple" in candidates and score_gap <= 0.08:
            return "purple" if a >= 158 else "blue"
        return top1

    def classify(self, frame: np.ndarray, bbox: Sequence[int], return_debug: bool = False) -> Dict[str, Any]:
        roi, center = self._extract_center_roi(frame, bbox)
        if roi.size == 0:
            out = {"team_color": "unknown", "confidence": 0.0}
            if return_debug:
                out.update({"scores": {}, "centroid": center, "appearance_histogram": None, "mask": None})
            return out

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1, 2], None, [self.hist_bins_l, self.hist_bins_ab, self.hist_bins_ab], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)

        scores: Dict[str, float] = {}
        for name, ref_hist in self.reference_hists.items():
            score = float(cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CORREL))
            scores[name] = score

        if not scores:
            team_color = "unknown"
            confidence = 0.0
        else:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            (team_color, confidence) = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -1.0
            score_gap = float(confidence - second_score)

            lab_mean = np.mean(lab.reshape(-1, 3), axis=0).astype(np.float32)
            team_color = self._resolve_green_yellow_blue_purple(
                lab_mean,
                team_color,
                ranked[1][0] if len(ranked) > 1 else team_color,
                score_gap,
            )

            if team_color == "white":
                mean_brightness = float(np.mean(roi))
                if mean_brightness >= self.white_brightness_threshold:
                    confidence += self.white_boost

            if confidence < self.unknown_threshold or score_gap < self.margin_threshold:
                team_color = "unknown"

            if team_color != "unknown" and confidence >= (self.unknown_threshold + 0.10):
                self._adaptive_update_reference(team_color, lab, hist)

        out: Dict[str, Any] = {
            "team_color": team_color,
            "confidence": float(max(0.0, min(1.0, confidence))),
        }
        if return_debug:
            out["scores"] = scores
            out["centroid"] = (float(center[0]), float(center[1]))
            out["appearance_histogram"] = hist.flatten().astype(np.float32)
            out["mask"] = None
        return out

    def classify_batch(self, frame: np.ndarray, bboxes: Sequence[Sequence[int]], return_debug: bool = False) -> List[Dict[str, Any]]:
        return [self.classify(frame, bbox, return_debug=return_debug) for bbox in bboxes]


def load_color_classifier_from_config(cfg: Dict[str, Any]) -> ColorClassifier:
    return ColorClassifier(
        reference_colors=cfg.get("reference_colors"),
        roi_radius=cfg.get("roi_radius", 5),
        unknown_threshold=cfg.get("unknown_threshold", 0.40),
        margin_threshold=cfg.get("margin_threshold", 0.10),
        white_brightness_threshold=cfg.get("white_brightness_threshold", 170.0),
        white_boost=cfg.get("white_boost", 0.10),
        adaptive_alpha=cfg.get("adaptive_alpha", 0.00),
        hist_bins_l=cfg.get("hist_bins_l", 8),
        hist_bins_ab=cfg.get("hist_bins_ab", 8),
    )
