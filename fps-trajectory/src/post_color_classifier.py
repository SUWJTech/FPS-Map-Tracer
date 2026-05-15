from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


TEAM_BGR: Dict[str, Tuple[int, int, int]] = {
    "white": (245, 245, 245),
    "red": (60, 60, 220),
    "green": (40, 180, 40),
    "purple": (190, 70, 190),
    "blue": (220, 140, 40),
}


def _ratio_in_range(hsv: np.ndarray, rule: Dict[str, float]) -> float:
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (
        (h >= int(rule.get("h_min", 0)))
        & (h <= int(rule.get("h_max", 180)))
        & (s >= int(rule.get("s_min", 0)))
        & (s <= int(rule.get("s_max", 255)))
        & (v >= int(rule.get("v_min", 0)))
        & (v <= int(rule.get("v_max", 255)))
    )
    return float(mask.mean())


def classify_track_color(frame: np.ndarray, bbox: list[int], color_cfg: Dict[str, Any] | None = None) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "unknown"

    # robust center patch, reduce bbox border/background contamination
    h, w = roi.shape[:2]
    cx1, cy1 = int(0.25 * w), int(0.25 * h)
    cx2, cy2 = int(0.75 * w), int(0.75 * h)
    core = roi[cy1:cy2, cx1:cx2] if (cx2 > cx1 and cy2 > cy1) else roi

    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hch = hsv[:, :, 0].astype(np.float32)
    sch = hsv[:, :, 1].astype(np.float32)
    vch = hsv[:, :, 2].astype(np.float32)

    h_med = float(np.median(hch))
    s_med = float(np.median(sch))
    v_med = float(np.median(vch))

    # Config-driven HSV voting first
    if color_cfg:
        scores: Dict[str, float] = {}
        for team in ["white", "red", "green", "purple", "blue"]:
            rule = color_cfg.get(team)
            if rule is None:
                continue
            if isinstance(rule, list):
                score = max(_ratio_in_range(hsv, r) for r in rule)
            else:
                score = _ratio_in_range(hsv, rule)
            scores[team] = score
        if scores:
            best_team = max(scores.items(), key=lambda kv: kv[1])[0]
            best_score = scores[best_team]
            if best_score >= 0.08:
                return best_team

    # white: low saturation high value
    if s_med <= 70 and v_med >= 150:
        return "white"

    # red wraps hue domain
    if s_med >= 70 and ((h_med <= 12) or (h_med >= 165)):
        return "red"

    if s_med >= 70 and 35 <= h_med <= 90:
        return "green"

    if s_med >= 45 and 120 <= h_med <= 160:
        return "purple"

    # fallback to reference BGR distance
    mean_bgr = np.mean(core.reshape(-1, 3), axis=0)
    best_team = "unknown"
    best_dist = 1e18
    for team, ref in TEAM_BGR.items():
        d = float(np.linalg.norm(mean_bgr - np.asarray(ref, dtype=float)))
        if d < best_dist:
            best_dist = d
            best_team = team

    if best_dist > 130.0:
        return "unknown"
    return best_team
