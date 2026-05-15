from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from scipy.signal import savgol_filter

TEAM_COLORS = {
    "all": (0, 255, 255),
    "unknown": (0, 255, 255),
}

# vivid, high-contrast BGR palette for prettier route rendering
VIVID_PALETTE = [
    (255, 80, 80),    # vivid blue-ish red tone (BGR)
    (80, 200, 255),   # orange
    (80, 255, 120),   # spring green
    (255, 120, 255),  # magenta
    (255, 220, 80),   # gold
    (120, 255, 255),  # yellow-cyan
    (255, 140, 40),   # deep orange
    (180, 120, 255),  # violet
    (120, 220, 255),  # warm light blue
    (255, 120, 180),  # pink
]


def team_color(team: str) -> tuple[int, int, int]:
    if team in TEAM_COLORS:
        return TEAM_COLORS[team]
    idx = abs(hash(team)) % len(VIVID_PALETTE)
    return VIVID_PALETTE[idx]


def smooth_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 7:
        return pts
    win = min(len(pts) if len(pts) % 2 == 1 else len(pts) - 1, 11)
    if win < 5:
        return pts
    x = savgol_filter(pts[:, 0], window_length=win, polyorder=2, mode="interp")
    y = savgol_filter(pts[:, 1], window_length=win, polyorder=2, mode="interp")
    return np.stack([x, y], axis=1)


class ProfessionalRenderer:
    def __init__(self, max_jump: float = 40.0) -> None:
        self.max_jump = float(max_jump)

    def render_routes(self, map_img: np.ndarray, grouped_tracks: Dict[int, np.ndarray], team_by_track: Dict[int, str], team_filter: str = "all") -> np.ndarray:
        canvas = map_img.copy()
        overlay = canvas.copy()

        for tid, pts_raw in grouped_tracks.items():
            team = team_by_track.get(int(tid), "unknown")
            if team_filter != "all" and team != team_filter:
                continue
            base_color = team_color(team)
            pts = smooth_points(pts_raw.astype(np.float32))
            if len(pts) < 2:
                continue

            speeds = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.array([0.0])
            mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
            thickness = int(np.clip(2 + mean_speed / 4.0, 2, 8))

            for i in range(1, len(pts)):
                if np.linalg.norm(pts[i] - pts[i - 1]) > self.max_jump:
                    continue
                alpha = i / max(1, len(pts) - 1)
                color = (
                    int(np.clip(base_color[0] * (0.72 + 0.28 * alpha), 0, 255)),
                    int(np.clip(base_color[1] * (0.72 + 0.28 * alpha), 0, 255)),
                    int(np.clip(base_color[2] * (0.72 + 0.28 * alpha), 0, 255)),
                )
                p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                cv2.line(overlay, p0, p1, color, thickness + 2, cv2.LINE_AA)
                cv2.line(overlay, p0, p1, (255, 255, 255), max(1, thickness - 2), cv2.LINE_AA)

            sx, sy = int(round(pts[0][0])), int(round(pts[0][1]))
            ex, ey = int(round(pts[-1][0])), int(round(pts[-1][1]))
            cv2.circle(canvas, (sx, sy), 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (ex, ey), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"ID{int(tid)}", (ex + 6, ey - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, base_color, 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        y = 22
        for team in sorted(set(team_by_track.values())):
            col = team_color(team)
            cv2.rectangle(canvas, (8, y - 10), (22, y + 2), col, -1)
            cv2.putText(canvas, team, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
        return canvas
