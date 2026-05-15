from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

TEAM_COLOR_BGR = {
    "marker": (0, 255, 0),
    "unknown": (160, 160, 160),
}


class Visualizer:
    def __init__(self, draw_traj: bool = True, draw_ids: bool = True, line_thickness: int = 2, max_jump_draw: float = 35.0) -> None:
        self.draw_traj = bool(draw_traj)
        self.draw_ids = bool(draw_ids)
        self.line_thickness = max(1, int(line_thickness))
        self.max_jump_draw = float(max_jump_draw)

    @staticmethod
    def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(path), fourcc, float(fps), frame_size)

    @staticmethod
    def _color(name: Optional[str]) -> Tuple[int, int, int]:
        return TEAM_COLOR_BGR.get(str(name or "unknown").lower(), TEAM_COLOR_BGR["unknown"])

    @staticmethod
    def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
        cv2.putText(frame, text, (x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def draw_debug_frame(self, frame: np.ndarray, tracks: List[Dict[str, Any]], fps: Optional[float] = None) -> np.ndarray:
        out = frame.copy()

        for trk in tracks:
            # archived tracks include full history but may not have bbox/center
            if trk.get("archived"):
                continue
            bbox = trk.get("bbox")
            center = trk.get("smoothed_center", trk.get("center"))
            traj = trk.get("trajectory", [])

            color = TEAM_COLOR_BGR["marker"]

            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(out, (x1, y1), (x2, y2), color, self.line_thickness)

            if center is not None:
                cx, cy = [int(v) for v in center]
                cv2.circle(out, (cx, cy), max(2, self.line_thickness + 1), color, -1, cv2.LINE_AA)

            # draw trajectory with anti-alias and jump break
            if self.draw_traj and traj:
                pts = np.asarray(traj, dtype=np.int32)
                for i in range(1, len(pts)):
                    p0 = pts[i - 1]
                    p1 = pts[i]
                    if float(np.linalg.norm(p1.astype(float) - p0.astype(float))) > self.max_jump_draw:
                        continue
                    cv2.line(out, tuple(p0), tuple(p1), color, self.line_thickness, cv2.LINE_AA)

            if self.draw_ids:
                track_id = trk.get("track_id", "?")
                label = f"ID:{track_id}"

                if bbox is not None:
                    x1, y1, _, _ = [int(v) for v in bbox]
                    self._draw_label(out, label, x1, y1 - 5, color)
                elif center is not None:
                    cx, cy = [int(v) for v in center]
                    self._draw_label(out, label, cx, cy - 8, color)

        if fps is not None:
            cv2.putText(out, f"FPS: {fps:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        return out
