from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


class HeatmapGenerator:
    def __init__(self, blur_ksize: int = 31) -> None:
        self.blur_ksize = int(blur_ksize) if int(blur_ksize) % 2 == 1 else int(blur_ksize) + 1

    def generate_from_tracks(self, tracks_payload: Dict[str, List[dict]], frame_shape: tuple, output_path: str) -> str:
        h, w = frame_shape[:2]
        density = np.zeros((h, w), dtype=np.float32)

        for _, samples in tracks_payload.items():
            for s in samples:
                x = int(round(float(s["x"])))
                y = int(round(float(s["y"])))
                if 0 <= x < w and 0 <= y < h:
                    density[y, x] += 1.0

        if np.max(density) > 0:
            density = density / np.max(density)
        density_blur = cv2.GaussianBlur(density, (self.blur_ksize, self.blur_ksize), 0)
        heat_u8 = np.clip(density_blur * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), colored)
        return str(out)
