from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class AdvancedHeatmap:
    def __init__(self, blur_ksize: int = 41, alpha: float = 0.56) -> None:
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.alpha = float(alpha)

    @staticmethod
    def _normalize_percentile(arr: np.ndarray, p: float = 99.0) -> np.ndarray:
        nz = arr[arr > 0]
        if nz.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        hi = float(np.percentile(nz, p))
        hi = max(hi, 1e-6)
        return np.clip(arr / hi, 0.0, 1.0)

    @staticmethod
    def _add_title(img: np.ndarray, title: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (10, 10), (280, 46), (0, 0, 0), -1)
        cv2.putText(out, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    @staticmethod
    def _add_colorbar(img: np.ndarray, label: str) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        bar_h = 14
        x0, x1 = w - 230, w - 30
        y0, y1 = h - 30, h - 16
        grad = np.linspace(0, 255, x1 - x0, dtype=np.uint8).reshape(1, -1)
        grad = np.repeat(grad, bar_h, axis=0)
        bar = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
        out[y0:y1, x0:x1] = bar
        cv2.rectangle(out, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (255, 255, 255), 1)
        cv2.putText(out, "Low", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(out, "High", (x1 - 36, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(out, label, (x0 - 2, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
        return out

    def generate(self, mapped_csv: str, output_dir: str, map_image: str | None = None) -> tuple[str, str, str]:
        df = pd.read_csv(mapped_csv)
        if len(df) == 0:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            base = np.zeros((256, 256, 3), dtype=np.uint8)
            d = str(out / "density_heatmap.png")
            s = str(out / "speed_heatmap.png")
            st = str(out / "stay_heatmap.png")
            cv2.imwrite(d, base)
            cv2.imwrite(s, base)
            cv2.imwrite(st, base)
            return d, s, st
        if map_image is not None and Path(map_image).exists():
            base = cv2.imread(str(map_image))
            if base is None:
                raise FileNotFoundError(f"Cannot open map image: {map_image}")
            h, w = base.shape[:2]
        else:
            max_x = int(np.ceil(df["map_x"].max())) + 2
            max_y = int(np.ceil(df["map_y"].max())) + 2
            w = max(64, max_x)
            h = max(64, max_y)
            base = np.zeros((h, w, 3), dtype=np.uint8)

        density = np.zeros((h, w), dtype=np.float32)
        speed_acc = np.zeros((h, w), dtype=np.float32)
        speed_cnt = np.zeros((h, w), dtype=np.float32)
        stay = np.zeros((h, w), dtype=np.float32)

        for _, g in df.groupby("track_id"):
            g = g.sort_values("frame_id")
            pts = g[["map_x", "map_y"]].to_numpy(dtype=float)
            for i, p in enumerate(pts):
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= x < w and 0 <= y < h:
                    density[y, x] += 1.0
                    stay[y, x] += 1.0
                    if i > 0:
                        v = float(np.linalg.norm(pts[i] - pts[i - 1]))
                        speed_acc[y, x] += v
                        speed_cnt[y, x] += 1.0

        speed = np.divide(speed_acc, np.maximum(speed_cnt, 1.0))

        def to_overlay(arr: np.ndarray, title: str, label: str) -> np.ndarray:
            b = cv2.GaussianBlur(arr, (self.blur_ksize, self.blur_ksize), 0)
            n = self._normalize_percentile(b, p=99.0)
            u8 = np.clip(255.0 * n, 0, 255).astype(np.uint8)
            color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
            out = cv2.addWeighted(base, 1.0 - self.alpha, color, self.alpha, 0)
            out = self._add_title(out, title)
            out = self._add_colorbar(out, label)
            return out

        density_img = to_overlay(density, "Density Heatmap", "Visit Frequency")
        speed_img = to_overlay(speed, "Speed Heatmap", "Speed")
        stay_img = to_overlay(stay, "Stay Heatmap", "Dwell Time")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        d = str(out / "density_heatmap.png")
        s = str(out / "speed_heatmap.png")
        st = str(out / "stay_heatmap.png")
        cv2.imwrite(d, density_img)
        cv2.imwrite(s, speed_img)
        cv2.imwrite(st, stay_img)
        return d, s, st
