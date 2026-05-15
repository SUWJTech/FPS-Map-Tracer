from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class SpeedHeatmapGenerator:
    def __init__(self, blur_ksize: int = 31, alpha: float = 0.55) -> None:
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.alpha = float(alpha)

    def generate(self, mapped_csv: str, density_out: str, speed_out: str, map_image: str | None = None) -> tuple[str, str]:
        df = pd.read_csv(mapped_csv)
        if len(df) == 0:
            # emit empty fallback images
            base = np.zeros((256, 256, 3), dtype=np.uint8)
            Path(density_out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(density_out, base)
            cv2.imwrite(speed_out, base)
            return density_out, speed_out

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
        speed_sum = np.zeros((h, w), dtype=np.float32)
        speed_cnt = np.zeros((h, w), dtype=np.float32)

        for tid, g in df.groupby("track_id"):
            g = g.sort_values("frame_id")
            pts = g[["map_x", "map_y"]].to_numpy(dtype=float)
            for i, p in enumerate(pts):
                x, y = int(round(p[0])), int(round(p[1]))
                if not (0 <= x < w and 0 <= y < h):
                    continue
                density[y, x] += 1.0
                if i > 0:
                    v = float(np.linalg.norm(pts[i] - pts[i - 1]))
                    speed_sum[y, x] += v
                    speed_cnt[y, x] += 1.0

        avg_speed = np.divide(speed_sum, np.maximum(speed_cnt, 1.0))

        density_blur = cv2.GaussianBlur(density, (self.blur_ksize, self.blur_ksize), 0)
        speed_blur = cv2.GaussianBlur(avg_speed, (self.blur_ksize, self.blur_ksize), 0)

        d = (255.0 * density_blur / max(float(density_blur.max()), 1e-6)).astype(np.uint8)
        s = (255.0 * speed_blur / max(float(speed_blur.max()), 1e-6)).astype(np.uint8)

        d_img = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)

        # custom speed palette: blue(slow) -> yellow(mid) -> red(fast)
        speed_norm = s.astype(np.float32) / 255.0
        speed_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # BGR output
        speed_rgb[:, :, 0] = np.clip(255 * (1.0 - speed_norm), 0, 255).astype(np.uint8)  # blue down
        speed_rgb[:, :, 1] = np.clip(255 * np.minimum(speed_norm * 1.6, 1.0), 0, 255).astype(np.uint8)  # green/yellow mid
        speed_rgb[:, :, 2] = np.clip(255 * speed_norm, 0, 255).astype(np.uint8)  # red up

        d_img = cv2.addWeighted(base, 1.0 - self.alpha, d_img, self.alpha, 0)
        s_img = cv2.addWeighted(base, 1.0 - self.alpha, speed_rgb, self.alpha, 0)

        Path(density_out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(density_out, d_img)
        cv2.imwrite(speed_out, s_img)
        return density_out, speed_out
