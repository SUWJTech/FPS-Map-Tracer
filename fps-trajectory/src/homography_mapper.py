from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd


class HomographyMapper:
    def __init__(self, matrix: np.ndarray | None = None) -> None:
        self.matrix = matrix if matrix is not None else np.eye(3, dtype=np.float32)

    @staticmethod
    def find_homography(src_pts: Iterable[Tuple[float, float]], dst_pts: Iterable[Tuple[float, float]]) -> np.ndarray:
        src = np.asarray(list(src_pts), dtype=np.float32)
        dst = np.asarray(list(dst_pts), dtype=np.float32)
        if len(src) < 4 or len(dst) < 4:
            raise ValueError("Need at least 4 correspondence points")
        h, _ = cv2.findHomography(src, dst, method=0)
        if h is None:
            raise ValueError("Failed to estimate homography")
        return h.astype(np.float32)

    @staticmethod
    def _pick_points(window_name: str, image: np.ndarray, max_points: int = 4) -> List[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        canvas = image.copy()

        def _on_mouse(event, x, y, flags, param):
            nonlocal canvas
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < max_points:
                points.append((float(x), float(y)))
                cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(canvas, str(len(points)), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _on_mouse)
        while True:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == 10:  # Enter
                break
            if key == 27:  # ESC
                points = []
                break
        cv2.destroyWindow(window_name)
        return points

    def calibrate_from_images(self, minimap_img_path: str, map_img_path: str, max_points: int = 4) -> np.ndarray:
        minimap = cv2.imread(minimap_img_path)
        world = cv2.imread(map_img_path)
        if minimap is None or world is None:
            raise FileNotFoundError("Cannot open minimap or map image for calibration")

        src_pts = self._pick_points("Pick points on minimap (Enter to finish)", minimap, max_points=max_points)
        dst_pts = self._pick_points("Pick corresponding points on map (Enter to finish)", world, max_points=max_points)
        self.matrix = self.find_homography(src_pts, dst_pts)
        return self.matrix

    def set_matrix(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def save_matrix(self, path: str) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.matrix.tolist()), encoding="utf-8")
        return str(out)

    def load_matrix(self, path: str) -> np.ndarray:
        mat = json.loads(Path(path).read_text(encoding="utf-8"))
        self.matrix = np.asarray(mat, dtype=np.float32)
        return self.matrix

    def map_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, self.matrix)
        return mapped.reshape(-1, 2)

    def export_mapped_csv(self, trajectory_csv: str, output_csv: str) -> str:
        df = pd.read_csv(trajectory_csv)
        pts = df[["x", "y"]].to_numpy(dtype=np.float32)
        mapped = self.map_points(pts)
        df["map_x"] = mapped[:, 0]
        df["map_y"] = mapped[:, 1]
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return str(out)

    def debug_visualize(self, map_img_path: str, mapped_csv: str, out_path: str) -> str:
        base = cv2.imread(map_img_path)
        if base is None:
            raise FileNotFoundError(f"Cannot open map image: {map_img_path}")
        df = pd.read_csv(mapped_csv)
        for _, row in df.iterrows():
            x = int(round(float(row["map_x"])))
            y = int(round(float(row["map_y"])))
            if 0 <= x < base.shape[1] and 0 <= y < base.shape[0]:
                cv2.circle(base, (x, y), 1, (0, 255, 255), -1, cv2.LINE_AA)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), base)
        return str(out)

