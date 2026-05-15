from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class RegistrationResult:
    homography: np.ndarray
    inliers: int
    matches: int
    score: float
    debug_image: Optional[np.ndarray] = None


class MapRegistrar:
    """Automatic minimap->global-map registration using ORB + RANSAC."""

    def __init__(self, nfeatures: int = 3000, ratio_test: float = 0.8, ransac_thresh: float = 4.0) -> None:
        self.nfeatures = int(nfeatures)
        self.ratio_test = float(ratio_test)
        self.ransac_thresh = float(ransac_thresh)

    @staticmethod
    def _resolve_path(value: str, base_dir: Path) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        c1 = (base_dir / p).resolve()
        if c1.exists():
            return c1
        c2 = (base_dir.parent / p).resolve()
        if c2.exists():
            return c2
        return c2

    @staticmethod
    def extract_minimap_reference_from_config(config_path: str, out_path: str) -> str:
        cfg_path = Path(config_path).resolve()
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        base_dir = cfg_path.parent

        video_path = MapRegistrar._resolve_path(cfg["video_path"], base_dir)
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read first frame from video: {video_path}")

        roi = cfg.get("minimap", {}).get("minimap_roi")
        if roi:
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"])
            frame = frame[y : y + h, x : x + w]

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        return str(out)

    def register(self, minimap_img: np.ndarray, map_img: np.ndarray) -> RegistrationResult:
        gray1 = cv2.cvtColor(minimap_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=self.nfeatures)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            raise RuntimeError("Not enough features for registration")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf.knnMatch(des1, des2, k=2)

        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)

        if len(good) < 8:
            raise RuntimeError(f"Not enough good matches: {len(good)}")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        if H is None or mask is None:
            raise RuntimeError("findHomography failed")

        inliers = int(mask.ravel().sum())
        matches = int(len(good))
        score = float(inliers / max(matches, 1))

        draw_mask = mask.ravel().tolist()
        dbg = cv2.drawMatches(
            minimap_img,
            kp1,
            map_img,
            kp2,
            good,
            None,
            matchesMask=draw_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        return RegistrationResult(homography=H.astype(np.float32), inliers=inliers, matches=matches, score=score, debug_image=dbg)

    @staticmethod
    def save_matrix(matrix: np.ndarray, path: str) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(np.asarray(matrix, dtype=float).tolist()), encoding="utf-8")
        return str(out)

    @staticmethod
    def load_matrix(path: str) -> np.ndarray:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return np.asarray(payload, dtype=np.float32)

    @staticmethod
    def save_debug(image: np.ndarray, path: str) -> str:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)
        return str(out)

    @staticmethod
    def map_points(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float32))
        return mapped.reshape(-1, 2)

    @staticmethod
    def map_trajectory_csv(trajectory_csv: str, output_csv: str, H: np.ndarray) -> str:
        import pandas as pd
        from pandas.errors import EmptyDataError

        in_path = Path(trajectory_csv)
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise FileNotFoundError(f"Trajectory CSV not found: {in_path}")

        try:
            df = pd.read_csv(in_path)
        except EmptyDataError:
            # write an empty but schema-correct mapped CSV
            empty_cols = ["frame_id", "timestamp", "track_id", "x", "y", "predicted", "confidence", "map_x", "map_y"]
            pd.DataFrame(columns=empty_cols).to_csv(out, index=False)
            return str(out)

        if len(df) == 0:
            if "map_x" not in df.columns:
                df["map_x"] = []
            if "map_y" not in df.columns:
                df["map_y"] = []
            df.to_csv(out, index=False)
            return str(out)

        required = {"x", "y"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Trajectory CSV missing required columns: {missing}")

        mapped = MapRegistrar.map_points(df[["x", "y"]].to_numpy(dtype=np.float32), H)
        df["map_x"] = mapped[:, 0]
        df["map_y"] = mapped[:, 1]
        df.to_csv(out, index=False)
        return str(out)
