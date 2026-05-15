import cv2
import os
from typing import Tuple, Iterator, Dict


class VideoReader:
    def __init__(self, path: str, roi: Dict = None, debug: bool = False):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_id = 0
        self.roi = roi or {"x": 0, "y": 0, "width": 0, "height": 0}
        self.debug = debug

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_id += 1
        if self.roi and self.roi.get("width", 0) > 0:
            x = int(self.roi.get("x", 0))
            y = int(self.roi.get("y", 0))
            w = int(self.roi.get("width", 0))
            h = int(self.roi.get("height", 0))
            frame = frame[y : y + h, x : x + w]
        timestamp = self.frame_id / self.fps
        return {"frame": frame, "frame_id": self.frame_id, "timestamp": timestamp}

    def release(self):
        self.cap.release()


def extract_map(map_path: str) -> "ndarray":
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)
    img = cv2.imread(map_path, cv2.IMREAD_COLOR)
    return img
