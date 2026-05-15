import cv2
import numpy as np
import json
from typing import List, Tuple


class HomographyMapper:
    def __init__(self, H=None):
        self.H = H

    def compute(self, src_pts: List[Tuple[float, float]], dst_pts: List[Tuple[float, float]]):
        src = np.array(src_pts, dtype=float)
        dst = np.array(dst_pts, dtype=float)
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
        self.H = H
        return H

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({'H': (self.H.tolist() if self.H is not None else None)}, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            d = json.load(f)
            if d.get('H'):
                self.H = np.array(d['H'])
        return self.H

    def pixel_to_map(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        if self.H is None:
            raise RuntimeError('Homography matrix not set')
        p = np.array([pt[0], pt[1], 1.0])
        q = self.H.dot(p)
        q = q / q[2]
        return (float(q[0]), float(q[1]))
