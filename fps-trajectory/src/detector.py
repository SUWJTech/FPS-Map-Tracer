from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional when torch is unavailable
    torch = None

_HAS_ULTRALYTICS = True

logger = logging.getLogger("fps_trajectory")


@dataclass
class Detection:
    bbox: List[int]
    center: List[float]
    conf: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [int(v) for v in self.bbox],
            "center": [float(v) for v in self.center],
            "conf": float(self.conf),
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _custom_ultralytics_root() -> Optional[Path]:
    candidate = Path(__file__).resolve().parents[2] / "ultralytics-yolo11-main"
    if (candidate / "ultralytics").exists():
        return candidate.resolve()
    return None


def _load_yolo_class():
    """Load YOLO from the custom training repo when available, otherwise fall back to pip ultralytics."""
    custom_root = _custom_ultralytics_root()
    if custom_root is not None:
        sys.path.insert(0, str(custom_root))
        for module_name in list(sys.modules.keys()):
            if module_name == "ultralytics" or module_name.startswith("ultralytics."):
                del sys.modules[module_name]
        from ultralytics import YOLO as custom_yolo  # type: ignore

        logger.info("Using custom ultralytics package from %s", custom_root)
        return custom_yolo

    from ultralytics import YOLO as pip_yolo  # type: ignore

    return pip_yolo


def _resolve_weights_path(weights_path: Optional[str]) -> Path:
    candidates: List[Path] = []
    if weights_path:
        path = Path(weights_path)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append(_project_root() / path)
    candidates.extend(
        [
            _project_root() / "artifacts/yolov11-ASF-P2/weights/best.pt",
            _project_root() / "weights/best.pt",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find YOLO weights. Checked explicit path and default locations: "
        "artifacts/yolov11-ASF-P2/weights/best.pt, weights/best.pt."
    )


def _resolve_device(device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    if torch is not None:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    return "cpu"


class YOLODetector:
    """Ultralytics YOLO detector with explicit device auto-selection and batch inference."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_thresh: float = 0.25,
        device: Optional[str] = "auto",
        imgsz: int = 640,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> None:
        self.weights_path = _resolve_weights_path(weights_path)
        self.conf_thresh = float(conf_thresh)
        self.device = _resolve_device(device)
        self.imgsz = int(imgsz)
        self.batch_size = max(1, int(batch_size))
        self.verbose = verbose
        yolo_class = _load_yolo_class()
        self.model = yolo_class(str(self.weights_path))

        logger.info(
            "Loaded YOLO weights: %s | device=%s | conf=%.3f | imgsz=%d | batch=%d",
            self.weights_path,
            self.device,
            self.conf_thresh,
            self.imgsz,
            self.batch_size,
        )

    def _predict(self, source: Union[np.ndarray, Sequence[np.ndarray]], conf: Optional[float] = None):
        threshold = self.conf_thresh if conf is None else float(conf)
        return self.model.predict(
            source=source,
            conf=threshold,
            imgsz=self.imgsz,
            device=self.device,
            batch=self.batch_size,
            verbose=self.verbose,
        )

    @staticmethod
    def _result_to_detections(result) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            print("YOLO raw detections: 0")
            return detections

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        print(f"YOLO raw detections: {len(xyxy)}")

        for box, score in zip(xyxy, confs):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            detections.append(
                Detection(
                    bbox=[x1, y1, x2, y2],
                    center=[(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    conf=float(score),
                ).to_dict()
            )
        return detections

    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Dict[str, Any]]:
        """Run single-frame inference and return normalized detection dicts."""
        results = self._predict(frame, conf=conf)
        if not results:
            return []
        return self._result_to_detections(results[0])

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        conf: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Run batch inference and return detections per frame."""
        if not frames:
            return []
        results = self._predict(list(frames), conf=conf)
        return [self._result_to_detections(result) for result in results]

    def predict(self, frame: np.ndarray, conf: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.detect(frame, conf=conf)

    def predict_batch(
        self,
        frames: Sequence[np.ndarray],
        conf: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        return self.detect_batch(frames, conf=conf)


def make_detector(cfg: Dict[str, Any]):
    mode = cfg.get("mode", "yolo")
    if mode == "yolo":
        return YOLODetector(
            weights_path=cfg.get("weights") or cfg.get("model_path"),
            conf_thresh=cfg.get("conf", 0.25),
            device=cfg.get("device", "auto"),
            imgsz=cfg.get("imgsz", 640),
            batch_size=cfg.get("batch_size", 1),
            verbose=cfg.get("verbose", False),
        )

    raise ValueError(f"Unsupported detector mode: {mode}. This stage only implements YOLO inference.")
