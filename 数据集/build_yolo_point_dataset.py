from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

TARGET_COLORS_BGR = {
    "c1": (32, 224, 225),   # #e1e020 -> bgr
    "c2": (227, 75, 175),   # #af4be3 -> bgr
    "c3": (92, 214, 57),    # #39d65c -> bgr
    "c4": (231, 147, 67),   # #4393e7 -> bgr
    "c5": (95, 27, 234),    # #ea1b5f -> bgr
    "c6": (230, 230, 230),  # #e6e6e6 -> bgr
    "a1": (100, 41, 220),   # #dc2964 -> near-match
    "a2": (114, 224, 71),   # #47e072 -> near-match
    "a3": (188, 130, 206),  # #ce82bc -> near-match
}
COLOR_TOLERANCES = {
    # tight tolerances to avoid background/highlight false positives
    "c1": (24, 24, 24),
    "c2": (28, 28, 28),
    "c3": (12, 12, 12),
    "c4": (28, 28, 28),
    "c5": (28, 28, 28),
    "c6": (20, 20, 20),
    "a1": (22, 22, 22),
    "a2": (22, 22, 22),
    "a3": (22, 22, 22),
}
@dataclass
class Candidate:
    cx: float
    cy: float
    diameter: float


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_yolo(self, width: int, height: int) -> str:
        box_w = self.x2 - self.x1
        box_h = self.y2 - self.y1
        x_center = self.x1 + box_w / 2.0
        y_center = self.y1 + box_h / 2.0
        return f"0 {x_center / width:.6f} {y_center / height:.6f} {box_w / width:.6f} {box_h / height:.6f}"


@dataclass
class Sample:
    source_video: str
    frame_index: int
    image_path: Path
    candidates: list[Candidate]
    detections: list[Detection] | None = None


def iter_videos(input_dir: Path) -> list[Path]:
    videos = [path for path in sorted(input_dir.iterdir()) if path.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No video files found in {input_dir}")
    return videos


def build_masks(frame: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for color_name, target_bgr in TARGET_COLORS_BGR.items():
        tolerance = COLOR_TOLERANCES[color_name]
        lower = np.array(
            [max(0, target_bgr[0] - tolerance[0]), max(0, target_bgr[1] - tolerance[1]), max(0, target_bgr[2] - tolerance[2])],
            dtype=np.uint8,
        )
        upper = np.array(
            [min(255, target_bgr[0] + tolerance[0]), min(255, target_bgr[1] + tolerance[1]), min(255, target_bgr[2] + tolerance[2])],
            dtype=np.uint8,
        )
        bgr_mask = cv2.inRange(frame, lower, upper)

        # try HSV fallback if enabled via global flag _USE_HSV
        if globals().get("_USE_HSV", False):
            hsv_tol = globals().get("_HSV_TOL", (12, 60, 60))
            # convert single BGR color to HSV
            color_pixel = np.uint8([[list(target_bgr)]])
            hsv_pixel = cv2.cvtColor(color_pixel, cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
            ht, st, vt = hsv_tol
            lower_hsv = np.array([max(0, h - ht), max(0, s - st), max(0, v - vt)], dtype=np.uint8)
            upper_hsv = np.array([min(179, h + ht), min(255, s + st), min(255, v + vt)], dtype=np.uint8)
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
            color_mask = cv2.bitwise_or(bgr_mask, hsv_mask)
        else:
            color_mask = bgr_mask

        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        masks[color_name] = color_mask

    return masks


def detect_points(
    frame: np.ndarray,
    min_area: float,
    max_area: float,
    min_circularity: float,
    use_hsv: bool = False,
    hsv_tol: tuple = (12, 60, 60),
) -> list[Candidate]:
    candidates: list[Candidate] = []
    # set globals for build_masks to pick up
    globals()["_USE_HSV"] = bool(use_hsv)
    globals()["_HSV_TOL"] = tuple(hsv_tol)
    masks = build_masks(frame)
    seen_boxes = set()
    for color_name in tuple(TARGET_COLORS_BGR.keys()):
        mask = masks.get(color_name)
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            # per-color circularity override: tighten for green
            color_min_circ = min_circularity
            if color_name == "c3":
                color_min_circ = max(color_min_circ, 0.65)
            if circularity < color_min_circ:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.75 or aspect_ratio > 1.33:
                continue

            fill_ratio = area / float(w * h)
            # stricter fill for green UI artifacts
            color_min_fill = 0.45
            if color_name == "c3":
                color_min_fill = 0.6
            if fill_ratio < color_min_fill:
                continue

            if area < 10:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            diameter = max(w, h)

            # color consistency (stddev) inside contour
            mask_roi = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_roi, [contour], -1, color=255, thickness=-1)
            _, stddev = cv2.meanStdDev(frame, mask=mask_roi)
            std_mean = float(np.mean(stddev))
            std_thresh = 30.0
            if color_name == "c1":
                std_thresh = 28.0
            if color_name == "c3":
                std_thresh = 18.0
            if std_mean > std_thresh:
                continue

            # attempt to split large blobs (overlapping circles) using distance transform
            split_generated = False
            if area > 200 and fill_ratio > 0.45:
                x0, y0, ww, hh = x, y, w, h
                submask = mask[y0 : y0 + hh, x0 : x0 + ww]
                if submask.size > 0:
                    _, th = cv2.threshold(submask, 0, 255, cv2.THRESH_BINARY)
                    dist = cv2.distanceTransform(th, cv2.DIST_L2, 5)
                    if dist is not None:
                        _, maxVal, _, _ = cv2.minMaxLoc(dist)
                        if maxVal > 3.0:
                            dilated = cv2.dilate(dist, np.ones((3, 3), np.uint8))
                            local_max = (dist == dilated) & (dist > (0.5 * maxVal))
                            local_max = local_max.astype(np.uint8)
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(local_max)
                            for i in range(1, num_labels):
                                cx_local, cy_local = centroids[i]
                                r = dist[int(round(cy_local)), int(round(cx_local))]
                                if r > 1.5:
                                    gx = x0 + cx_local
                                    gy = y0 + cy_local
                                    pd = max(4.0, 2.0 * float(r))
                                    cand = (round(gx, 2), round(gy, 2), round(pd, 2))
                                    if cand not in seen_boxes:
                                        seen_boxes.add(cand)
                                        candidates.append(Candidate(cx=gx, cy=gy, diameter=pd))
                            # if any peaks were added within this bbox, skip adding the bbox center
                            if any(c.cx >= x0 and c.cx <= x0 + ww and c.cy >= y0 and c.cy <= y0 + hh for c in candidates):
                                split_generated = True

            if split_generated:
                continue

            candidate = (round(cx, 2), round(cy, 2), round(diameter, 2))
            if candidate in seen_boxes:
                continue
            seen_boxes.add(candidate)
            candidates.append(Candidate(cx=cx, cy=cy, diameter=diameter))

    candidates.sort(key=lambda d: (d.cy, d.cx, d.diameter))
    return candidates


def estimate_fixed_box_side(samples: list[Sample]) -> int:
    diameters = [candidate.diameter for sample in samples for candidate in sample.candidates]
    if not diameters:
        return 14

    median_diameter = float(np.median(diameters))
    side = int(round(median_diameter + 4))
    return max(10, min(24, side))


def fixed_square_detection(candidate: Candidate, side: int, width: int, height: int) -> Detection:
    half = side // 2
    x1 = int(round(candidate.cx)) - half
    y1 = int(round(candidate.cy)) - half
    x1 = max(0, min(x1, width - side))
    y1 = max(0, min(y1, height - side))
    x2 = x1 + side
    y2 = y1 + side
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2)


def finalize_detections(candidates: list[Candidate], side: int, width: int, height: int) -> list[Detection]:
    detections = [fixed_square_detection(candidate, side=side, width=width, height=height) for candidate in candidates]

    def iou(a: Detection, b: Detection) -> float:
        xi1 = max(a.x1, b.x1)
        yi1 = max(a.y1, b.y1)
        xi2 = min(a.x2, b.x2)
        yi2 = min(a.y2, b.y2)
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter = inter_w * inter_h
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: (item.y1, item.x1)):
        # allow larger overlap (for closely spaced/occluded points) before suppressing
        if all(iou(detection, kept_item) <= 0.60 for kept_item in kept):
            kept.append(detection)

    kept.sort(key=lambda d: (d.y1, d.x1, d.x2, d.y2))
    return kept


def extract_samples(
    input_dir: Path,
    sample_fps: float,
    min_area: float,
    max_area: float,
    min_circularity: float,
) -> list[Sample]:
    samples: list[Sample] = []
    videos = iter_videos(input_dir)

    for video_path in videos:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(source_fps / sample_fps)))

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % step == 0:
                candidates = detect_points(
                    frame, min_area=min_area, max_area=max_area, min_circularity=min_circularity
                )
                image_name = f"{video_path.stem}_{frame_index:06d}.jpg"
                samples.append(
                    Sample(
                        source_video=video_path.name,
                        frame_index=frame_index,
                        image_path=Path(image_name),
                        candidates=candidates,
                    )
                )
            frame_index += 1

        capture.release()
        print(f"Processed {video_path.name}: {frame_index} frames, sampled every {step} frames, target={sample_fps} fps")

    return samples


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_sample(sample: Sample, split: str, output_dir: Path, frame: np.ndarray) -> dict:
    image_output = output_dir / "images" / split / sample.image_path.name
    label_output = output_dir / "labels" / split / f"{sample.image_path.stem}.txt"
    cv2.imwrite(str(image_output), frame)
    with label_output.open("w", encoding="utf-8") as handle:
        for detection in sample.detections or []:
            handle.write(detection.to_yolo(frame.shape[1], frame.shape[0]) + "\n")
    return {
        "split": split,
        "image": str(image_output.relative_to(output_dir)),
        "label": str(label_output.relative_to(output_dir)),
        "frame_index": sample.frame_index,
        "source_video": sample.source_video,
        "objects": len(sample.detections),
    }


def export_dataset(
    input_dir: Path,
    output_dir: Path,
    sample_fps: float,
    val_ratio: float,
    seed: int,
    min_area: float,
    max_area: float,
    min_circularity: float,
    target_size: int = 0,
    auto_crop: bool = True,
    augment: bool = False,
    hflip: bool = False,
    brightness: float = 0.0,
    contrast: float = 0.0,
    use_hsv: bool = False,
    hsv_tol: tuple = (12, 60, 60),
    debug_limit: int = 5,
) -> dict:
    samples = []
    videos = iter_videos(input_dir)

    for video_path in videos:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(source_fps / sample_fps)))
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % step == 0:
                detections = detect_points(
                    frame,
                    min_area=min_area,
                    max_area=max_area,
                    min_circularity=min_circularity,
                    use_hsv=use_hsv,
                    hsv_tol=hsv_tol,
                )
                samples.append(
                    {
                        "sample": Sample(
                            source_video=video_path.name,
                            frame_index=frame_index,
                            image_path=Path(f"{video_path.stem}_{frame_index:06d}.jpg"),
                            candidates=detections,
                        ),
                        "frame": frame,
                    }
                )
            frame_index += 1

        capture.release()

    rng = random.Random(seed)
    rng.shuffle(samples)
    val_count = max(1, int(round(len(samples) * val_ratio))) if len(samples) > 1 else 0
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    prepare_output(output_dir)

    fixed_box_side = estimate_fixed_box_side([item["sample"] for item in samples])

    for item in samples:
        sample = item["sample"]
        frame = item["frame"]
        sample.detections = finalize_detections(
            sample.candidates,
            side=fixed_box_side,
            width=frame.shape[1],
            height=frame.shape[0],
        )

    # Helper augmentation utilities
    def crop_black_borders(frame: np.ndarray, thresh: int = 8):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cols = np.where(np.max(gray, axis=0) > thresh)[0]
        rows = np.where(np.max(gray, axis=1) > thresh)[0]
        if cols.size == 0 or rows.size == 0:
            return frame, (0, 0, w, h)
        x1, x2 = int(cols[0]), int(cols[-1])
        y1, y2 = int(rows[0]), int(rows[-1])
        return frame[y1 : y2 + 1, x1 : x2 + 1], (x1, y1, x2 + 1, y2 + 1)

    def resize_and_pad(frame: np.ndarray, target: int):
        h, w = frame.shape[:2]
        if target <= 0 or (w == target and h == target):
            return frame, 1.0, 0, 0
        scale = min(target / float(w), target / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        top = pad_y
        bottom = target - new_h - top
        left = pad_x
        right = target - new_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded, scale, left, top

    def transform_detections_for_transform(
        detections: list[Detection], crop_box: tuple, scale: float, pad_x: int, pad_y: int, target_w: int, target_h: int, do_hflip: bool = False
    ) -> list[Detection]:
        xoff, yoff, _, _ = crop_box
        out: list[Detection] = []
        for d in detections:
            x1 = int(round((d.x1 - xoff) * scale)) + pad_x
            y1 = int(round((d.y1 - yoff) * scale)) + pad_y
            x2 = int(round((d.x2 - xoff) * scale)) + pad_x
            y2 = int(round((d.y2 - yoff) * scale)) + pad_y
            if do_hflip:
                nx1 = target_w - x2
                nx2 = target_w - x1
                x1, x2 = nx1, nx2
            x1 = max(0, min(int(x1), target_w - 1))
            x2 = max(1, min(int(x2), target_w))
            y1 = max(0, min(int(y1), target_h - 1))
            y2 = max(1, min(int(y2), target_h))
            out.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2))
        return out

    manifest = []
    split_counts = {"train": 0, "val": 0}
    object_counts = {"train": 0, "val": 0}

    for split_name, split_samples in (("train", train_samples), ("val", val_samples)):
        for item in split_samples:
            sample = item["sample"]
            frame = item["frame"]
            aug_frame = frame
            crop_box = (0, 0, frame.shape[1], frame.shape[0])
            scale = 1.0
            pad_x = 0
            pad_y = 0
            do_hflip = False

            if auto_crop:
                try:
                    aug_frame, crop_box = crop_black_borders(aug_frame)
                except Exception:
                    crop_box = (0, 0, frame.shape[1], frame.shape[0])

            if target_size and target_size > 0:
                aug_frame, scale, pad_x, pad_y = resize_and_pad(aug_frame, target_size)

            if augment and (brightness != 0.0 or contrast != 0.0):
                b = random.uniform(-brightness, brightness) if brightness != 0.0 else 0.0
                c = 1.0 + random.uniform(-contrast, contrast) if contrast != 0.0 else 1.0
                aug_frame = cv2.convertScaleAbs(aug_frame, alpha=c, beta=int(b))

            if augment and hflip:
                if random.random() < 0.5:
                    aug_frame = cv2.flip(aug_frame, 1)
                    do_hflip = True

            final_w, final_h = aug_frame.shape[1], aug_frame.shape[0]
            transformed = transform_detections_for_transform(
                sample.detections or [], crop_box=crop_box, scale=scale, pad_x=pad_x, pad_y=pad_y, target_w=final_w, target_h=final_h, do_hflip=do_hflip
            )

            sample_to_write = Sample(
                source_video=sample.source_video,
                frame_index=sample.frame_index,
                image_path=sample.image_path,
                candidates=sample.candidates,
                detections=transformed,
            )

            manifest.append(write_sample(sample_to_write, split_name, output_dir, aug_frame))
            split_counts[split_name] += 1
            object_counts[split_name] += len(sample_to_write.detections)

    data_yaml = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: point",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sample_fps": sample_fps,
        "val_ratio": val_ratio,
        "seed": seed,
        "fixed_box_side": fixed_box_side,
        "min_area": min_area,
        "max_area": max_area,
        "min_circularity": min_circularity,
        "total_samples": len(samples),
        "split_counts": split_counts,
        "object_counts": object_counts,
        "manifest": manifest,
    }
    summary["debug_overlays"] = save_debug_overlays(output_dir, manifest, limit=debug_limit)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def save_debug_overlays(output_dir: Path, manifest: list[dict], limit: int = 5) -> list[str]:
    debug_dir = output_dir / "debug_overlays"
    debug_dir.mkdir(parents=True, exist_ok=True)

    selected = sorted(
        [item for item in manifest if item["objects"] > 0],
        key=lambda item: (-item["objects"], item["source_video"], item["frame_index"]),
    )[:limit]

    debug_paths: list[str] = []
    for index, item in enumerate(selected, start=1):
        image_path = output_dir / item["image"]
        label_path = output_dir / item["label"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        height, width = frame.shape[:2]
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                _, x_center, y_center, box_width, box_height = map(float, line.split())
                x1 = int((x_center - box_width / 2) * width)
                y1 = int((y_center - box_height / 2) * height)
                x2 = int((x_center + box_width / 2) * width)
                y2 = int((y_center + box_height / 2) * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        debug_path = debug_dir / f"debug_overlay_{index:02d}.jpg"
        cv2.imwrite(str(debug_path), frame)
        debug_paths.append(str(debug_path.relative_to(output_dir)))

    return debug_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a YOLO dataset with a single point class from video files.")
    parser.add_argument("--input", type=Path, default=Path("."), help="Directory containing source video files.")
    parser.add_argument("--output", type=Path, default=Path("yolo_point_dataset"), help="Output dataset directory.")
    parser.add_argument("--sample-fps", type=float, default=8.0, help="Frame sampling rate in frames per second.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--min-area", type=float, default=8.0, help="Minimum blob area in pixels.")
    parser.add_argument("--max-area", type=float, default=1500.0, help="Maximum blob area in pixels.")
    parser.add_argument("--min-circularity", type=float, default=0.50, help="Minimum contour circularity.")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation (brightness/contrast/hflip).")
    parser.add_argument("--target-size", type=int, default=640, help="Normalize output images to TARGETxTARGET (pad while keeping aspect). Set 0 to keep original size.")
    parser.add_argument("--no-auto-crop", dest="auto_crop", action="store_false", help="Disable automatic black-border cropping.")
    parser.add_argument("--hflip", action="store_true", help="Allow random horizontal flips when augmenting.")
    parser.add_argument("--brightness", type=float, default=0.0, help="Max brightness shift (±) when augmenting.")
    parser.add_argument("--contrast", type=float, default=0.0, help="Max contrast relative change fraction when augmenting (e.g., 0.15).")
    parser.add_argument("--use-hsv", action="store_true", help="Enable HSV-based color matching in addition to BGR.")
    parser.add_argument("--hsv-tol", type=int, nargs=3, default=(12, 60, 60), help="HSV tolerances H S V (e.g. 12 60 60)")
    parser.add_argument("--debug-limit", type=int, default=15, help="Number of debug overlay images to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    summary = export_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_fps=args.sample_fps,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        target_size=args.target_size,
        auto_crop=args.auto_crop,
        augment=args.augment,
        hflip=args.hflip,
        brightness=args.brightness,
        contrast=args.contrast,
        use_hsv=args.use_hsv,
        hsv_tol=tuple(args.hsv_tol),
        debug_limit=args.debug_limit,
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "manifest"}, ensure_ascii=False, indent=2))
    print(f"Dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
