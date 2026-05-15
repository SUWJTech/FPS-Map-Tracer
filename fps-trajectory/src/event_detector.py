from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd


@dataclass
class Event:
    track_id: int
    frame_id: int
    timestamp: float
    event_type: str
    value: float


class EventDetector:
    def __init__(self, direction_deg: float = 60.0, stop_speed: float = 2.0, rush_speed: float = 25.0, stop_seconds: float = 2.0) -> None:
        self.direction_deg = float(direction_deg)
        self.stop_speed = float(stop_speed)
        self.rush_speed = float(rush_speed)
        self.stop_seconds = float(stop_seconds)

    def detect_from_samples(self, tracks_payload: Dict[str, List[dict]], fps: float) -> List[Event]:
        events: List[Event] = []
        stop_frames = max(1, int(round(self.stop_seconds * max(fps, 1.0))))

        for track_id_str, samples in tracks_payload.items():
            if len(samples) < 3:
                continue
            track_id = int(track_id_str)
            speeds: List[float] = []
            dirs: List[np.ndarray] = []

            for i in range(1, len(samples)):
                p0 = np.array([samples[i - 1]["x"], samples[i - 1]["y"]], dtype=float)
                p1 = np.array([samples[i]["x"], samples[i]["y"]], dtype=float)
                d = p1 - p0
                speed = float(np.linalg.norm(d))
                speeds.append(speed)
                dirs.append(d)

                if speed > self.rush_speed:
                    events.append(Event(track_id, int(samples[i]["frame_id"]), float(samples[i]["timestamp"]), "high_speed_rush", speed))

            # direction change
            for i in range(1, len(dirs)):
                a = dirs[i - 1]
                b = dirs[i]
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na < 1e-6 or nb < 1e-6:
                    continue
                cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cosang)))
                if angle > self.direction_deg:
                    idx = min(i + 1, len(samples) - 1)
                    events.append(Event(track_id, int(samples[idx]["frame_id"]), float(samples[idx]["timestamp"]), "direction_change", angle))

            # stop_loot + possible_fire (micro jitter)
            run = 0
            jitter_run = 0
            for i, s in enumerate(speeds):
                if s < self.stop_speed:
                    run += 1
                    if run == stop_frames:
                        idx = min(i + 1, len(samples) - 1)
                        events.append(Event(track_id, int(samples[idx]["frame_id"]), float(samples[idx]["timestamp"]), "stop_loot", float(run)))
                else:
                    run = 0

                if 0.6 < s < 3.0:
                    jitter_run += 1
                    if jitter_run >= max(3, int(0.5 * max(fps, 1.0))):
                        idx = min(i + 1, len(samples) - 1)
                        events.append(Event(track_id, int(samples[idx]["frame_id"]), float(samples[idx]["timestamp"]), "possible_fire", float(jitter_run)))
                        jitter_run = 0
                else:
                    jitter_run = 0

            tail = speeds[-max(1, int(2.0 * max(fps, 1.0))):]
            if len(tail) > 0 and float(np.mean(tail)) < 0.5:
                last = samples[-1]
                events.append(Event(track_id, int(last["frame_id"]), float(last["timestamp"]), "possible_death", float(np.mean(tail))))

        return events

    def export_events_from_tracks(self, tracks_payload: Dict[str, List[dict]], output_csv: str, fps: float, output_json: str | None = None) -> str:
        events = self.detect_from_samples(tracks_payload, fps=fps)
        rows = [asdict(e) for e in events]
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        if output_json is not None:
            jout = Path(output_json)
            jout.parent.mkdir(parents=True, exist_ok=True)
            jout.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)
