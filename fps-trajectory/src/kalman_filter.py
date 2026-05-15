from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class KalmanState:
    x: float
    y: float
    vx: float
    vy: float


class KalmanFilter2D:
    """Constant-velocity Kalman filter for state [x, y, vx, vy]."""

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1, dt: float = 1.0) -> None:
        self.dt = float(dt)

        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        self.Q = np.eye(4, dtype=float) * float(process_noise)
        self.R = np.eye(2, dtype=float) * float(measurement_noise)

        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 10.0

    def initialize(self, center: Sequence[float], velocity: Sequence[float] = (0.0, 0.0)) -> None:
        self.x[:, 0] = [float(center[0]), float(center[1]), float(velocity[0]), float(velocity[1])]

    def predict(self) -> KalmanState:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement: Sequence[float]) -> KalmanState:
        z = np.array([[float(measurement[0])], [float(measurement[1])]], dtype=float)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)

        self.x = self.x + k @ y
        i = np.eye(4, dtype=float)
        self.P = (i - k @ self.H) @ self.P
        return self.state

    @property
    def state(self) -> KalmanState:
        return KalmanState(
            x=float(self.x[0, 0]),
            y=float(self.x[1, 0]),
            vx=float(self.x[2, 0]),
            vy=float(self.x[3, 0]),
        )

    @property
    def center(self) -> Tuple[float, float]:
        s = self.state
        return s.x, s.y

    @property
    def velocity(self) -> Tuple[float, float]:
        s = self.state
        return s.vx, s.vy

    def predicted_center(self) -> Tuple[float, float]:
        return self.center

    def predicted_velocity(self) -> Tuple[float, float]:
        return self.velocity
