import numpy as np
from scipy.signal import savgol_filter


class KalmanFilter1D:
    # 4-state: x, y, vx, vy
    def __init__(self, process_var=1e-2, meas_var=1e-1):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0
        self.F = np.eye(4)
        self.F[0, 2] = 1.0
        self.F[1, 3] = 1.0
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var

    def predict(self):
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        z = np.array(z).reshape((2, 1))
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(4) - K.dot(self.H)).dot(self.P)
        return self.x[:2].flatten()


class SmootherManager:
    def __init__(self, process_noise=1e-2, meas_noise=1e-1):
        self.filters = {}
        self.process_noise = process_noise
        self.meas_noise = meas_noise

    def predict(self, track_id):
        k = self.filters.get(track_id)
        if k is None:
            return None
        return k.predict()

    def update(self, track_id, measurement):
        if track_id not in self.filters:
            self.filters[track_id] = KalmanFilter1D(process_var=self.process_noise, meas_var=self.meas_noise)
            # initialize state
            self.filters[track_id].x[0, 0] = measurement[0]
            self.filters[track_id].x[1, 0] = measurement[1]
        return self.filters[track_id].update(measurement)

    def finalize_trajectory(self, traj: list, window=7, polyorder=2):
        if len(traj) < 3:
            return traj
        arr = np.array(traj)
        n = len(arr)
        w = min(window, n - (1 if n % 2 == 0 else 0))
        if w < 3:
            return traj
        if w % 2 == 0:
            w -= 1
        if w < 3:
            return traj
        xs = savgol_filter(arr[:, 0], window_length=w, polyorder=polyorder)
        ys = savgol_filter(arr[:, 1], window_length=w, polyorder=polyorder)
        return list(zip(xs.tolist(), ys.tolist()))
