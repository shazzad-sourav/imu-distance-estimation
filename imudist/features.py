"""Sliding windows, statistical features and augmentation."""

import numpy as np

from .data import CHANNELS, FS

WINDOW_S = 2.0
WINDOW = int(WINDOW_S * FS)     # 50 samples
HOP = 5                         # 0.2 s between window centres


def make_windows(entry, hop=HOP, window=WINDOW):
    """Overlapping windows of the smoothed signal.

    Returns X (n, window, 6), centre times (n,), and the mean ground-truth
    speed in each window (None for recordings without per-run labels).
    """
    df = entry["df"]
    values = df[CHANNELS].values
    starts = np.arange(0, len(df) - window + 1, hop)
    X = np.stack([values[s:s + window] for s in starts]).astype(np.float32)
    centres = df.t.values[starts + window // 2]
    y = None
    if entry["speed"] is not None:
        y = np.array([entry["speed"][s:s + window].mean() for s in starts], dtype=np.float32)
    return X, centres, y


def stat_features(X):
    """Per-channel statistics plus magnitudes, for Random Forest and SVR."""
    feats = [X.mean(1), X.std(1), X.min(1), X.max(1), X.max(1) - X.min(1),
             np.abs(np.diff(X, axis=1)).mean(1)]            # mean absolute change
    acc_mag = np.linalg.norm(X[:, :, :3], axis=2)
    gyro_mag = np.linalg.norm(X[:, :, 3:], axis=2)
    feats += [acc_mag.mean(1, keepdims=True), acc_mag.std(1, keepdims=True),
              gyro_mag.mean(1, keepdims=True), gyro_mag.std(1, keepdims=True)]
    return np.concatenate(feats, axis=1)


def augment(X, rng, jitter_std=0.05, scale_std=0.1):
    """Jittering (additive noise) and per-channel amplitude scaling.

    Applied to training windows only, in normalised units.
    """
    scale = rng.normal(1.0, scale_std, size=(X.shape[0], 1, X.shape[2]))
    noise = rng.normal(0.0, jitter_std, size=X.shape)
    return (X * scale + noise).astype(np.float32)


class Normaliser:
    """Per-channel standardisation fitted on training windows only."""

    def fit(self, X):
        self.mean = X.reshape(-1, X.shape[2]).mean(0)
        self.std = X.reshape(-1, X.shape[2]).std(0) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)
