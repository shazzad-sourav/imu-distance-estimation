"""Load the phone IMU recordings, resample them, and find the individual runs.

Each training recording contains 10 runs of exactly 5 m. Between runs the
robot turns around in place, which shows up as a ~180 degree change in
heading (integrated gyroscope z). Those turns split a recording into runs.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FS = 25.0                      # common sampling rate after resampling (Hz)
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
RUN_LENGTH_M = 5.0

# name: (number of 5 m runs, total distance in metres)
RECORDINGS = {
    "straight_slow": (10, 50.0),
    "straight_fast": (10, 50.0),
    "curve_slow": (10, 50.0),
    "curve_fast": (10, 50.0),
    "mixed_27m": (None, 27.0),   # one continuous straight + zigzag path
}
TRAIN_RECORDINGS = ["straight_slow", "straight_fast", "curve_slow", "curve_fast"]

# Segmentation thresholds
TURN_RATE = 0.8          # rad/s, |gyro z| above this may be a turn
TURN_ANGLE = 100.0       # degrees, heading change that counts as a turnaround
MOTION_STD = 0.15        # m/s^2, rolling std of |acc| above this = moving
MIN_RUN_MOVING_S = 3.0   # shorter active stretches are handling, not runs


@dataclass
class Run:
    recording: str
    index: int
    start: float          # seconds
    end: float
    moving_s: float       # seconds of detected motion inside the run


def load_recording(name, fs=FS):
    """Accelerometer + gyroscope on a common uniform time grid (seconds from start)."""
    folder = DATA_DIR / name
    acc = pd.read_csv(folder / "Accelerometer.csv")
    gyr = pd.read_csv(folder / "Gyroscope.csv")
    start = max(acc.seconds_elapsed.iloc[0], gyr.seconds_elapsed.iloc[0])
    end = min(acc.seconds_elapsed.iloc[-1], gyr.seconds_elapsed.iloc[-1])
    t = np.arange(start, end, 1.0 / fs)
    out = {"t": t - start}
    for src, prefix in ((acc, "a"), (gyr, "g")):
        for axis in "xyz":
            out[prefix + axis] = np.interp(t, src.seconds_elapsed.values, src[axis].values)
    return pd.DataFrame(out)


def smooth(df, window=5):
    """Centered moving average on the sensor channels (noise reduction)."""
    out = df.copy()
    out[CHANNELS] = df[CHANNELS].rolling(window, center=True, min_periods=1).mean()
    return out


def _intervals(mask, t, merge_gap):
    """Contiguous True stretches of mask as [start_idx, end_idx], merging small gaps."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    spans = edges.reshape(-1, 2)
    merged = []
    for s, e in spans:
        e -= 1
        if merged and t[s] - t[merged[-1][1]] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def heading(df):
    """Integrated gyroscope z (radians)."""
    dt = np.diff(df.t.values, prepend=df.t.values[0])
    return np.cumsum(df.gz.values * dt)


def moving_mask(df):
    """True where the accelerometer shows the robot (or phone) is in motion."""
    mag = np.sqrt(df.ax ** 2 + df.ay ** 2 + df.az ** 2)
    return (mag.rolling(int(0.4 * FS), center=True, min_periods=1).std() > MOTION_STD).values


def find_turns(df):
    """Turnarounds as (start_s, end_s, degrees)."""
    t = df.t.values
    gz = pd.Series(df.gz.values).rolling(int(0.3 * FS), center=True, min_periods=1).mean().values
    head = np.degrees(heading(df))
    turns = []
    for s, e in _intervals(np.abs(gz) > TURN_RATE, t, merge_gap=0.3):
        change = head[e] - head[s]
        if abs(change) > TURN_ANGLE:
            turns.append((t[s], t[e], change))
    return turns


def find_runs(df, name):
    """Split a training recording into its 5 m runs."""
    t = df.t.values
    moving = moving_mask(df)
    turns = find_turns(df)
    in_turn = np.zeros(len(t), bool)
    for s, e, _ in turns:
        in_turn |= (t >= s) & (t <= e)
    forward = moving & ~in_turn

    bounds = [(-np.inf, t[0])] + [(s, e) for s, e, _ in turns] + [(t[-1], np.inf)]
    runs = []
    for (_, prev_end), (next_start, _) in zip(bounds[:-1], bounds[1:]):
        inside = (t > prev_end) & (t < next_start) & forward
        if inside.sum() / FS < MIN_RUN_MOVING_S:
            continue
        idx = np.flatnonzero(inside)
        runs.append(Run(name, len(runs), t[idx[0]], t[idx[-1]], inside.sum() / FS))
    return runs, turns


def speed_labels(df, runs):
    """Ground-truth forward speed per sample.

    Each run covers exactly 5 m. Assuming constant speed while the robot is
    moving, speed = 5 m / (moving time of the run); zero during turns, pauses
    and outside runs.
    """
    t = df.t.values
    moving = moving_mask(df)
    speed = np.zeros(len(t))
    for run in runs:
        inside = (t >= run.start) & (t <= run.end) & moving
        speed[inside] = RUN_LENGTH_M / run.moving_s
    return speed


def load_all():
    """All recordings, smoothed, with runs and speed labels for the training set."""
    data = {}
    for name in RECORDINGS:
        raw = load_recording(name)
        entry = {"df": smooth(raw), "raw": raw, "runs": [], "turns": [], "speed": None}
        if name in TRAIN_RECORDINGS:
            runs, turns = find_runs(raw, name)
            expected = RECORDINGS[name][0]
            if len(runs) != expected:
                raise RuntimeError(f"{name}: found {len(runs)} runs, expected {expected}")
            entry.update(runs=runs, turns=turns, speed=speed_labels(raw, runs))
        data[name] = entry
    return data
