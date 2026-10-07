"""Speed models. Each maps a 2 s IMU window to the robot's mean forward speed (m/s)."""

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn

from .data import FS
from .features import augment, stat_features


# ---------------------------------------------------------------- baselines

def integrate_forward(df, start, end):
    """Physics baseline: double-integrate forward acceleration (phone y axis).

    Velocity is clamped at zero because the robot never reverses.
    """
    seg = df[(df.t >= start) & (df.t <= end)]
    dt = 1.0 / FS
    v = d = 0.0
    for a in seg.ay.values:
        v = max(v + a * dt, 0.0)
        d += v * dt
    return d


class ConstantSpeed:
    """Mean training-run speed x seconds the accelerometer shows motion."""

    def fit(self, run_speeds):
        self.speed = float(np.mean(run_speeds))
        return self

    def distance(self, moving_seconds):
        return self.speed * moving_seconds


# ---------------------------------------------------------------- classical ML

def fit_random_forest(X, y, groups, seed=0):
    grid = GridSearchCV(
        RandomForestRegressor(random_state=seed, n_jobs=-1),
        {"n_estimators": [200], "max_depth": [None, 10], "min_samples_leaf": [1, 5]},
        cv=GroupKFold(3), scoring="neg_mean_absolute_error")
    grid.fit(stat_features(X), y, groups=groups)
    return grid.best_estimator_, grid.best_params_


def fit_svr(X, y, groups):
    grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVR(kernel="rbf")),
        {"svr__C": [1, 10], "svr__epsilon": [0.02, 0.05], "svr__gamma": ["scale", 0.01]},
        cv=GroupKFold(3), scoring="neg_mean_absolute_error")
    grid.fit(stat_features(X), y, groups=groups)
    return grid.best_estimator_, grid.best_params_


def predict_sklearn(model, X):
    return np.clip(model.predict(stat_features(X)), 0, None)


# ---------------------------------------------------------------- CNN-GRU

class CNNGRU(nn.Module):
    """Conv1D blocks extract local motion patterns; a GRU models their order in time."""

    def __init__(self, channels=6, hidden=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):                 # x: (batch, time, channels)
        h = self.conv(x.transpose(1, 2))   # (batch, 64, time/4)
        _, last = self.gru(h.transpose(1, 2))
        return self.head(last[-1]).squeeze(1)


def fit_cnn_gru(X_tr, y_tr, X_val, y_val, seed=0, epochs=150, patience=15,
                batch_size=64, lr=1e-3):
    """Train with on-the-fly augmentation and early stopping on validation MAE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CNNGRU()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)

    best, best_state, best_epoch, wait = np.inf, None, 0, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(X_tr))
        Xa = torch.from_numpy(augment(X_tr, rng))
        ya = torch.from_numpy(y_tr)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xa[idx]), ya[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mae = (model(Xv).clamp(min=0) - yv).abs().mean().item()
        sched.step(val_mae)
        if val_mae < best - 1e-4:
            best, best_epoch, wait = val_mae, epoch + 1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def predict_cnn_gru(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X)).clamp(min=0).numpy()
