"""Estimate distance travelled by a Diablo robot from phone IMU data.

Models predict forward speed for each 2 s window (one every 0.2 s); distance
is the sum of speed x 0.2 s. Evaluation:

1. 5-fold cross-validation over the 40 runs (each fold holds out 2 runs per
   recording, never seen in training). Every held-out run is exactly 5 m.
2. Final test: train on all 40 runs, then estimate the full length of the
   separate 27 m straight + zigzag recording.

Usage:  python run_experiments.py [--epochs 150] [--seed 0]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from imudist.data import FS, RECORDINGS, TRAIN_RECORDINGS, heading, load_all, moving_mask
from imudist.features import HOP, WINDOW_S, Normaliser, augment, make_windows
from imudist import models

RESULTS = Path(__file__).resolve().parent / "results"
N_FOLDS = 5
STEP_S = HOP / FS
METHODS = ["Integration", "Constant speed", "Random Forest", "SVR", "CNN-GRU"]


def run_intervals(entry):
    """Split the whole recording between runs at the midpoints of the turns.

    A window's speed belongs to the run whose interval contains its centre, so
    every window is counted exactly once and run totals add up to the recording.
    """
    runs, t = entry["runs"], entry["raw"].t.values
    cuts = [t[0]] + [(a.end + b.start) / 2 for a, b in zip(runs[:-1], runs[1:])] + [t[-1] + 1]
    return [(cuts[i], cuts[i + 1]) for i in range(len(runs))]


def window_table(data):
    """Windows from all training recordings with run/fold assignment."""
    rows, Xs, ys = [], [], []
    for name in TRAIN_RECORDINGS:
        X, centres, y = make_windows(data[name])
        intervals = run_intervals(data[name])
        for c in centres:
            run = next(i for i, (lo, hi) in enumerate(intervals) if lo <= c < hi)
            rows.append({"recording": name, "centre": c, "run": run})
        Xs.append(X)
        ys.append(y)
    meta = pd.DataFrame(rows)
    meta["fold"] = meta.run % N_FOLDS
    meta["group"] = meta.recording + "_" + meta.run.astype(str)
    return meta, np.concatenate(Xs), np.concatenate(ys)


def train_run_speeds(data, exclude_fold=None):
    """Average speed of each training run (5 m / moving time)."""
    return np.array([5.0 / r.moving_s for name in TRAIN_RECORDINGS for r in data[name]["runs"]
                     if exclude_fold is None or r.index % N_FOLDS != exclude_fold])


def purge(meta, train, test):
    """Drop training windows that overlap a test window in time."""
    keep = train.copy()
    for name in TRAIN_RECORDINGS:
        tc = meta.centre[test & (meta.recording == name)].values
        idx = np.flatnonzero(train & (meta.recording == name).values)
        if len(tc) and len(idx):
            near = np.abs(meta.centre.values[idx][:, None] - tc[None, :]).min(1) < WINDOW_S
            keep[idx[near]] = False
    return keep


def fit_all(X_tr, y_tr, groups, X_val, y_val, run_speeds, args):
    """Fit every learned model on (already normalised) training windows."""
    rng = np.random.default_rng(args.seed)
    X_aug = np.concatenate([X_tr, augment(X_tr, rng)])
    y_aug = np.concatenate([y_tr, y_tr])
    g_aug = np.concatenate([groups, groups])
    rf, rf_params = models.fit_random_forest(X_aug, y_aug, g_aug, seed=args.seed)
    svr, svr_params = models.fit_svr(X_aug, y_aug, g_aug)
    cnn, epoch = models.fit_cnn_gru(X_tr, y_tr, X_val, y_val, seed=args.seed, epochs=args.epochs)
    const = models.ConstantSpeed().fit(run_speeds)
    info = {"rf": rf_params, "svr": svr_params, "cnn_best_epoch": epoch}
    return {"Random Forest": lambda X: models.predict_sklearn(rf, X),
            "SVR": lambda X: models.predict_sklearn(svr, X),
            "CNN-GRU": lambda X: models.predict_cnn_gru(cnn, X)}, const, info


def cross_validate(data, args):
    meta, X, y = window_table(data)
    window_preds = {m: np.full(len(y), np.nan) for m in ["Random Forest", "SVR", "CNN-GRU"]}
    run_rows, fold_info = [], []

    for k in range(N_FOLDS):
        test = (meta.fold == k).values
        train = purge(meta, ~test, test)
        val = train & (meta.fold == (k + 1) % N_FOLDS).values
        fit = train & ~val
        norm = Normaliser().fit(X[train])
        Xn = norm.transform(X)
        speeds = train_run_speeds(data, exclude_fold=k)
        predictors, const, info = fit_all(Xn[fit], y[fit], meta.group[fit].values,
                                          Xn[val], y[val], speeds, args)
        fold_info.append(info)
        print(f"fold {k + 1}/{N_FOLDS}: {info}")

        for name, predict in predictors.items():
            window_preds[name][test] = predict(Xn[test])

        for rec in TRAIN_RECORDINGS:
            df_raw = data[rec]["raw"]
            intervals = run_intervals(data[rec])
            for run in data[rec]["runs"]:
                if run.index % N_FOLDS != k:
                    continue
                row = {"recording": rec, "run": run.index, "true_m": 5.0,
                       "Integration": models.integrate_forward(df_raw, run.start, run.end),
                       "Constant speed": const.distance(run.moving_s)}
                sel = (test & (meta.recording == rec).values
                       & (meta.run == run.index).values)
                for name in predictors:
                    row[name] = window_preds[name][sel].sum() * STEP_S
                run_rows.append(row)

    runs = pd.DataFrame(run_rows)
    window_metrics = {}
    for name, pred in window_preds.items():
        window_metrics[name] = {
            "MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": r2_score(y, pred)}
    return runs, window_metrics, fold_info


def final_test(data, args):
    """Train on all 40 runs; estimate the 27 m recording."""
    meta, X, y = window_table(data)
    val = (meta.fold == N_FOLDS - 1).values
    norm = Normaliser().fit(X)
    Xn = norm.transform(X)
    predictors, const, info = fit_all(Xn[~val], y[~val], meta.group[~val].values,
                                      Xn[val], y[val], train_run_speeds(data), args)

    entry = data["mixed_27m"]
    X27, centres, _ = make_windows(entry)
    X27 = norm.transform(X27)
    raw = entry["raw"]
    moving_s = moving_mask(raw).sum() / FS
    curves = {name: (centres, np.cumsum(p(X27)) * STEP_S) for name, p in predictors.items()}
    totals = {"Integration": models.integrate_forward(raw, raw.t.iloc[0], raw.t.iloc[-1]),
              "Constant speed": const.distance(moving_s)}
    totals.update({name: float(c[1][-1]) for name, c in curves.items()})
    return totals, curves, info


def summarise_runs(runs):
    out = {}
    for m in METHODS:
        err = runs[m] - runs.true_m
        out[m] = {"MAE_m": float(err.abs().mean()), "mean_error_m": float(err.mean()),
                  "MAPE_pct": float((err.abs() / runs.true_m).mean() * 100),
                  "worst_m": float(err.abs().max())}
        out[m]["by_recording"] = {rec: float((runs[m] - runs.true_m)[runs.recording == rec].abs().mean())
                                  for rec in TRAIN_RECORDINGS}
    return out


def plot_segmentation(data, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    for ax, name in zip(axes, ["straight_slow", "curve_fast"]):
        df = data[name]["raw"]
        ax.plot(df.t, np.degrees(heading(df)), color="tab:blue", lw=1, label="heading (deg)")
        ax2 = ax.twinx()
        ax2.plot(df.t, data[name]["speed"], color="tab:orange", lw=1, label="label speed (m/s)")
        for run in data[name]["runs"]:
            ax.axvspan(run.start, run.end, color="tab:green", alpha=0.12)
        ax.set_title(f"{name}: 10 detected runs (shaded) between 180-degree turnarounds")
        ax.set_ylabel("heading (deg)")
        ax2.set_ylabel("speed (m/s)")
        ax2.set_ylim(0, 1.5)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_results(runs, curves, totals, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.5))
    errs = [runs[m] - runs.true_m for m in METHODS]
    a.boxplot(errs, tick_labels=METHODS, showfliers=True)
    a.axhline(0, color="grey", lw=0.8)
    a.set_ylabel("error per 5 m run (m)")
    a.set_title("Cross-validation: held-out 5 m runs")
    a.tick_params(axis="x", rotation=20)
    for name, (t, cum) in curves.items():
        b.plot(t, cum, label=f"{name}: {totals[name]:.1f} m")
    b.axhline(27, color="black", ls="--", lw=1, label="true: 27 m")
    b.set_xlabel("time (s)")
    b.set_ylabel("cumulative distance (m)")
    b.set_title("Unseen 27 m straight + zigzag recording")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_markdown(run_summary, window_metrics, totals, path):
    lines = ["# Results", "",
             "## Cross-validation: error on held-out 5 m runs (40 runs, 5 folds)", "",
             "| Method | MAE (m) | MAPE | Mean error (m) | Worst run (m) |",
             "|---|---|---|---|---|"]
    for m in METHODS:
        s = run_summary[m]
        lines.append(f"| {m} | {s['MAE_m']:.2f} | {s['MAPE_pct']:.1f}% | "
                     f"{s['mean_error_m']:+.2f} | {s['worst_m']:.2f} |")
    lines += ["", "MAE per run (m) by recording:", "",
              "| Method | " + " | ".join(TRAIN_RECORDINGS) + " |",
              "|---|" + "---|" * len(TRAIN_RECORDINGS)]
    for m in METHODS:
        lines.append(f"| {m} | " + " | ".join(f"{run_summary[m]['by_recording'][r]:.2f}"
                                               for r in TRAIN_RECORDINGS) + " |")
    lines += ["", "## Window-level speed (2 s windows, held-out folds)", "",
              "| Model | MAE (m/s) | RMSE (m/s) | R² |", "|---|---|---|---|"]
    for m, s in window_metrics.items():
        lines.append(f"| {m} | {s['MAE']:.3f} | {s['RMSE']:.3f} | {s['R2']:.3f} |")
    lines += ["", "## Final test: unseen 27 m recording", "",
              "| Method | Estimate (m) | Error (m) | Error (%) |", "|---|---|---|---|"]
    for m in METHODS:
        e = totals[m] - 27.0
        lines.append(f"| {m} | {totals[m]:.1f} | {e:+.1f} | {abs(e) / 27 * 100:.1f}% |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    data = load_all()
    for name in TRAIN_RECORDINGS:
        speeds = [5.0 / r.moving_s for r in data[name]["runs"]]
        print(f"{name}: {len(data[name]['runs'])} runs, speed {np.mean(speeds):.2f} m/s")

    runs, window_metrics, fold_info = cross_validate(data, args)
    run_summary = summarise_runs(runs)
    totals, curves, final_info = final_test(data, args)

    runs.round(3).to_csv(RESULTS / "cv_run_predictions.csv", index=False)
    (RESULTS / "metrics.json").write_text(json.dumps(
        {"cv_runs": run_summary, "cv_windows": window_metrics, "test_27m": totals,
         "fold_models": fold_info, "final_model": final_info, "seed": args.seed},
        indent=2, default=str))
    write_markdown(run_summary, window_metrics, totals, RESULTS / "metrics.md")
    plot_segmentation(data, RESULTS / "segmentation.png")
    plot_results(runs, curves, totals, RESULTS / "results.png")
    print((RESULTS / "metrics.md").read_text())


if __name__ == "__main__":
    main()
