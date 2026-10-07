# Results

## Cross-validation: error on held-out 5 m runs (40 runs, 5 folds)

| Method | MAE (m) | MAPE | Mean error (m) | Worst run (m) |
|---|---|---|---|---|
| Integration | 3.03 | 60.6% | -2.96 | 4.44 |
| Constant speed | 0.38 | 7.7% | +0.05 | 1.15 |
| Random Forest | 0.35 | 6.9% | -0.00 | 1.35 |
| SVR | 0.36 | 7.1% | -0.05 | 0.91 |
| CNN-GRU | 0.37 | 7.3% | -0.04 | 0.92 |

MAE per run (m) by recording:

| Method | straight_slow | straight_fast | curve_slow | curve_fast |
|---|---|---|---|---|
| Integration | 3.40 | 2.90 | 3.44 | 2.39 |
| Constant speed | 0.19 | 0.51 | 0.59 | 0.26 |
| Random Forest | 0.19 | 0.43 | 0.36 | 0.40 |
| SVR | 0.27 | 0.44 | 0.30 | 0.42 |
| CNN-GRU | 0.26 | 0.45 | 0.42 | 0.34 |

## Window-level speed (2 s windows, held-out folds)

| Model | MAE (m/s) | RMSE (m/s) | R² |
|---|---|---|---|
| Random Forest | 0.052 | 0.077 | 0.938 |
| SVR | 0.053 | 0.075 | 0.941 |
| CNN-GRU | 0.050 | 0.070 | 0.950 |

## Final test: unseen 27 m recording

| Method | Estimate (m) | Error (m) | Error (%) |
|---|---|---|---|
| Integration | 6.2 | -20.8 | 77.0% |
| Constant speed | 28.0 | +1.0 | 3.7% |
| Random Forest | 23.3 | -3.7 | 13.9% |
| SVR | 23.6 | -3.4 | 12.7% |
| CNN-GRU | 24.2 | -2.8 | 10.2% |
