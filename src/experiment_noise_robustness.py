#!/usr/bin/env python3
"""Noise robustness: add controlled synthetic noise to target; evaluate best model.

WHAT: Add 0%, 1%, 5%, 10% noise to `position_cm` (target only); evaluate best RF model.
WHY: Answer: How robust is the model to noisy measurements?
HOW: Synthetic Gaussian noise added to target ONLY (not features); clearly labeled.
WHEN: After baseline model finalized; results clearly labeled synthetic.
"""
import pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from dataset import MagLevDataset
from features import engineer_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def add_noise(y, noise_pct, seed=42):
    np.random.seed(seed)
    std = np.std(y)
    noise_std = std * noise_pct
    return y + np.random.normal(0, noise_std, size=y.shape)


def main():
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    df = pd.DataFrame({
        "time_s": arr[:, 0],
        "position_cm": arr[:, 1],
        "control_normalized": arr[:, 2],
    })
    df = engineer_features(df)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)

    feature_cols = ["time_s", "control_normalized", "time_squared", "control_squared",
                    "time_x_control", "control_deviation_from_mean"]
    X_train = train_df[feature_cols].values
    y_train_orig = train_df["position_cm"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["position_cm"].values

    noise_levels = [0.0, 0.01, 0.05, 0.10]
    results = []

    for level in noise_levels:
        # Synthetic noise added to TRAIN TARGET ONLY (simulating measurement noise)
        y_train_noisy = add_noise(y_train_orig, level, seed=RANDOM_STATE)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, max_depth=10, min_samples_split=2)
        model.fit(X_train_s, y_train_noisy)
        y_pred = model.predict(X_test_s)

        results.append({
            "noise_level": level,
            "noise_description": f"{int(level*100)}% synthetic Gaussian noise on training target",
            "test_mae_cm": float(mean_absolute_error(y_test, y_pred)),
            "test_r2": float(r2_score(y_test, y_pred)),
            "synthetic": True,
            "seed": RANDOM_STATE,
            "note": "Synthetic noise added only to training target. Real measurements unchanged in test set.",
        })
        print(f"Noise {level:.0%}: MAE={results[-1]['test_mae_cm']:.6f} cm, R²={results[-1]['test_r2']:.6f}")

    reports_dir = pathlib.Path("reports")
    with open(reports_dir / "noise_robustness.json", "w") as f:
        json.dump({
            "experiment_type": "noise_robustness",
            "dataset": "Classic_PID.mat",
            "model": "RandomForestRegressor",
            "synthetic_note": "All noise levels are artificially generated. Results are labeled synthetic.",
            "random_state": RANDOM_STATE,
            "results": results,
        }, f, indent=2)
    print(f"Saved reports/noise_robustness.json (all levels labeled synthetic)")


if __name__ == "__main__":
    main()
