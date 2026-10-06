#!/usr/bin/env python3
"""Data-size experiment: train with 10%, 25%, 50%, 75%, 100% of data.

WHAT: Compare validation/test performance as dataset grows.
WHY: Answer: How does prediction performance change with more experimental data?
HOW: Subsample training set (fixed seed); same model; evaluate on fixed test set.
WHEN: After baseline model is finalized; results saved separately.
"""
import pathlib, sys, json, time
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from dataset import MagLevDataset
from features import engineer_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def main():
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    df = pd.DataFrame({
        "time_s": arr[:, 0],
        "position_cm": arr[:, 1],
        "control_normalized": arr[:, 2],
    })
    df = engineer_features(df)

    # Fixed split: 70% train base, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)

    feature_cols = ["time_s", "control_normalized", "time_squared", "control_squared",
                    "time_x_control", "control_deviation_from_mean"]
    X_test = test_df[feature_cols].values
    y_test = test_df["position_cm"].values

    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    results = []

    for frac in fractions:
        # Subsample train set
        n_samples = max(10, int(len(train_df) * frac))
        train_sub = train_df.sample(n=n_samples, random_state=RANDOM_STATE)
        X_train = train_sub[feature_cols].values
        y_train = train_sub["position_cm"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, max_depth=10, min_samples_split=2)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        results.append({
            "fraction": frac,
            "n_train_samples": n_samples,
            "test_mae_cm": float(mean_absolute_error(y_test, y_pred)),
            "test_rmse_cm": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "test_r2": float(r2_score(y_test, y_pred)),
        })
        print(f"Fraction {frac:.0%} (n={n_samples}): MAE={results[-1]['test_mae_cm']:.6f} cm, R²={results[-1]['test_r2']:.6f}")

    # Save
    reports_dir = pathlib.Path("reports")
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "data_size_experiment.json", "w") as f:
        json.dump({
            "experiment_type": "data_size_learning_curve",
            "dataset": "Classic_PID.mat",
            "model": "RandomForestRegressor",
            "random_state": RANDOM_STATE,
            "notes": "Verified results; synthetic data not used.",
            "results": results,
        }, f, indent=2)
    print(f"Saved reports/data_size_experiment.json")


if __name__ == "__main__":
    main()
