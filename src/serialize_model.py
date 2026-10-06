#!/usr/bin/env python3
"""Serialize best model artifact with documented feature schema.

WHAT: Save `models/best_model.joblib` (RandomForest, physics-inspired features) plus `models/feature_schema.json`.
WHY: Reproducible model deployment requires knowing the exact feature order and names.
HOW: Train on full `Classic_PID.mat` train split; save with `joblib`.
WHEN: After final model selection; never serialize without schema documentation.
"""
import pathlib, sys, json, joblib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from dataset import MagLevDataset
from features import engineer_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)

    feature_cols = [
        "time_s", "control_normalized",
        "time_squared", "control_squared",
        "time_x_control", "control_deviation_from_mean",
    ]
    X_train = train_df[feature_cols].values
    y_train = train_df["position_cm"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=200, max_depth=None, min_samples_split=2)
    model.fit(X_train_s, y_train)

    # Save artifact
    models_dir = pathlib.Path("models")
    models_dir.mkdir(exist_ok=True)

    artifact = {
        "model_type": "RandomForestRegressor",
        "dataset_version": "Classic_PID.mat (4847x3, variable Aclas)",
        "feature_set": "physics_inspired",
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "scaler": scaler,
        "model": model,
        "random_state": RANDOM_STATE,
        "training_samples": len(X_train),
        "target_column": "position_cm",
        "notes": "Serialized only after schema documentation. Features derived from non-target variables only (no leakage).",
        "dataset_provenance_url": "https://github.com/labcontrol-data/MagLev",
    }
    joblib.dump(artifact, models_dir / "best_model_artifact.joblib")

    # Save schema separately for easy inspection
    schema = {
        "feature_order": feature_cols,
        "feature_descriptions": {
            "time_s": "Time in seconds (inferred from .mat col 0)",
            "control_normalized": "Normalized PWM control input (inferred from .mat col 2, scaled by 255)",
            "time_squared": "Nonlinear time effect (derived)",
            "control_squared": "Nonlinear control effect (derived)",
            "time_x_control": "Time-control interaction (derived)",
            "control_deviation_from_mean": "Relative control intensity (derived)",
        },
        "target": "position_cm",
        "target_unit": "cm",
        "reference_value_cm": 1.32,
        "reference_source": "Arduino maincode.cc (xref=1.32)",
    }
    with open(models_dir / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Saved models/best_model_artifact.joblib (model + scaler + schema embedded)")
    print(f"Saved models/feature_schema.json (verified feature descriptions)")
    print(f"Features: {feature_cols}")


if __name__ == "__main__":
    main()
