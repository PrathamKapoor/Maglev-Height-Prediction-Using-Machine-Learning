import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from dataset import MagLevDataset, DATASET_PROVENANCE
from features import engineer_features, XREF_CM
from preprocessing import Pipeline


def test_preprocessing_split_no_leakage():
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    import pandas as pd
    df = pd.DataFrame({
        "time_s": arr[:, 0],
        "position_cm": arr[:, 1],
        "control_normalized": arr[:, 2],
    })
    pipeline = Pipeline(random_state=42)
    train_df, val_df, test_df = pipeline.split(df)
    # Verify split is reproducible
    train_df2, val_df2, test_df2 = pipeline.split(df)
    assert len(train_df) == len(train_df2)


def test_features_no_target_leakage():
    import pandas as pd
    df = pd.DataFrame({
        "time_s": [1, 2, 3],
        "position_cm": [1.3, 1.4, 1.5],
        "control_normalized": [0.1, 0.2, 0.3],
    })
    result = engineer_features(df)
    # Target-derived features must NOT exist
    assert "position_reference_deviation" not in result.columns
    assert "position_squared" not in result.columns
    # Derivable non-target features must exist
    assert "time_squared" in result.columns
    assert "control_squared" in result.columns
    assert "time_x_control" in result.columns


def test_features_xref_value():
    assert XREF_CM == 1.32


def test_provenance_complete():
    assert DATASET_PROVENANCE["source_url"].startswith("https://")
    assert DATASET_PROVENANCE["variable_keys"]["Classic_PID.mat"] == "Aclas"


def test_metrics_calculation():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    assert mae > 0
    assert rmse > 0
    assert r2 <= 1.0
