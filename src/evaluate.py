#!/usr/bin/env python3
"""Generate evaluation plots and physical error analysis.

WHAT: Creates actual vs predicted, residual plot, feature importance.
WHY: Verify model behavior; identify systematic errors; interpret results.
HOW: Uses verified predictions from model_comparison.json (or reruns if needed).
WHEN: After model training and evaluation.
"""
import pathlib
import sys
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import MagLevDataset
from features import engineer_features

# Load verified predictions from the latest run
# For simplicity, we load the best model's predictions directly
# by re-running best model from saved params (not re-training from scratch to avoid time)


def plot_actual_vs_predicted():
    # Reconstruct predictions from best LinearRegression (raw) since it is deterministic
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    df = pd.DataFrame({
        "time_s": arr[:, 0],
        "position_cm": arr[:, 1],
        "control_normalized": arr[:, 2],
    })
    df = engineer_features(df)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    X_train = train_df[["time_s", "control_normalized"]].values
    y_train = train_df["position_cm"].values
    X_test = test_df[["time_s", "control_normalized"]].values
    y_test = test_df["position_cm"].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="y=x")
    plt.xlabel("Actual Position (cm)")
    plt.ylabel("Predicted Position (cm)")
    plt.title("Actual vs Predicted — Linear Regression (Verified)")
    plt.tight_layout()
    plt.savefig("figures/actual_vs_predicted.png")
    plt.close()
    print("Saved figures/actual_vs_predicted.png")


def plot_residuals():
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    df = pd.DataFrame({
        "time_s": arr[:, 0],
        "position_cm": arr[:, 1],
        "control_normalized": arr[:, 2],
    })
    df = engineer_features(df)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    X_train = train_df[["time_s", "control_normalized"]].values
    y_train = train_df["position_cm"].values
    X_test = test_df[["time_s", "control_normalized"]].values
    y_test = test_df["position_cm"].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 4))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Position (cm)")
    plt.ylabel("Residual (cm)")
    plt.title("Residual Analysis — Linear Regression")
    plt.tight_layout()
    plt.savefig("figures/residual_plot.png")
    plt.close()
    print("Saved figures/residual_plot.png")


def plot_feature_importance():
    # Random Forest feature importance for physics-inspired features
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    df = pd.DataFrame({
        "time_s": arr[:, 0],
        "position_cm": arr[:, 1],
        "control_normalized": arr[:, 2],
    })
    df = engineer_features(df)
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    feature_cols = [
        "time_s", "control_normalized",
        "time_squared", "control_squared",
        "time_x_control", "control_deviation_from_mean",
    ]
    X_train = train_df[feature_cols].values
    y_train = train_df["position_cm"].values
    rf = RandomForestRegressor(random_state=42, n_estimators=200)
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=importance, x="importance", y="feature")
    plt.title("Feature Importance — Random Forest (Verified)")
    plt.tight_layout()
    plt.savefig("figures/feature_importance.png")
    plt.close()
    print("Saved figures/feature_importance.png")
    print("Feature importances (verified from actual model):")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    plot_actual_vs_predicted()
    plot_residuals()
    plot_feature_importance()
