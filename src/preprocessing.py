"""Preprocessing pipeline with leakage prevention."""
from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Pipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.train_indices: np.ndarray | None = None
        self.test_indices: np.ndarray | None = None

    def split(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train/val/test. Never uses complete data for fitting."""
        train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=self.random_state)
        val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=self.random_state)
        return train_df, val_df, test_df

    def fit_transform_train(self, train_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        X_train = train_df[feature_cols].values
        self.scaler.fit(X_train)
        return self.scaler.transform(X_train)

    def transform(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        X = df[feature_cols].values
        return self.scaler.transform(X)
