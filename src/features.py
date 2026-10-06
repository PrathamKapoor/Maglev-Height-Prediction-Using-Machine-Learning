"""Feature engineering — only justified, derivable features without data leakage.

Verified source variables (no headers in .mat; inferred from code):
- time_s (col 0)
- position_cm (col 1) — TARGET VARIABLE (equilibrium height)
- control_normalized (col 2)

Verified physics reference from Arduino `maincode.cc`:
- xref = 1.32 cm

CRITICAL: `position_cm` is the target. It must NOT appear in feature lists
for regression models, otherwise results represent perfect leakage (R² ≈ 1)
and are scientifically meaningless.

Justified derived features (derived ONLY from non-target variables):
1. time_squared = (time_s)^2  — nonlinear temporal dynamics
2. control_squared = (control_normalized)^2  — nonlinear magnetic force effect
3. time_x_control = time_s * control_normalized  — time-dependent control interaction
4. control_reference_deviation = control_normalized - mean(control_normalized)  — relative control intensity

NOT implemented (missing variables, not invented):
- current_squared (no current measurement in dataset or code)
- inverse_gap (no gap measurement)
- position_reference_deviation (uses target — would cause leakage)
- position_squared (uses target — would cause leakage)
- time_x_position (uses target — would cause leakage)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

XREF_CM = 1.32  # Verified from Arduino `maincode.cc`: `xref=1.32`


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate physics-informed and interaction features without leakage.

    WHAT: Add `time_squared`, `control_squared`, `time_x_control`, `control_deviation_from_mean`.
    WHY: Nonlinear dynamics and interaction terms capture physical behavior without using the target as input.
    HOW: Vectorized arithmetic on pandas Series.
    WHEN: After dataset loading, before split/preprocessing, to avoid target leakage in derived features.
    """
    df = df.copy()
    df["time_squared"] = df["time_s"] ** 2
    df["control_squared"] = df["control_normalized"] ** 2
    df["time_x_control"] = df["time_s"] * df["control_normalized"]
    mean_control = df["control_normalized"].mean()
    df["control_deviation_from_mean"] = df["control_normalized"] - mean_control
    return df
