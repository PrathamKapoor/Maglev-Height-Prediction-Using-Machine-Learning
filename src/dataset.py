"""MagLev ML dataset loader and validator.

Dataset provenance:
- Source: https://github.com/labcontrol-data/MagLev
- Paper: "Semi-Active Magnetic Levitation System for Education"
  (MDPI Applied Sciences, 2021)
- DOI: 10.5281/zenodo.4678906
- License: See repo LICENSE (academic/research use permitted with attribution)
- Files: data/raw/Classic_PID.mat, data/raw/Proposed_PID.mat

NOTE: The source `.mat` files contain NO column headers. The variable keys
(`Aclas` for Classic_PID, `A` for Proposed_PID) are not documented in the
repository. Column meanings are INFERRED from the Arduino source code
(`maincode.cc`) and the paper:

- Column 0: time (seconds) — inferred from range 0-99 and experiment duration
- Column 1: measured position `x` (cm) — matches Arduino `xref = 1.32` cm
- Column 2: control input `u` normalized by 255 — inferred from Arduino PWM logic

These inferences are documented but not guaranteed by the dataset authors.
If the dataset is updated with headers, this loader should be revised.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import numpy as np
import scipy.io


DATASET_PROVENANCE: Dict[str, str] = {
    "source_url": "https://github.com/labcontrol-data/MagLev",
    "dataset_repo": "https://github.com/labcontrol-data/MagLev",
    "paper_title": "Semi-Active Magnetic Levitation System for Education",
    "paper_url": "https://www.mdpi.com/2076-3417/11/12/5330",
    "doi": "10.5281/zenodo.4678906",
    "license_note": "Academic/research use permitted; cite source and contact authors.",
    "collection_method": "Laboratory testbed measurements at Universitat Politecnica de Catalunya-BarcelonaTech (2021).",
    "files": ["Classic_PID.mat", "Proposed_PID.mat"],
    "format": ".mat (MATLAB binary, readable via scipy.io.loadmat)",
    "variable_keys": {"Classic_PID.mat": "Aclas", "Proposed_PID.mat": "A"},
    "inferred_columns": [
        "time_s (inferred from Arduino experiment duration)",
        "position_cm (inferred from xref=1.32 in maincode.cc)",
        "control_normalized (inferred from Arduino PWM 0-255 scaled by 255)",
    ],
}


class MagLevDataset:
    """Loader and validator for MagLev `.mat` dataset files."""

    def __init__(self, data_dir: pathlib.Path | str | None = None):
        if data_dir is not None:
            self.data_dir = pathlib.Path(data_dir)
        else:
            default_path = pathlib.Path("data/raw")
            if default_path.exists():
                self.data_dir = default_path
            else:
                repo_root = pathlib.Path(__file__).resolve().parent.parent
                fallback = repo_root / "data" / "raw"
                if fallback.exists():
                    self.data_dir = fallback
                else:
                    self.data_dir = default_path
        self.files: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, object] = {}

    def load(self, filename: str) -> np.ndarray:
        """Load a `.mat` file and return the numeric array."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        mat = scipy.io.loadmat(str(path))
        # Extract the first non-dunder key
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No data variable found in {filename}")
        var_name = keys[0]
        arr = mat[var_name]
        if arr.dtype != np.float64 and arr.dtype != np.float32:
            # Some .mat files may return ints; coerce safely
            arr = arr.astype(np.float64)
        self.files[filename] = arr
        self.metadata[filename] = {
            "variable_key": var_name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
        }
        return arr

    def validate(self, arr: np.ndarray, expected_shape: Tuple[int, int] | None = None) -> Dict[str, object]:
        """Run basic validation checks on loaded array."""
        report = {
            "shape": arr.shape,
            "nan_count": int(np.sum(np.isnan(arr))),
            "inf_count": int(np.sum(np.isinf(arr))),
            "duplicate_rows": int(arr.shape[0] - np.unique(arr, axis=0).shape[0]),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
        }
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {arr.shape}")
        return report
