import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from dataset import MagLevDataset, DATASET_PROVENANCE


def test_dataset_load_classic():
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    assert arr.shape == (4847, 3)
    assert loader.metadata["Classic_PID.mat"]["variable_key"] == "Aclas"


def test_dataset_load_proposed():
    loader = MagLevDataset()
    arr = loader.load("Proposed_PID.mat")
    assert arr.shape == (1638, 3)
    assert loader.metadata["Proposed_PID.mat"]["variable_key"] == "A"


def test_validation_no_nans():
    loader = MagLevDataset()
    arr = loader.load("Classic_PID.mat")
    report = loader.validate(arr)
    assert report["nan_count"] == 0
    assert report["inf_count"] == 0
    assert report["duplicate_rows"] == 0


def test_provenance_documented():
    assert DATASET_PROVENANCE["source_url"].startswith("https://")
    assert "doi" in DATASET_PROVENANCE
