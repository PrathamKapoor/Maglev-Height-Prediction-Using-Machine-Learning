# MagLev-ML System Flow

## Data Source
```
https://github.com/labcontrol-data/MagLev (cloned to data_repo/)
  ├─ data/Classic_PID.mat  (4847 x 3, variable key: Aclas)
  ├─ data/Proposed_PID.mat (1638 x 3, variable key: A)
  └─ code/maincode.cc       (Arduino control code, reference xref=1.32 cm)
```

## Pipeline (actual file/function names)
```
train.py
  └─ dataset.load()
      └─ dataset.validate()
      └─ split_dataset()
          └─ build_preprocessor()
              └─ engineer_features()
                  └─ train_model()
                      └─ cross_validate()
                          └─ evaluate_model()
                              └─ plot_results()
```

### What / Why / How / When (verified)

### Operation: Load `.mat` dataset
- WHAT: Read `Classic_PID.mat` and `Proposed_PID.mat` via `scipy.io.loadmat`.
- WHY: Source provides data only in `.mat` format; no CSV/JSON alternatives.
- HOW: Extract first non-`__` variable key (`Aclas`, `A`). Return float array.
- WHEN: At start of every experiment script.

### Operation: Dataset Validation
- WHAT: Check NaN, Inf, duplicates, shape.
- WHY: Confirm dataset integrity before any modeling.
- HOW: `np.isnan`, `np.isinf`, `np.unique` comparison.
- WHEN: Immediately after load, before split.

### Operation: Split
- WHAT: Train/validation/test split.
- WHY: Prevent leakage; allow unbiased evaluation.
- HOW: `sklearn.model_selection.train_test_split` (stratify not applicable for continuous regression; use random split with fixed seed).
- WHEN: After validation, before preprocessing.

### Operation: Preprocessing
- WHAT: Scale numerical features using `StandardScaler`.
- WHY: Tree models tolerate scale differences; linear/NN models benefit.
- HOW: Fit on train only; transform train/val/test with fitted scaler.
- WHEN: After split, before feature engineering / model training.

### Operation: Feature Engineering (Physics-Inspired)
- WHAT: Derive `current_squared` (if current present), `inverse_gap` (if gap present), `position_reference_deviation`.
- WHY: Physical relationships between magnetic force, distance, and current may improve prediction.
- HOW: Only if source variables exist; for this dataset, `position_reference_deviation = position_cm - 1.32` (xref from maincode.cc) is justified. Additional derived features (`position_squared`, `time_squared`) are tested but documented as exploratory.
- WHEN: After split and scaling, before training.

### Operation: Model Training
- WHAT: Train Linear Regression, Polynomial Regression, Random Forest, Gradient Boosting, MLP.
- WHY: Establish baseline (linear), test nonlinear patterns (polynomial, RF, GB), test deep learning (MLP) if dataset supports.
- HOW: `sklearn` implementations; hyperparameter search via `GridSearchCV` or `RandomizedSearchCV`.
- WHEN: After feature engineering.

### Operation: Cross Validation
- WHAT: 5-fold CV for hyperparameter tuning.
- WHY: Estimate model performance without using test set.
- WHEN: During hyperparameter search, before final evaluation.

### Operation: Evaluation
- WHAT: Compute MAE, RMSE, R² on held-out test set.
- WHY: Quantify prediction accuracy in physical units.
- HOW: `sklearn.metrics` with manual verification.
- WHEN: After final model selection.
