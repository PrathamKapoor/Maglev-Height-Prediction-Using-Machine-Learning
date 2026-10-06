# MagLev-ML System Flow

## Data Source
```
https://github.com/labcontrol-data/MagLev (cloned to data_repo/)
  ├─ data/Classic_PID.mat  (4847 x 3, variable key: Aclas)
  ├─ data/Proposed_PID.mat (1638 x 3, variable key: A)
  └─ code/maincode.cc       (Arduino control code, reference xref=1.32 cm)
```

## Pipeline Execution Flow

### 1. Interactive Walkthrough (Canonical Notebook)
```text
notebooks/MagLev_ML_Complete_Analysis.ipynb
 └─ MagLevDataset.load()
     └─ MagLevDataset.validate()
         └─ Visual Exploratory Data Analysis (EDA)
             └─ engineer_features() (non-target derived features only)
                 └─ train_test_split() (70% train / 15% val / 15% test, seed=42)
                     └─ StandardScaler.fit(X_train) / transform(val, test)
                         └─ Model Training (Linear, Poly, RF, GB, MLP)
                             └─ 5-Fold GridSearchCV on Train
                                 └─ Evaluation on Held-Out Test Set
                                     └─ Residual Diagnostics & Physical Interpretation
```

### 2. Command-Line Batch Pipeline
```text
train.py
  └─ dataset.load()
      └─ dataset.validate()
          └─ engineer_features()
              └─ split_dataset()
                  └─ build_preprocessor()
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

### Operation: Feature Engineering (Physics-Inspired, Leakage-Free)
- WHAT: Derive non-target features: `time_squared`, `control_squared`, `time_x_control`, `control_deviation_from_mean`.
- WHY: Quadratic magnetic force behavior ($F \propto i^2$) and time-varying dynamics can be modeled without using the target variable.
- HOW: Vectorized operations in `src/features.py`. Target-dependent features (`position_cm - xref`, `position_squared`) are strictly omitted to prevent data leakage.
- WHEN: Before train/test split, ensuring target column `position_cm` is strictly isolated as the prediction label.
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
