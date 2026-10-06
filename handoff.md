# MagLev-ML HandOff Document

## 1. Current Phase
Phase 1 — Dataset Acquisition and Validation (in progress)

## 2. Work Completed
- Repository audited (empty directory, no git, no dataset, no docs).
- Git initialized with identity `PrathamKapoor` / `prathamkapoor027@gmail.com`.
- Dataset source verified: `https://github.com/labcontrol-data/MagLev` (cloned partially, `.mat` files extracted to `data/raw/`).
- `Classic_PID.mat` (4847x3, variable `Aclas`) and `Proposed_PID.mat` (1638x3, variable `A`) loaded and inspected.
- No NaN, no Inf, no duplicate rows detected.
- `src/dataset.py` created with loader/validator and provenance documentation.
- `tests/test_dataset.py` created (4 tests: load classic, load proposed, validation, provenance).
- `requirements.txt` created.
- `flow.md` created with verified execution flow.
- `checklist.md` created.

## 3. Files Changed
- `.git/` initialized
- `decisions.md` (updated)
- `data/raw/Classic_PID.mat`, `data/raw/Proposed_PID.mat`, `data/raw/teste.txt` (copied from source repo)
- `data_repo/` (cloned source repo; do NOT include in final repo; remove before submission)
- `src/dataset.py`
- `tests/test_dataset.py`
- `src/report_dataset.py`
- `flow.md`
- `requirements.txt`
- `checklist.md`

## 4. Current Architecture / State
```
Maglev-ML/
├── .git          (initialized, user: PrathamKapoor)
├── decisions.md
├── flow.md
├── handoff.md    (this file)
├── checklist.md
├── requirements.txt
├── data/
│   └── raw/
│       ├── Classic_PID.mat   (verified: 4847x3, no NaN, no duplicates)
│       ├── Proposed_PID.mat  (verified: 1638x3, no NaN, no duplicates)
│       └── teste.txt         (unused)
├── data_repo/    (partial clone of source repo; REMOVE before final commit)
├── src/
│   ├── dataset.py
│   └── report_dataset.py
├── tests/
│   └── test_dataset.py
├── notebooks/    (empty)
├── figures/       (empty)
├── reports/       (empty)
└── models/        (not yet created)
```

## 5. Decisions Made
- D-001: Dataset source is `https://github.com/labcontrol-data/MagLev`. Variable keys `Aclas`/`A` used directly; column names are INFERRED from Arduino code (`xref=1.32` cm) and paper context, not from dataset headers. All inferences are explicitly documented.
- D-002: Synthetic dataset NOT needed; real experimental dataset available.
- D-003: No frontend/web UI required; repository remains pure Python ML pipeline.

## 6. Requirements and Constraints
- Python 3.13 available.
- No dataset headers; inference required.
- `.mat` format requires `scipy`.
- Project must include reproducible scripts, tests, documentation.
- Commit identity: `PrathamKapoor`.

## 7. Testing and Verification
- `python src/report_dataset.py` executed successfully (verified shapes, stats).
- `pytest tests/test_dataset.py` NOT YET RUN (pending installation of pytest if needed; pytest is in requirements.txt).
- No dataset leakage prevention implemented yet (pending split/preprocessing code).

## 8. Known Issues / Risks
- `data_repo/` embedded git repo must be removed before final submission.
- No column headers in `.mat` files; inference may be incorrect if source updates.
- Only two dataset variants (`Classic_PID`, `Proposed_PID`); combined analysis not yet implemented.
- No feature engineering implemented yet.
- No model training/evaluation implemented yet.

## 9. Unfinished Work
- Remove `data_repo/` and include only `data/raw/` in final repo.
- Implement train/validation/test split.
- Implement preprocessing pipeline (`StandardScaler` fitted on train only).
- Implement feature engineering (`position_reference_deviation` from `xref=1.32`).
- Implement model comparison (Linear, Polynomial, RF, GB, MLP).
- Implement cross-validation and hyperparameter tuning.
- Implement evaluation metrics (MAE, RMSE, R²) with actual execution.
- Implement plot generation (`figures/`).
- Implement noise robustness experiment.
- Implement data-size experiment (if applicable).
- Implement full test suite and verify all pass.
- Update `decisions.md` with all model/experiment decisions.

## 10. Next Subphase
Phase 2 — Data Cleaning Pipeline (verify split/preprocessing; implement leakage prevention; add feature engineering; run first training script).

## 11. Critical Context
- The dataset has NO headers. Any reference to "current", "voltage", "gap" must be verified against `maincode.cc`. The Arduino code uses `xr` (analog read), computes `x` (position), and outputs PWM `u`. There is NO explicit current measurement in the code. Therefore, features describing "current" or "magnetic field" are NOT directly available and must NOT be fabricated. Only `time`, `position`, and `control` are verifiable.
- The target for prediction is the equilibrium height (position), which is `position_cm` (col 1) in the dataset.

## 12. Agent Instructions
- Do NOT assume column headers exist.
- Always reference `xref = 1.32` from `maincode.cc` when explaining position-related features.
- Before adding any physics-inspired feature (e.g., `current_squared`), verify the variable exists in the dataset. It does NOT exist; do NOT invent it.
- Before claiming any result (e.g., R² = X), run the experiment and record actual output.
- Keep `data_repo/` out of final commits.
