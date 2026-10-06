# Phase 0 — Repository Audit

- [x] Inspect directory: `C:/Projects/Maglev` (empty, 0 files)
- [x] Inspect git: not initialized
- [x] Inspect dataset: none present (no .csv, .xlsx, .parquet, .json)
- [x] Inspect docs: `decisions.md`, `flow.md`, `handoff.md` missing
- [x] Inspect requirements: none present
- [x] Python version: 3.13.14 (available)
- [x] Git identity: `PrathamKapoor` configured

Next: ask about dataset source/provenance before proceeding.

### D-001: Dataset Source Verified (MagLev GitHub repo)
- Date: 2026-10-06 / Phase 1
- Context: User provided `https://github.com/labcontrol-data/MagLev`. Repository cloned (partial due to timeout); `.mat` files extracted to `data/raw/`.
- Decision: Use real experimental dataset (`Classic_PID.mat`, `Proposed_PID.mat`) rather than synthetic data.
- Why: Real data available; synthetic not required.
- Alternatives: Synthetic dataset (rejected — real data exists and is scientifically justified).
- Consequences: Column names must be INFERRED from Arduino code (`maincode.cc`) since `.mat` files have no headers. Provenance fully documented.
- Files: `data/raw/*.mat`, `src/dataset.py`, `tests/test_dataset.py`, `decisions.md` (this entry)

### D-002: Column Names Inferred, Not Invented
- Date: 2026-10-06 / Phase 1
- Context: `.mat` variables `Aclas` and `A` have no header documentation in repo.
- Decision: Assign descriptive names (`time_s`, `position_cm`, `control_normalized`) and clearly document them as INFERRED from `xref=1.32` (Arduino) and PWM logic.
- Why: The dataset provides no headers; the loader must label columns for modeling without inventing physical relationships.
- Alternatives: Leave unnamed (rejected — modeling requires column references).
- Consequences: Any future user of the dataset must verify these inferences against the paper or updated source.
- Files: `src/dataset.py`, `reports/dataset_report.txt` (future)

### D-003: No Unverified Physics Features
- Date: 2026-10-06 / Phase 1
- Context: Instructions ask for physics-inspired features such as `current_squared`, `inverse_gap`.
- Decision: Only implement derivable features (`position_reference_deviation` from `xref`); do NOT invent `current` or `gap` variables that are absent from the dataset.
- Why: The dataset contains only time, position, and control. No current, voltage, gap, or mass measurements are present in `.mat` files or Arduino code.
- Alternatives: Invent synthetic current/gap (rejected — violates no-fabrication rule).
- Consequences: Physics-inspired feature experiment will be limited to what is derivable from position and reference value; results will document this limitation.
