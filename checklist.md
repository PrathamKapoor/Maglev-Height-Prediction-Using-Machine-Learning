# Phase 1 — Dataset Acquisition and Validation

- [x] Inspect dataset source URL (https://github.com/labcontrol-data/MagLev)
- [x] Download/clone dataset repo
- [x] Identify dataset files (`Classic_PID.mat`, `Proposed_PID.mat`)
- [x] Identify target column (inferred: position measurement, col 1 in `.mat` arrays)
- [x] Identify feature columns (inferred: time, control input; actual variables lack header names)
- [x] Verify file format (`.mat` binary, loadable via `scipy.io.loadmat`)
- [x] Report dataset shapes (`Classic`: 4847x3, `Proposed`: 1638x3)
- [ ] Verify units (inferred from Arduino code: position ~cm, time ~s, control normalized 0-1)
- [ ] Validate missing values
- [ ] Validate duplicate rows
- [ ] Implement loader
- [ ] Add validation
- [ ] Add tests
- [ ] Run dataset loader
- [ ] Generate dataset report
- [ ] Update decisions.md
- [ ] Update flow.md
- [ ] Update handoff.md
