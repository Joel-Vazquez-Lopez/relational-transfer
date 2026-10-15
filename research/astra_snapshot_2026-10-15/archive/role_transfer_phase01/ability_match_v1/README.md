# Ability-matched ordered-role development test

This stage follows the completed Phase 2 report. It uses the **same eight previously examined graph families**. Its sole purpose is to test whether nonzero ordered-role supervision can be compared with participant-only supervision at closely matched general transfer performance. It is development, not independent confirmation.

The frozen [PROTOCOL.json](PROTOCOL.json) was written before any calibration model ran. The preflight checked finite gradients for every candidate weight without evaluating the focal pair. Four fixed nonzero role-loss weights (0.001, 0.005, 0.02, 0.08) are calibrated on families 0 and 1, with sham/high conditions and both paired seeds. The selection code reads only the eight prespecified **nonfocal** unseen pairs and four familiar pairs per cell. It chooses one weight closest to the existing participant-only 6,000-step nonfocal score, subject to at least 95% familiar accuracy and a gap no larger than two points.

Only a passing calibration weight is trained across all eight development families. The same nonfocal ability gate must pass across all eight before `focal_eval.py` is allowed to load the focal pair. Family-specific ability gaps remain visible even if the aggregate gate passes. No candidate weights, seeds, graph families, model architecture, data schedule, optimizer, update count, or metric may be changed based on focal outcomes.

From this directory, with `/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python`:

```sh
python preflight.py
python run_all.py calibration
python select_weight.py calibration
python run_all.py full
python select_weight.py full
python focal_eval.py
```

Each runner stage skips only completed `metrics.json` directories and resumes a partial configuration from its saved model and optimizer state. The ledger and model hashes document execution. If either ability gate fails, stop **before** focal evaluation and report that this fixed calibration grid did not establish a valid match. A positive focal development effect would still require a separately frozen independent graph test; this stage does not launch that test.
