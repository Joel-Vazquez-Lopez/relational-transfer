# Compass Investigation 21 — Exact-position operand shielding

A standalone exploratory scientific package: **no new training, pretrained LLM inference, sealed data or human-annotated sources**.

## Reproduce

```bash
python standalone_certificate.py   # stdlib; reproduce known-mechanism controls
python exp21_factorial.py         # CPU PyTorch; re-run 3 frozen checkpoints
python shortcut_controls.py      # stdlib + local sources, known algorithms
python -m unittest -v test_exp21  # 5 checks of gold, replay and uncertainty units
python verify_manifest.py        # verify file hashes from manifest
```

Read `RESEARCH_REPORT.md` for numbers and limitations. Source-world-level repeated predictions are **not independent observations**. This package is for review and falsification, not proof of internal neural computation.

Research branch: `research/compass-reuse`; no merge to main.
**Verified archive:** `COMPASS_INVESTIGATION_21_EXACT_POSITION_SHIELDING_2026-10-16.zip` (SHA-256 `48f622d675e123125de537ef84b42a7529ef9ea658e5d242af85f1deb8f49d35`). GitHub carries the [standalone Python certificate](standalone_certificate.py); neural weights and raw outputs remain in the separate conversation archive.
