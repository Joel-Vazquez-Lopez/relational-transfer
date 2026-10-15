# Reproducing the relational role-transfer studies

Updated 14 October 2026. Run commands from `/Users/joelvazquezlopez/Documents/ChatGPT/project null`. The working Python environment is `/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python` (PyTorch, NumPy, SciPy, NetworkX). Preserve all existing output directories. Scripts refuse completed model directories and use saved hashes/ledgers to detect changes.

See [RESEARCH_CHECKPOINT_2026-10-14.md](RESEARCH_CHECKPOINT_2026-10-14.md) for the cross-study interpretation and [RESEARCH_ARTIFACT_MANIFEST_2026-10-14.json](RESEARCH_ARTIFACT_MANIFEST_2026-10-14.json) for local artifact checksums.

## Existing evidence

- [Fresh overlap study](role_transfer_phase01/rq1_overlap_confirmation_v1/): `PROTOCOL.json`, `GRAPH_MANIFEST.json`, `SUMMARY.json`, `RESULTS.md`, and all saved model metrics/checkpoints. Code: `role_transfer_phase01/rq1_overlap_confirmation_v1.py`.
- [Fresh rewire repair study](role_transfer_phase01/rq2_rewire_repair_v1/): same artifact pattern, including eight families × two conditions × two objectives × three seeds. Code: `role_transfer_phase01/rq2_rewire_repair_v1.py` and its runner.
- [Strict Phase 1 factorial](role_transfer_phase01/rq1_factorial12_strict_v1/): `GRAPH_MANIFEST.json`, `PROTOCOL.json`, `RUN_LEDGER.jsonl`, `PHASE1_STRICT_REPORT.md`, all 96 checkpoint directories. `python role_transfer_phase01/rq1_factorial12_strict_v1/summarize.py` reproduces the frozen summary.

## Completed Phase 2 development

Source: [experiment.py](role_transfer_phase01/phase2_development_v1/experiment.py), [analyze.py](role_transfer_phase01/phase2_development_v1/analyze.py), [run_all.py](role_transfer_phase01/phase2_development_v1/run_all.py). Exact design and hashes: [PROTOCOL.json](role_transfer_phase01/phase2_development_v1/PROTOCOL.json). The source graph manifest is `role_transfer_phase01/rq2_rewire_repair_v1/GRAPH_MANIFEST.json`; no graph is newly selected for this development comparison.

```sh
PY=/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python
$PY role_transfer_phase01/phase2_development_v1/experiment.py audit
$PY role_transfer_phase01/phase2_development_v1/run_all.py
$PY role_transfer_phase01/phase2_development_v1/analyze.py
```

The run completed all 160 models successfully. The last command verifies model, manifest, script, and protocol hashes and writes [SUMMARY.json](role_transfer_phase01/phase2_development_v1/SUMMARY.json). The scientific interpretation is in [RESULTS.md](role_transfer_phase01/phase2_development_v1/RESULTS.md). The run has eight graph families, sham/high conditions, seeds 321001 and 321002, five arms, 6,000 updates, batch 128, learning rate 0.0008, and checkpoints at steps 1,500/3,000/4,500/6,000. Scene seeds 260107 and 260108 are fixed. The primary score is withheld-focal-pair binary actor/patient accuracy. The 960-step preflight artifacts, including two failed role-swap weights, are retained separately and never enter the final development analysis.

After the main analysis, run `$PY role_transfer_phase01/phase2_development_v1/full_pair_audit.py` for a separately declared diagnostic on all 45 pairs at every final checkpoint. All 160 audit files were completed and checked against their saved model hashes. Its scope and source hash are in [FULL_PAIR_AUDIT_PROTOCOL.json](role_transfer_phase01/phase2_development_v1/FULL_PAIR_AUDIT_PROTOCOL.json). It does not change the primary endpoint or performance-matching selection.

`run_all.py` skips only directories with a completed `metrics.json`. A partial model resumes from its latest `training_state.pt`; the same graph, seed, arm, data schedule, and optimizer state are retained. Inspect `RUN_LEDGER.jsonl` and `logs/` for technical failures. Never replace a run because its score is unfavorable. The output `SUMMARY.json` is development evidence, not a sealed confirmation result.

## Research chronology

The 10-entity overlap and rewire studies preceded the two 12-entity factorials. Both 12-entity factorials missed their frozen success criteria. The current Phase 2 development reuses the previously examined 10-entity rewire families; any later independent confirmation must use genuinely new graph families and a separately frozen protocol. The pretrained-LLM question-wording and authentic-language experiments are separate from this graph-interaction analysis.

## Completed ability-matching development

The screen is documented in [ability_match_v1/README.md](role_transfer_phase01/ability_match_v1/README.md), frozen [PROTOCOL.json](role_transfer_phase01/ability_match_v1/PROTOCOL.json), and [RESULTS.md](role_transfer_phase01/ability_match_v1/RESULTS.md). It calibrated four nonzero ordered-role loss weights using nonfocal scores only, then ran the selected weight on all eight examined families. Both the calibration and full aggregate ability gates passed before focal evaluation. The focal result did not support ability-independent attenuation. All 32 calibration models, 24 additional full-stage models, checkpoints, hashes, and run ledger are retained. The full runner was resumed after an execution-session interruption; only unfinished configurations were retried, and the original interrupted start records remain in the ledger.

From `role_transfer_phase01/ability_match_v1`, using the working Python environment:

```sh
python preflight.py
python run_all.py calibration
python select_weight.py calibration
python run_all.py full
python select_weight.py full
python focal_eval.py
```

The selector reads only nonfocal ability until the gate passes. `CALIBRATION.json`, `ABILITY_GATE.json`, and `FOCAL_DEVELOPMENT.json` record each decision. All selected-weight 6,000-step model hashes and frozen script/protocol hashes were checked after training. This is development on already examined graphs, not independent confirmation or a pretrained-LLM result.

## Candidate-free natural role-binding development

See [protocol](research/candidate_free_role_binding_2026-10-14/PROTOCOL.json), [technical amendment](research/candidate_free_role_binding_2026-10-14/AMENDMENT_001.md), and [results](research/candidate_free_role_binding_2026-10-14/RESULTS.md). Four matched Qwen2.5-0.5B adapter continuations are complete. From the project root, run `$PY research/candidate_free_role_binding_2026-10-14/analyze.py` to recheck protocol, source, split, checkpoint, and evaluation hashes and regenerate `ANALYSIS.json`. `run_all.py` resumes only missing configurations; it refuses to retrain completed models. This is a QA-SRL Bank development set, not independent human-reviewed confirmation.

## 14 October research-to-paper audit

Read `research/paper_audit_2026-10-14/ACL_RESEARCH_STRATEGY.md`. Reproduce cached development diagnostics with `python3 research/paper_audit_2026-10-14/audit_cached_outputs.py` and `python3 research/paper_audit_2026-10-14/audit_common_pairs.py`. Neither command performs inference or reads sealed evaluation sets. `REPORT_INDEX.json` records current historical report hashes, not independent verification of every reported experiment.