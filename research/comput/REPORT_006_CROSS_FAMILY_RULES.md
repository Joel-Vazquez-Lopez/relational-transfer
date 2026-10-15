# Comput Report 006 — rule agreements on actual historical cross-family summary
**Date:** 2026-10-15; **scope:** independent recomputation from Astra's archived `CROSS_ANALYSIS.json` (original per-answer predictions *not* hosted). Input: `codex/astra-research-2026-10-15:research/astra_snapshot_2026-10-15/archive/research/llama_gemma_cross_family_2026-10-15/CROSS_ANALYSIS.json`.

The archive contains **model-to-rule agreement counts** for eight strong free-answer construction/wording cells. Comput recomputed model-specific best fixed rule across the 192 answers, along with a **post hoc cell-wise best-rule envelope** (not a deployable predictor; uses the evaluation answers to select rules). This is agreement to the **model**, NOT semantic-role correctness.

| Model | Best single retrospective rule | Cellwise retrospective rule envelope | 192 answer observations |
|---|---:|---:|---|
| llama | 127 | 181 | 192 |
| llama_no_prefill | 124 | 178 | 192 |
| phi3 | 123 | 181 | 192 |
| qwen05 | 138 | 147 | 192 |
| qwen15 | 133 | 159 | 192 |
| qwen7b | 136 | 177 | 192 |
| gemma | 144 | 192 | 192 |

Interpret with extreme care: 192 observations reuse 12 events/six name-pair groups in 8 conditions. The cellwise maxima are hindsight-fitted and **cannot** establish predictive generalization; high in-sample agreement can arise from memorizing tiny cells. `event_subject_as_actor` is a grammatical-subject heuristic only if the subject can genuinely be extracted from the presented sentence without privileged semantic labels; this feature audit remains open. Gemma's cellwise envelope of 192 merely reflects the presence of a rule agreeing on each evaluation cell; it proves no internal mechanism or algorithmic equivalence. No pretrained model was run in this Comput audit.

## Reproduction
Download historical CROSS_ANALYSIS.json from its research branch to a local **unsealed** path and execute:
`python3 research/comput/cross_family_rule_audit.py /path/to/CROSS_ANALYSIS.json`
`python3 -m unittest discover -s research/comput -p 'test_*.py'`

## Falsification question
Can a *pre-registered*, feature-valid and low-complexity mixture of grammatical-subject, first/last, question, and construction rules match held-out raw predictions? To answer, use raw event-level outputs, leave-name-pair-and-predicate-out folds, and ablate metadata that is not observable in raw text. Do not select different winning rules on held-out cells. Preserve output-format failures and all negative cases.
