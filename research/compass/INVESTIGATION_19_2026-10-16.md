# Compass Investigation 19 — Correct update versus raw intervention sensitivity

**16 October 2026. Status: completed read-only, synthetic, post-hoc falsification audit. No native LLM mechanism demonstrated.**

## Scientific decision

**A larger response-change rate is not evidence of a correct relational computation.** The apparent instruction–operand asymmetry from Investigation 18 *reverses* when scored for semantically correct counterfactual updates. Across three frozen causal Transformers (seeds 19/29/41), both presentation layouts and the same 188 paired equal-gold counterfactuals per model/layout:

| Endpoint | Instruction edit | Source-operand edit |
|---|---:|---:|
| Changed the native prediction | **690/1,128 (61.2%)** | 302/1,128 (26.8%) |
| Returned the correct **new** gold | 270/1,128 (23.9%) | **457/1,128 (40.5%)** |

Only **201/1,128 (17.8%)** pairs returned the correct new gold after **both** edits. Among **445** original-correct questions, both interventions were correct on just **121/445 (27.2%)**.

Instruction-minus-operand *change rate* was positive in all six seed/layout conditions, but instruction-minus-operand *correct update rate* was **negative in all six**; seed 29 table-first's exploratory world-cluster confidence interval for the correct-update gap includes zero.

Pooled descriptive effect across models/layouts: response-change gap **+34.40 percentage points** (95% conditional source-world bootstrap **[+29.48,+39.24]**), whereas correct-update gap **−16.58 pp** (95% **[−21.23,−12.06]**). Resampling clusters the **95 underlying synthetic worlds** and keeps all repeated seeds/layouts attached. These are **not** population-level or preregistered intervals; they exclude model selection, post-hoc test choice and authentic language.

## New matched-value off-path negative control

Of the original 188 pairs, **91 pairs across 66 worlds** permit changing an **unused** B-table cell with the *identical old→new value substitution* used for the gold-relevant B-cell edit. Underlying A remains bijective and the original gold answer is preserved for the unused-cell edit. **Changed cell indices and token positions differ**; this is a token-substitution-matched, not fully position-matched, null control.

Across 3 seeds × 2 layouts × 91 paired items:
- Gold-relevant B-cell edit changed 166/546 (30.4%) predictions.
- Gold-irrelevant B-cell edit changed **104/546 (19.0%)** predictions.
- Paired difference **+11.36 pp**, conditional world-cluster bootstrap **[+2.65,+20.29] pp**.

There is some behavioral differentiation of relevant vs irrelevant source, but many irrelevant edits still perturb outputs. This is not evidence of correctly executed binding.

## Provenance / reproduction

All **1,128** original model outputs were independently regenerated from the exact **three unmodified Investigation 18 checkpoints** and compared **byte-for-value** with the archived paired predictions before adding the null edit. Four new unit tests pass; strict gold consistency and null invariance are verified.

**Standalone 22-member ZIP** in Compass conversation (no files copied to GitHub): `COMPASS_INVESTIGATION_19_CORRECT_UPDATES_2026-10-16.zip`. SHA-256 `89dfb6d3f21837ba7bc0ec0da2cb76212d6fe76b67c366fd352ff61bf73081c5`. Includes frozen `reference18/` source and all three weights, full PyTorch replay code `exp19_correctness_audit.py`, raw results `results/paired_predictions.jsonl`, `cluster_audit.py`, 4 tests, README, detailed report, SHA256 manifest and verifier. Archive CRC and 21 per-file SHA hashes verified.

The stdlib-only `audit_raw.py` (also stored in this branch) rechecks the main exact counts and source-world bootstrap from `results/paired_predictions.jsonl` after extraction. GitHub alone does not contain the full raw predictions/weights, so no independent model replay can be claimed from the GitHub checkout alone.

## Relation to other teams and next action

This is **Compass-only** reanalysis; Astra's Qwen output-field-order study and Mira's authentic EWT intake remain separate. No other branches, main or sealed data changed; no paid inference.

**Stop rule:** Do not train additional synthetic models to interpret this sensitivity asymmetry as a native relation-binding mechanism. Prioritize independent review of correct-update scoring and off-path null, then a rights-cleared, human-verified authentic multi-event panel and competent pretrained model controls, unless a new matched-competent architecture offers a decisive native-path falsifier.

**Conclusion:** The evidence supports **inconsistent counterfactual updating**, not an identified representation or reusable relational algorithm.