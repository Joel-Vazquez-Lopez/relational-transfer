# Comput Report 007 — retrospective oracle leakage and transfer audit
Date: 15 Oct 2026. Input: Astra snapshot `research/astra_snapshot_2026-10-15/archive/research/llama_gemma_cross_family_2026-10-15/CROSS_ANALYSIS.json` on `codex/astra-research-2026-10-15`. No pretrained inference, no sealed data, no raw event-level predictions.

## Critical correction to Report 006
Astra's source `analyze_cross.py` uses `r['actor']` and `r['patient']` (gold participant assignments) to define several descriptive oracles. **always_actor, always_patient, event_subject_as_actor, event_subject_as_patient** are not certified raw-text, label-free baselines. In particular `event_subject_as_actor` uses a hard-coded answer-role mapping by construction rather than an independently audited syntactic parser. Previous headline maxima (e.g. 181/192 or 192/192 cellwise) must NOT be reported as viable deployable simple baselines; they are hindsight-fitted oracle envelopes.

We restrict retrospective transfer comparison to four participant-order response policies: always first, always last, first if actor-question else last, reverse. The input may still require externally known question role and ordered participants. The archived heuristic counts were themselves computed with reference `actor`/`patient` to assign first/last by known construction; independently extracting positions from raw sentence must be verified. This is a **restricted proxy**, not a verified end-to-end heuristic.

## Transfer summary (agreement with MODEL OUTPUTS, not answer accuracy)
| Model | In-sample best restricted single rule /192 | Leave WORDING out /192 | Leave CONSTRUCTION out /192 |
|---|---:|---:|---:|
| Llama | 123 | 99–123 | 99–123 |
| Llama no prefill | 120 | 108–120 | 72 |
| Phi-3 | 117 | 117 | 21–69 |
| Qwen 0.5B | 133 | 133 | 133 |
| Qwen 1.5B | 133 | 133 | 111 |
| Qwen 7B | 113 | 113 | 67 |
| Gemma | 96 | 96 | 0 |

Training chooses the fixed rule maximizing archived agreement on seven of eight cells when leaving one cell axis out, with deterministic tie-range reporting rather than peeking into the held-out cell. Each held-out construction includes two wordings; each held-out wording includes four constructions. Tie bounds are ranges over training-tied rules: there is no claim that an oracle picks among them after inspecting test data.

**Important:** All folds reuse the SAME 12 events and six name-pair clusters. This audits condition transfer, *not* lexical, document or participant-pair out-of-distribution generalization. It also does not prevent selection bias from the original investigative process or allow confidence intervals for a population.

## Falsification results
- Phi-3 restricted fixed-rule fit (117/192) is considerably weaker than the old 181/192 post-hoc cellwise oracle envelope, and its construction-held-out rule agreement drops to 21–69/192.
- Qwen0.5B's simple always-last policy stays at 133/192 across both axis-held-out evaluations. This is evidence of behavioral positional consistency on this repeated panel, not a neural pointer circuit.
- Gemma fully correct across the strong panel is incompatible with the four restricted fixed position rules when transfer is by construction (0/192). A parser with construction-specific syntax could do better, but would need a predeclared implementation with only permitted observed inputs. This is not evidence of a unique internal relational algorithm.
- Infer neither trained architecture nor causal mechanism from model-rule agreement.

## Reproduce
`python3 research/comput/rule_transfer_audit.py /path/to/CROSS_ANALYSIS.json`
`python3 -m unittest discover -s research/comput -p 'test_*.py'`

Next: collect original unsealed per-event output rows and sentence strings; implement independently extracted participant offsets and grammatical subjects, cross-validate by name-pair, predicate, and construction with proper separation; track all format failures. No sealed confirmation data touched.
