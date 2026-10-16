# Brain-O Day 2 — E043/E044: separated query-factor shortcuts
**Date:** 2026-10-16  
**Status:** Executed, replayed synthetic development study. No pretrained LM or authentic-language claim.

## E043 — Finish previously interrupted E041 ordinary-Transformer baseline
On E040's three-event collision grammar `(v0,t0), (v0,t1), (v1,t0)`, trained **unchanged 3-layer width64 causal Transformer, 500 AdamW steps, batch128, +3-way event-index auxiliary** for two seeds. The previous E041 budget did not finish; **E043 completes the intended 500-step budget** and saves both model state dictionaries, raw training history and exact 200-world held-out evaluation.

| Seed | Answer QA (out of 1,200) | Event index QA (out of 1,200) | Entire correct six-question worlds (out of 200) |
|---|---:|---:|---:|
| 1040 | 376 (31.33%) | 802 (66.83%) | 0 |
| 1041 | 389 (32.42%) | 804 (67.00%) | 0 |

The explicit learned E040 structured copier previously scored 1,200/1,200 answers per seed. **This is not an ability-, parameter-, architecture-, or training-compute-matched comparison.** It is a bounded optimization screen on the same task, not evidence that unrestricted Transformers cannot solve it.

## E044 — Three-way, same-source query-key counterfactual test
Generate 600 **fresh worlds** per frozen E043 checkpoint (two roles/world, three valid queries/role). Identify the unique intersection record `(v0,t0)`. Holding source world and queried role fixed, edit only the query **verb** to `(v1,t0)` or only the query **context tag** to `(v0,t1)`; each change provably changes gold event and gold participant. Save every output event/answer prediction (600 worlds × 2 roles × 3 conditions = 3,600 rows/seed) and enforce invariants.

| Prediction change | Seed 1040 | Seed 1041 |
|---|---:|---:|
| Event index changes when **verb** changes | 2.33% | 2.00% |
| Event index changes when **tag** changes | **100.00%** | **100.00%** |
| Final answer changes when **verb** changes | 0.33% | 1.33% |
| Final answer changes when **tag** changes | 1.17% | 2.83% |
| Final answer changes when **role** changes (600 worlds, intersection query) | 581/600 | 593/600 |
| Event index changes when **role** changes | 0/600 | 0/600 |

Event-index accuracy on the *unique-tag* case is 100% both seeds; on repeated-tag cases it is approximately 47–54%. The E040 tag-only baseline scores 66.67% overall. This supports a **tag-marginal event-index shortcut** at its designated readout and a **role-marginal final-answer shortcut** at the terminal readout. It does not uniquely identify an internal circuit. Later training could produce conjunction competence.

### Mechanism / novelty boundary
Our observation is a *readout-specific separation of single-key dependencies*, consistent with token-position residual accessibility and optimization shortcuts. Earlier Brain-O E010–015 showed analogous query-order effects on a simpler two-event grammar. Astra's independent output-field-order × input-participant-position experiment is **not duplicated** here: no Qwen inference, no output-field-order changes, no adaptation. Compass I12/I15 and Comput audits warn generic lookup/length shortcuts are live alternatives.

A dual-readout system can be fairly good at two separate tasks while still failing to combine their outputs to answer a composed question. This is not proof that the required full binding exists; a **single-key** event-index classifier is not a successful compound address. **No validated new mechanism or ACL novelty.**

### Next decisive gate
1. Competence-matched *ordinary neural* lookup/copying controls at fixed compute and depth/width, or factor-specific training contrasts that attain ≥95% clean accuracy before mechanistic interpretation.
2. Independently audit artifact; report full-world and per-family metrics, not only marginal QA.
3. Translate to authentic, independently human-adjudicated repeated-predicate scope + informative context cues with ≥2 untouched pretrained model **families** after #23's source/rights/format gate; compare output field orders separately under Astra.

### Reproducibility
Conversation ZIP: `BRAIN_O_DAY2_E043_E044_2026-10-16.zip`, **SHA256 `ac6ea721a374ba66284f9dde7fe1429c8d28e95afa47bf368743f91afa4cae32`**. 30 entries including source E040/E041/E043/E044, both 3-layer frozen checkpoints, complete fixed-budget logs, per-question E044 predictions, all numeric JSONs, manifest, `verify_saved.py`. ZIP CRC PASS; saved model re-evaluation PASS; E044 counterfactual recomputation PASS from release directory. Binary archives remain in Brain-O conversation; not pushed to GitHub or automatically available across chat sessions. No sealed data or paid inference, no other branches modified.
