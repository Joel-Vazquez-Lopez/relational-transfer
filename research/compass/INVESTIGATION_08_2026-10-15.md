# Compass Investigation 08 — Jointly learned cache and reconstruction routes
**Date:** 2026-10-15 · **Status:** controlled negative synthetic result; no pretrained-model claim.

Single shared-backbone causal Transformer, prefix-only straight-through 1-bit cache, query-time raw-reader branch, and bit+question-conditioned gate. 16 binary worlds × 4 queries, all seen during principal model training. Three seeds (19,29,41), three reconstruction unit costs, 450 pretrain + 500 joint steps.

| Unit reconstruction cost | Exhaustive finite-world optimum | Trained seed 19 | Seed 29 | Seed 41 |
|---|---:|---:|---:|---:|
| 0.25 | 12.0 | 15.0 (4 errors, 44 fallbacks) | 15.0 (4,44) | 15.0 (0,60) |
| 0.40 | 18.4 | 20.0 (20 errors, 0 fallbacks) | 20.0 (20,0) | 20.0 (20,0) |
| 0.75 | 20.0 | 20.0 (20 errors, 0 fallbacks) | 20.0 (20,0) | 20.0 (20,0) |

Raw-reader route was error-free on all 64 evaluation questions for every principal run. Posthoc optimal gate **conditional on the learned cache code and its predictions** matched the learned gate objective in all runs: the cache representation, not only the gate, was suboptimal.

**Oracle-distilled capacity control:** same architecture with exact oracle code (index 831) and optimal code/query fallback labels supervised at cost .40 reached **18.4** in all three seeds (12 errors, 16 fallback calls, 64/64 correct oracle-code assignments). This is an existence control only, not spontaneous learning.

**Withheld-world stress test:** four held-out worlds (indices 3,6,9,12; 16 questions) at cost .40. All three seeds failed **8/16** held-out answers, invoked zero reconstructions, although raw-reader errors were **1/16, 0/16, 0/16** respectively. Small synthetic split; not broad OOD generalization.

## Interpretation
A representation can achieve an accuracy optimum relative to its local objective yet prevent a gate from learning globally beneficial conditional reuse. Learning the relational representation, its question-dependent selection, and the compute-allocation policy are separable. **The result does not show how pretrained LLMs naturally reuse relational computations.** This is a constrained rate-distortion/selective-prediction-style scenario, not a verified new principle or paper result.

## Next falsification
Ablate or change the discrete estimator, pretraining schedule, and routing regularizer under a fixed compute budget. Verify with truly withheld lexical/structural event families and compare against exact cost baselines. Do not presume that optimizer failure implies an inherent architecture limitation.

## Reproduction
Verified source, checkpoints, per-run JSON, withheld-world metrics, oracle-supervised controls, logs, report and SHA-256 manifest are attached in the user conversation as `COMPASS_INVESTIGATION_08_LEARNED_RELATIONAL_ROUTING_2026-10-15.zip`. Source: `shared_router.py`, `oracle_witness_control.py`, `withheld_worlds.py`. **The binary archive is not stored in GitHub**. No sealed data used, no merges or other branch changes.
