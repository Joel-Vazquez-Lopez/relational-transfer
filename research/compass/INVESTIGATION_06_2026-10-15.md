# Compass Investigation 06 — Accurate caches can be unsafe to reuse
**Date:** 15 October 2026
**Status:** exact finite-universe combinatorial result and synthetic learned-model calibration. No pretrained LLM claim.

## Main result
For four independent uniform binary event directions (16 contexts, four question choices, 64 complete context–question pairs), enumerate **all 65,536 possible one-bit encoders** that create a question-independent cached message. With an accuracy-optimal per-question decoder:
- Best achievable cache-only answer correctness: **44/64**.
- Exactly **992** encoders attain that optimum.
- All 992 provide **zero guaranteed-correct cache-only questions**, where a query is guaranteed-correct only if every world with the same cached message has the identical value for that queried relation.
- Encoding one event bit exactly achieves **40/64** accuracy but **16/64** guaranteed-correct queries.
- Among encoders providing at least 16 guaranteed queries, the maximum possible accuracy is 40/64.

This is an **accuracy vs zero-risk reuse tradeoff** under a *hard one-bit information constraint*. It is **not evidence of an LLM's memory capacity or naturally chosen route**.

## Trained-model sanity checks
A learned straight-through binary cache with a simple MLP attained 44/64 and zero safe queries across three seeds (19,29,41). A small causal self-attention prefix encoder with an engineered hard one-bit cache showed 44/64 and zero safe questions at seeds 19 and 41, while seed 29 reached 40/64 and 16 safe queries. With two cache bits, three completed causal attention runs each reached 52/64 and 16 safe queries. A partial three-bit sweep was interrupted by execution time; one completed three-bit run achieved 53/64 and 40 safe queries. **Training and evaluation were on the same finite universe**, with accuracy-based checkpoint selection. Not a holdout result.

## Methodological implication
Maximizing cached-answer accuracy can favor compressions that are never sufficient to *guarantee* a particular answer. More conservative cached information can support safe, selective reuse at the cost of aggregate accuracy. When a question-time route to raw evidence is available, the best compute-saving policy need not follow the accuracy-optimal cache.

The task-specific guarantee uses exhaustive equivalence classes of world-to-message encodings; **a pretrained model will not have oracle access to this certification**, and logits alone do not produce such guarantees.

## Theoretical extension
For N independent uniform binary event facts and a B-bit pre-question cache Z, H(X|Z)≥N−B. If each ambiguous query needs direct access to its source fact for a certain answer, the expected number of such accesses per N uniform queries is ≥ N−B (and B/N cache-only fraction at most), with coordinate storage attaining the bound. This is a standard information-capacity consequence, **not a novel theorem**. Continuous high-dimensional states do not satisfy a B-bit restriction without a justified channel model.

## Next falsifiable experiment
Train a common-backbone two-route model with a question-independent capacity-limited cache and an optional question-time source-reader, making the gate depend only on cache and query. Compare accuracy-only, selective-risk, and compute-cost objectives on *held-out structurally varied relations*. Measure actual fallback accuracy and invocation cost. Preregister the gate and negative controls before larger models.

## Reproducibility
The exact proof is committed as `research/compass/exp06/capacity_certificate.py` in this branch. Full runnable PyTorch source, checkpoints, per-world raw predictions, unfinished-run information, and SHA-256 manifest reside in the user-provided artifact `COMPASS_INVESTIGATION_06_RELATIONAL_CACHE_CAPACITY_2026-10-15.zip`. The package itself was not uploaded to GitHub. Files: `capacity_test.py`, `transformer_bottleneck.py`, `audit_capacity.py`, `CERTIFIED_RESULTS.json`, `RESEARCH_REPORT_2026-10-15.md`.

No external model API used. No changes to main, other investigator branches, frozen sources, or sealed evaluation material.

## Novelty
The underlying issue is close to classical source coding/rate-distortion and selective prediction. Scientific originality would require novel neural evidence and a mechanistic causal explanation beyond this finite example. **Do not market this as a new theorem or established natural Transformer phenomenon.**
