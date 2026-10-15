# Compass Investigation 13 — Partial length generalization, failed systematicity

**Date:** 15 October 2026 · **Scope:** research/compass-reuse.
**Status:** executed exploratory CPU-PyTorch experiment with 9 models; not a pretrained-LLM causal discovery.

## Design
Plain 3-layer, width-48 causal Transformer, no engineered permutation operator or precomputed-world head. Four entities, six transposition primitives, three operation slots. Train on 0–2 nonempty swap slots, test **all 216 unseen ordered 3-swap sequences × four room questions = 864 answers**. Of the 216 triple contexts, 96 produce genuine **4-cycles (384 questions)**—final world permutations not appearing anywhere in the zero-, one-, or two-swap training labels. Remaining 120 triples are transpositions.

Three arms, all trained with AdamW, lr .003, batch 128, 1,200 steps, seeds 19/29/41:
- Pair-only (432 labels, 108 distinct two-swap contexts).
- Single + pairs (504 labels).
- All shorter, including identity (508 labels).

All nine checkpoints reach **100% training accuracy**. The same 216 triple contexts are used in all runs.

## Measured generalization

| Seed | Pair-only correct /864 | Single+pairs | All shorter |
|---|---:|---:|---:|
| 19 | 514 | 537 | 553 |
| 29 | 470 | 513 | 523 |
| 41 | 467 | 508 | 519 |

Pooled descriptive totals (dependent observations across seeds): **1451/2592** pair-only, **1558/2592** single+pairs, **1595/2592** all-short (61.5%). Improvement is present in all three paired seeds, but arm comparisons confound examples seen with objective/curriculum; no population-level significance established.

**Critical heuristic:** Skip each of three swaps in turn, compute three two-swap answers, then take deterministic majority vote. This non-systematic partial-composition baseline achieves **504/864 (58.3%)** on triples and **264/384 (68.75%)** on unseen four-cycles. The **all-short models average 774/1152 (67.19%) on four-cycle queries**: slightly **worse** than the baseline. Thus token-wise generalization above chance **does not demonstrate learning the full relational operation**.

## Stronger global metrics
Four answers must be correct for a complete relational world. Under all-short training, models solve **19/216, 15/216, 18/216** entire three-swap worlds (seeds 19/29/41) and **15/96, 13/96, 15/96** of previously unseen four-cycle worlds. The partial-vote baseline solves **0/216 complete worlds**; hence there is modest evidence of a few jointly coherent outputs, but not general systematicity.

Full entity-relabeling equivariance on all 23 nonidentity permutations: **69.0%, 68.4%, 64.9%** for the all-short models, well short of exact covariance of a permutation-composing algorithm.

Seed-19 all-short 2,000-step sensitivity run: **556/864** versus **553/864** at 1,200 steps; four-cycle accuracy unchanged at **269/384**. Not evidence that further optimization in general cannot improve performance.

## Scientific decision
**Gate not passed.** We have partial transfer and a useful curriculum effect on this restricted grammar, but systematic relational computation is not demonstrated. Before a causal reuse-vs-reconstruction attribution, require high held-out **whole-world** consistency, strong relabeling equivariance, and wins over partial-composition heuristics across many seeds, constructions, and independent entity families. Do not infer a latent relational circuit from prefix/query patching alone.

## Literature/novelty boundary
Li, Guo & Andreas, [(How) Do Language Models Track State?](https://proceedings.mlr.press/v267/li25r.html), ICML 2025, have already studied permutation composition and associative/parity-based mechanisms. Ramesh et al., [Compositional Capabilities of Autoregressive Transformers](https://proceedings.mlr.press/v235/ramesh24a.html), ICML 2024, examined related easy-to-hard composition. **No claim of novelty or natural pre-question reuse.**

## Artifacts and integrity
Complete runnable release is in the Compass conversation attachment `COMPASS_INVESTIGATION_13_COMPOSITION_EMERGENCE_2026-10-15.zip` (SHA-256: `3a9954b652f7569d2c65df93557e59a27ebc346f296ccb3670a4cc6bb313b5f5`). It contains nine primary frozen checkpoints, one 2,000-step sensitivity checkpoint, **13,720** model × example raw prediction records (10 × 1,372), scripts, exact run logs, unit tests, baseline audit, equivariance audit, verified file manifest, and detailed report. Main executable source also committed to `research/compass/exp13/exp13_curriculum.py` in this branch.

A single combined batch timed out after two completed models; all remaining configurations were run individually with the same fixed settings and independently evaluated. No sealed data, Nova/Astra files, restricted inputs or external LLM inference used. No merge into main.

**Next Compass hypothesis:** the gap between query-wise accuracy and *entire-event correctness* may be a stronger developmental discriminator than marginal accuracy, but it needs hard shortcut controls and genuinely variable relational structures before any general claim.
