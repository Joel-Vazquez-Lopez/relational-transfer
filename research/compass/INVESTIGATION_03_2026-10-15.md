# Compass Investigation 03 — structural algebra and causal identifiability

**Date:** 15 October 2026 · **Status:** exploratory falsification, not a new pretrained-model discovery.

## What we learned

1. **Previous generated challenge is algebraically compressed.** All 12,288 labels in `repeated_verb_graph_development.jsonl` are determined by two generator edit-switch bits via XOR, conditional on question type and name family (4,096/4,096 per family). **Caveat:** those bits are generator metadata, not observed sentence features; a language model may still need relational parsing to extract them. This is not a viable raw-text shortcut without an extractor.
2. **New commuting/noncommuting control.** In 256 synthetic development questions (8 name families, 2 operation types, 2 temporal orders, 2 presentation orders, 4 rooms), swapping the timeline of overlapping exchanges changes the gold answer in 96/128 checks, while independent exchanges change 0/128. All 256 presentation-order invariance and word-bag checks passed. A schedule-blind oracle cannot exceed 208/256 (81.25%), versus 256/256 for an ordered composition interpreter.
3. **Two explicit mechanisms are behaviorally indistinguishable.** Transparent reuse/cache and question-time reconstruction programs both score 256/256 and follow a whole-prefix donor in all 256/256 counterfactuals. Their actual computation differs: 64 unique context compositions before questions versus 256 after questions.
4. **Oracle causal cuts discriminate them:** when query-time raw operations are inaccessible, reuse remains at 256/256 and reconstruction yields no defined response; when the cached result is inaccessible, the reverse occurs. These cuts presuppose knowing where the information lives; this is exactly what an opaque neural model does not grant us.
5. **A real causal Transformer provides another negative control.** Three training seeds (19/29/41) for a 40,236-parameter, two-layer active-only synthetic Transformer achieved 1,024/1,024 answers correct, and 1,024/1,024 name-swap counterfactuals correct. Resetting contextualized *prefix* states before its final block to raw token/position embeddings preserved 1,024/1,024 accuracy for all seeds. This intervention leaves earlier query states intact, and cannot rule out native or redundant precomputation. The active-only grammar has a position shortcut; it does not demonstrate abstract role understanding.
6. **Negative result preserved.** A mixed active/passive, two-event-query training attempt failed to attain high competence in the local budget (~50% near 1,500 steps; job timed out). No internal inference is made from it.

## Mathematical point

Structural competence and native pre-question reuse are different axes. Correctly tracking noncommuting state transitions **does not** distinguish reuse from recomputation. Whole-prefix patching does not settle it either. A rigorous method must infer the *natural causal route* and reject matched learned reconstruction controls.

## Novelty boundary

[Li, Guo and Andreas, ICML 2025](https://proceedings.mlr.press/v267/li25r.html) already studied permutation-composition state tracking, associative-scan mechanisms and parity heuristics. [Sutter et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dbb98528c9870377f3f0d133aae6050b-Abstract-Conference.html) showed unconstrained nonlinear alignment maps can falsely certify causal abstraction. **Neither permutation composition nor an arbitrary successful causal patch is a new finding here.**

## Next decision gate

Build *matched-competence learned causal Transformer controls* with enforced pre-question versus post-question computation, then preregister a hidden-state discrimination method (and require that it does not mistake general prefix information for composed relations). Do not expend substantial pretrained-model compute until that falsification gate passes.

## Reproduction and files

The full working package was created and verified in the Compass conversation: `COMPASS_INVESTIGATION_03_ALGEBRA_AND_IDENTIFIABILITY_2026-10-15.zip` (SHA-256 `7d79d7e7894dafffe8c33a77808aed8b8295726ef4a6e452cc4352a691263df8`). It contains Python source, 256 development questions, raw-derived audits, trained-model result JSONs and checkpoints, an incomplete training log, and a file hash manifest. The archive is **not** in the GitHub repo and should not be assumed retrievable from GitHub.

**Research integrity:** No frozen historical artifact altered, no sealed confirmation data used, no merge to main, no pretrained mechanistic claim.
