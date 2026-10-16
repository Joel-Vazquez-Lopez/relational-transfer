# Compass Investigation 17 — Depth, intermediate supervision and operand–instruction dissociation

**16 October 2026 — completed open-development synthetic experiment.** **No proven pretrained LLM mechanism, natural-language transfer, or causal relational cache.** No sealed data or paid inference.

## Coordination and independence
Work on \`research/compass-reuse\` only; no change to Astra, Nova or other research branches. Day 2 continuation of Investigation 16, where supervised early intermediate states failed to rescue length-three composition. Context checked against the latest open GitHub PRs, including Mira's authentic EWT pilot, without touching those workstreams.

## Protocol, fixed before model training
Identical width-48 causal Transformer architecture and short-chain training generator from Investigation 16, copied into the reproducible artifact as \`exp16_reference.py\`. Two arbitrary functions A,B on four entities (repeated targets permitted). Train on one or two active operations with NOOPs in any of three positions. Test every three-active-operation chain on disjoint worlds.

Fixed new development split seed **1717**: 9,000 training worlds with A nonbijective; 96 unseen in-family worlds; 96 unseen A-bijective-family worlds. **Within-world questions are dependent.** 600 AdamW updates, batch 128, LR .003, same initialization and sampling seeds 19/29/41. Factorial arms \`answer\` (no auxiliary loss), \`state\` (gold accumulated intermediate states), \`source\` (matched local one-hop lookup auxiliary targets) at 3 versus 5 causal-attention layers: **18 trained checkpoints**. Three-layer models have 82,232 parameters; five-layer have 129,464. Depth and parameter count are therefore confounded. Auxiliary target difficulty is not matched.

Training protocol and graph families were frozen before fitting in \`runs/frozen_protocol.json\`. **18/18 checkpoints independently reloaded and exactly replayed**; 3,456 checkpoint×world×panel rows of all gold and predicted answers preserved.

## Held-out A-bijective graph-family results
Each arm/depth pools 3 model seeds × 96 identical worlds × 8 programs × 4 starts = 9,216 **dependent** answer records:

| Depth | Arm | Correct /9,216 | Accuracy | Entire 32-answer worlds correct /288 |
|---|---|---:|---:|---:|
| 3 | Answer | 3,388 | 36.76% | 4 |
| 3 | State | 3,588 | 38.93% | 5 |
| 3 | Source | 3,636 | 39.45% | 5 |
| 5 | Answer | 3,478 | 37.74% | 3 |
| 5 | State | 3,361 | 36.47% | 5 |
| 5 | Source | 3,522 | 38.22% | 3 |

A simple **last-operation-only** deterministic shortcut gets **1,413/3,072 = 46.0%** on the same A-bijective panel, above every trained arm mean. A complete symbolic interpreter gets 100%. **13/18** checkpoints achieve ≥99% held-out one-hop and two-hop accuracy; the remaining five are partially undertrained. Within this fixed-update study, five layers **do not reliably rescue** three-hop world-level composition. This does not imply Transformers cannot learn it with other training or recurrence.

## Exploratory post hoc matched operator–operand intervention
Initially, whole-family single-token flips and source-cell edits suggested that some models respond strongly to changing an operation token while responding weakly to a consequential source-table edit. The first equal-gold matched implementation \`matched_final_step.py\` could also move a world out of the A-bijective family when changing A; **preserved as a confounded pilot**.

Corrected design \`matched_family_preserving.py\` compares **the same source world, start and program**, with two different edits having exactly the **same changed gold answer**:
1. Flip the *last* operation token **B→A**; or
2. Retain the final B token but change the **B-table cell actually consulted at the final step** to yield exactly that same alternative gold answer.

The last B cell must not have been used in earlier steps, and A stays bijective before and after both edits. Precomputed a fixed two-example-per-eligible-world set: **188 paired examples from 95/96 held-out worlds**. The remaining world has no eligible contrast. Both counterfactuals remain within the family and require identical correct answers.

**All 18 checkpoints change their answers more often under the final *instruction* edit than the *referenced source-value* edit.** Illustrative five-layer local-lookup model:

| Seed | B→A instruction changes output | Matched B-cell edit changes output | Difference |
|---|---:|---:|---:|
| 19 | 50/188 | 0/188 | +26.60 pp |
| 29 | 145/188 | 33/188 | +59.57 pp |
| 41 | 147/188 | 37/188 | +58.51 pp |

For seeds 29 and 41, the matched instruction-edited answer is **correct only 33/188 and 32/188 times**, respectively; more answer *changes* do **not** mean successful relational composition. The same-target B-cell edit gives 77/188 and 78/188 answer matches, many potentially coincidental.

The per-world paired differences have exploratory, conditional 95% world-bootstrap intervals: seed 29 **[+51.60,+67.38] pp**, seed 41 **[+49.73,+66.49] pp**. These do not account for adaptive selection of tests, worlds or models. This is a behavioral counterfactual, **not an internal activation intervention**. The two edits also occur at different token positions and may induce different generic attention/position effects; source availability versus recency remains an unexcluded explanation.

**Conservative hypothesis:** these models sometimes register which instruction was requested while failing to apply it faithfully to the necessary relational operand and intermediate state. This is *not* proof of any specific hidden variable, binding code, cache, or native computational schedule.

## Artifact, integrity and reproducibility
Complete verified ZIP in Compass conversation: \`COMPASS_INVESTIGATION_17_DEPTH_OPERAND_BINDING_2026-10-16.zip\`, **SHA-256 \`ef27ac588628ec34d535f64503e4112e332cdda2bee52f541fbdbb2dc221e74a\`**. 58 ZIP members (57 individually SHA256-authenticated files plus manifest), CRC check passed. Includes the full executable PyTorch code, all 18 frozen state dictionaries, train/eval logs, five passing source tests, exact replay, raw world-level predictions, heuristic and causal-negative controls, initially confounded and corrected same-gold intervention, paired statistics and per-world counts. Full source archive **not hosted on GitHub**; GitHub holds report and selected executable controls.

## Scientific decision
**Depth rescue: negative at fixed budget. Operand-vs-instruction behavioral asymmetry: supported in this synthetic exploratory setting. Native algorithm identification: not achieved.** Before interpreting this as a novel relation-binding failure, prospectively replicate with positional/distance controls on multiple non-template graph and language families, and on a model that actually demonstrates held-out **whole-world** compositional competence. Then test the native answer pathway with known-reconstruction adversarial controls.

Adjacent work includes Tang, Lake & Jazayeri (PLOS ONE, February 2026) on mechanistic function composition, Chen (arXiv:2603.21676) on depth-recurrent composition and Liang et al. (arXiv:2609.33144) on length-generalization mechanisms. Novelty and authentic language generalization are **not established**.

**No other branches altered. No main merge.**
