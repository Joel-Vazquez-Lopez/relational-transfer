# Compass Investigation 16 — Intermediate-state supervision versus matched local-lookup control

**2026-10-16 · Day 2 · exploratory completed CPU PyTorch experiment.** **No natural pretrained-LLM reuse mechanism identified.** This work is confined to `research/compass-reuse`, and does not touch Astra, Nova, or other branches.

## Goal
Follow up Day 1's non-bijective graph result: training on short relational programs made outputs relatively insensitive to later operations. Test whether **correct intermediate-state supervision at successive causal operation-token states**, rather than more local auxiliary labels, improves generalization to unseen three-operation compositions.

## Frozen environment
Two arbitrary maps A,B on 4 entities, **repeated targets allowed**. Input includes 8 map values, a start entity, 3 operation positions with interleaved NOOPs, and a query position. Unrestricted **three-block, width-48 causal Transformer**, no hard-coded composition or gate. Train only 1- or 2-operation programs. Every position can contain A, B, or NOOP in training.

Disjoint split from fixed seed 1616, written in `frozen_protocol.json` before training:
- **9,000 training worlds:** A is not bijective.
- **96 in-family development worlds:** A is not bijective, no training overlap.
- **96 structurally shifted development worlds:** A **is** bijective, an entire graph family absent from training.
- Per development world: all 8 length-3 chains × 4 start identities = 32 dependent questions; also held-out shorter controls.

Three seeds (19/29/41), identical initialization/training-sample RNG by seed and arm, 400 AdamW steps, batch 128, LR .003, aux coefficient .5. **All three arms have exactly the same trainable architecture**:
1. `answer`: answer-only loss.
2. `state`: answer loss + correct accumulated relational state at each of three operation-token positions.
3. `source`: answer loss + same-position, same-size auxiliary cross-entropies predicting an **independent one-step lookup from the original start entity**. Comparable computationally, but **not difficulty matched** to accumulated-state labels.

## Held-out length-3 results
Each count out of **3,072** answers (96 worlds × 8 chains × 4 starts).

| Seed | Answer in-family | State in-family | Source in-family | Answer new family | State new family | Source new family |
|---|---:|---:|---:|---:|---:|---:|
| 19 | 1,655 | 1,424 | 1,530 | 1,288 | 1,132 | 1,172 |
| 29 | 1,433 | 1,552 | 1,581 | 970 | 1,198 | 1,229 |
| 41 | 1,573 | 1,561 | 1,583 | 1,187 | 1,212 | 1,190 |

Mean new-family accuracy: **37.38% answer**, **38.43% state**, **38.96% source**. Correct state targets average **+1.05 percentage points** over answer-only, versus **+1.58 points** for the local-lookup control. State effect varies by seed: **−5.08, +7.42, +0.81 pp** new family, **−7.52, +3.87, −0.39 pp** in-family. Per-world paired conditional bootstrap in artifact; **not independent population-level confidence intervals**.

**Complete structural generalization: 0/96 new-family worlds for every arm and every seed**, where a world requires all 8 triples × 4 starts correct. Short-task accuracy at 400 steps is not perfectly matched; training-competence gate therefore **has not passed**.

## What the supervised operation-token heads actually learned
On all **3,072** new-family three-step queries per seed, correct-prefix-state decode at operation positions 1/2/3:

| Seed | State head correctly decodes step 1 | Step 2 | Step 3 |
|---|---:|---:|---:|
| 19 | 3,060 | 2,774 | 1,204 |
| 29 | 3,072 | 3,066 | 1,323 |
| 41 | 3,072 | 3,060 | 1,198 |

So state supervision yields excellent **first-two-step** decodability, but only **39–43%** third-step gold decoding. The matched `source` arm's separate **single-hop** head accurately predicts its own local target at all three positions (seed 29: 3,068/3,072/3,072). **Probe decodability is not causal use by the native answer pathway** (cf. Compass Investigation 11).

## Counterfactual sensitivity — corrected design
The initial 192-edit cap selected in generator order and biased toward early worlds. The original logs remain in `results.json` but **should not be interpreted as a balanced study**. A separate frozen-checkpoint audit was run over **96/96 worlds, two eligible edits per world per step**, conditional on edits changing the gold final result, plus oracle-preserving irrelevant-cell edits.

| Arm | Seed | First operation output changed | Third operation output changed | Irrelevant edit changed |
|---|---|---:|---:|---:|
| Answer | 19 / 29 / 41 | 59.4 / 53.6 / 91.7% | 38.5 / 25.5 / 22.9% | 22.9 / 15.1 / 7.8% |
| State | 19 / 29 / 41 | 83.9 / 92.7 / 91.7% | 23.4 / 22.4 / 21.4% | 10.9 / 9.4 / 8.3% |
| Source | 19 / 29 / 41 | 67.7 / 85.9 / 88.5% | 39.6 / 27.6 / 25.0% | 16.1 / 10.4 / 9.9% |

The state arm **did not restore late-step dependence**. These are behavioral counterfactual responses to input changes, not neural causal cuts.

## Decision
**Negative development result.** We can train causally available, reliably decodable *early* intermediate-state outputs without obtaining systematic three-step whole-world composition or a reproducible advantage over an auxiliary local-lookup control. No claim that intermediate supervision is intrinsically useless: short-task training competence was not fully matched, and architecture depth, optimization and different aux-task difficulty remain material confounds. Only 3-operation input capacity was tested, **not the planned 4–5 steps**. No native computation has been identified.

Next discriminating experiment: training-budget/short-competence-matched depth comparison, then a preregistered native-path intervention **only if** held-out structural performance clears competence and shortcut gates. Preserve an explicitly engineered competent positive control, or stop pursuing the task if ordinary models remain unconvincing.

## Reproducibility
**Complete local artifact** `COMPASS_INVESTIGATION_16_INTERMEDIATE_STATES_2026-10-16.zip`, SHA-256 **`0bbe38442cdb1aa66aa1b595c8244c087e419666fc94f78f0ac0c6e974a1ac6a`**, approximately 3.0 MB, verified 32-member ZIP and per-file SHA256 manifest.

Archive includes **full executable source** `exp16.py`, `test_exp16.py` (6/6 pass), 9 checkpoint state dictionaries, all 55,296 held-out three-hop prediction rows, test oracle IDs/protocol, `trace_audit.py`, world-balanced `counterfactual_balanced.py`, `paired_audit.py`, frozen reports, unbalanced preliminary result and complete training stdout. The binary artifact is a conversation file **not contained in GitHub**. The report is reviewable here; obtaining the archive is needed for independent rerun. CPU PyTorch and no external inference required.

No sealed data, paid APIs, cross-branch modifications, or main merge. **Day 2 active; Day 1 complete.**
