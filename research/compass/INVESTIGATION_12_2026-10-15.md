# Compass Investigation 12 — Native-path causal boundary controls
**Date:** 2026-10-15 · **Status:** executed synthetic negative result. No pretrained-model mechanism identified.

## Aim
Test whether an unrestricted causal Transformer's hidden-state interventions distinguish naturally reused pre-question relational structure from question-time reconstruction **without hardcoding a cache/reconstruction route**.

## Method
Identical causal Transformers (3 causal-attention blocks, width 48, **76,708 parameters**), independently trained at seeds 19,29,41 to **144/144 familiar-condition answers** on three tasks: (A) composition of two swaps over four occupants, (B) copying a queried input slot, (C) mod-4 sum of two queried input slots. None has an explicit cache gate, external probe, or prebuilt relational readout.

For each target/donor pair with same query and different gold labels, replace **either** the four prefix hidden states **or** the final question-position hidden state at boundaries after blocks 1 or 2. Resume native forward pass. Also test same-state, full-state donor and third-answer controls. All model weights and raw patch counts preserved.

## Exact pooled results
| Task | Boundary | Informative donor pairs | Prefix patch: donor answer | Question patch: donor answer | Prefix patch: neither target nor donor |
|---|---|---:|---:|---:|---:|
| Two-swap relation | After block 1 | 11,520 | 6,466 | 2,544 | 2,510 |
| Two-swap relation | After block 2 | 11,520 | 2,450 | 6,612 | 2,458 |
| Plain copying | After block 1 | 9,504 | 8,536 | 966 | 2 |
| Plain copying | After block 2 | 9,504 | 675 | 8,829 | 0 |
| Modular addition | After block 1 | 11,232 | 6,685 | 1,887 | 2,660 |
| Modular addition | After block 2 | 11,232 | 666 | 9,833 | 733 |

Every same-state splice was a no-op; every full-state donor splice reproduced donor output. Pairs are exhaustive *dependent* combinations from the training universe; not independent examples.

**Falsifications:**
1. Early-prefix → late-question causal influence is also seen in plain copying and modular addition. This is not a relation-specific mechanism signature.
2. Hybrid patches giving a **third** answer are also common in nonrelational modular addition (2,660/11,232 after block 1). This is not sufficient evidence for an abstract relational transform and may reflect off-manifold patch states.

## Actual model-heldout compositionality check
Fixed random.Random(555) split: train on 27/36 distinct ordered-swap contexts (108 questions), hold out 9 contexts (36 questions). Train **same unrestricted architecture for 400 steps**, three seeds:

| Seed | Train | Held-out |
|---|---:|---:|
| 19 | 108/108 | 12/36 |
| 29 | 108/108 | 10/36 |
| 41 | 108/108 | 14/36 |

Held-out labels are perfectly balanced; a constant-answer baseline is 9/36. Scores are weak and the same 36 test questions are reused across seeds. A longer 1800-step seed-19 attempt showed 12/36, but seeds 29/41 were interrupted before completion; retain it as partial evidence only.

## Mechanistic conclusion
The interventions demonstrate **native causal pathway influence**, not a uniquely identifiable cached relational algorithm. Neither activation-transport shift nor third-answer mixing distinguishes relational composition from generic attention-based computation. Memorization remains viable; no pretrained model was tested. Do not advertise a new relational circuit, systematicity, or ACL-level novelty.

## Reproduction and integrity
Full executable sources, all nine saved checkpoints, per-task intervention JSON, fixed OOD split/metrics, negative-control results, incomplete-run log, stdlib verifier and SHA-256 manifest are archived as:
`COMPASS_INVESTIGATION_12_NATIVE_CAUSAL_BOUNDARY_2026-10-15.zip`
SHA-256: `5a529bf9a316fcbb6e57763f9cdc9f90358ce4a1c7383662fe82270b327f1676`.
Verified ZIP member hashes. Archive was produced in the Compass conversation and is **not included in GitHub**. Read its `RESEARCH_REPORT.md` and run `python verify_results.py` in the extracted folder to reproduce exact aggregates. CPU PyTorch is required for model reruns. GitHub's report itself should not be mistaken for a code checkout.

No sealed data used, no other branches touched, no main merge. Expense: local CPU only.

## Next evidence gate
Do not promote a native-circuit reuse claim until a common architecture shows strong held-out relational generalization and a *preregistered blinded assay* can reject known reconstruction models and matched nonrelational tasks. The causal identity of role-resolved information must be demonstrated, not merely its presence or propagation.
