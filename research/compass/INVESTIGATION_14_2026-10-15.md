# Compass Investigation 14 — Whole-world structure versus imposed decoding

**2026-10-15 · Completed frozen-checkpoint synthetic audit · No causal reuse finding.**

## Updates read before execution
- Main `research/WORKSTREAMS.md` reiterates investigator independence and provenance rules.
- Coordinator draft PR #15: crossed explicit cache/raw-port interventions can identify **known ports**, not native neural algorithms.
- Coordinator draft PR #17: a competent source-reader shortcut yields the **same valid-world intervention signature** as reconstruction, so port-based identity is insufficient.
- Protocol auditor draft PR #20: original same-query stimuli admitted a 75% alphabetical shortcut; corrected v2 eliminates that simple shortcut but is still 100% solvable by question-time surface parsing.
- Brain-O synthetic two-key event/role positive controls are related, but not evidence of pre-question caching.

These updates were read as **reported external workstream results**, not independently re-executed.

## New test
Read **all nine unmodified Compass 13 causal-Transformer checkpoints**, 3 curricula × 3 seeds, each tested on the same **216 unseen three-swap contexts × 4 room questions**, including **96 four-cycle** worlds. The original four independent answer logits are audited three ways:

1. **Native independent:** choose the highest logit per room.
2. **Externally constrained MAP:** among all **24 permutations** of four occupants, maximize the **sum of per-room logits**. Uses externally supplied one-to-one occupancy law, **not** gold labels or three-swap parity.
3. **Hard-answer-only negative control:** discard all sub-top1 logits, choose uniformly among permutations minimally changing the four original top answers; compute exact expected accuracy over tie resolutions.

An explicitly **privileged** secondary oracle restricts permutations to odd parity, a mathematical fact of three swaps; do not interpret as the model knowing parity.

## Results (all-short curriculum)
| Seed | Independent whole worlds /216 | Hard-answer repair expected /216 | Full-logit MAP /216 | Parity-oracle /216 | Independent individual answers /864 | MAP individual answers /864 |
|---|---:|---:|---:|---:|---:|---:|
| 19 | 19 | 66.2 | 77 | 148 | 553 | 572 |
| 29 | 15 | 50.9 | 53 | 115 | 523 | 514 |
| 41 | 18 | 52.8 | 50 | 112 | 519 | 500 |
| **Pooled descriptive** | **52/648** | **169.9 expected /648** | **180/648** | **375/648** | **1595/2592** | **1586/2592** |

All-short *four-cycle* whole-world success changes **43/288 → 134/288** with constrained MAP. The hard-output-only repair already yields **107.25 expected /288**.

Other curricula (3 seeds each):
- **Pairs only:** independent 29/648 complete worlds; hard-only 115.4 expected; MAP 118/648.
- **Singles+pairs:** independent 40/648; hard-only 153.8 expected; MAP 143/648.
- **All-short:** independent 52/648; hard-only 169.9 expected; MAP 180/648.

These are the same worlds repeatedly evaluated across seeds, **not 648 independent contexts**.

## Interpretation
- Imposing a valid permutation externally increases complete-world correctness markedly while **slightly lowering** overall per-question accuracy in all-short models (1595→1586). Formally coherent answer sets can still be wrong.
- **Most** of the gain is reproduced by a constrained repair using only the model's top answers. The full-logit advantage is **small and inconsistent** across seeds and curricula.
- More query-time forward passes and a hand-specified bijection are used at evaluation. Nothing shows that the native Transformer formed or reused a complete relational state.
- The odd-parity oracle achieves much larger scores but carries privileged grammar structure. Treat it as a **leakage stress-test**, not a fair headline metric.
- No trained mechanism is identified; no claim about pretrained LLMs or natural language, and no general novel algebra.

## Reproducibility
Verified archive `COMPASS_INVESTIGATION_14_WHOLE_WORLD_COHERENCE_2026-10-15.zip` is attached to the Compass research conversation, **not stored on GitHub**; archive SHA-256 `2595bb6b9cb5aef4cb77b42ec87da5f76f6577e7b111b8794d9986e472ce8664`. Contains the independent PyTorch forward runner, four unit tests (all passed), saved logits for **9 × 216 × 4** evaluations, exact hard-output expectations, a raw-result replay that passed, full report and file manifest. Every checkpoint hash is in the result summary. It requires the separately preserved Compass 13 frozen checkpoints to rerun model inference.

No sealed data, other investigator branches or frozen source files modified. No merges.

## Next scientific gate
A structural-coherence claim must survive **hard-output-only repairs**, parity-aware but externally supplied constraints, normal causal pathway controls, and genuinely novel event graphs. Do not label constrained output validity as a naturally computed relation.