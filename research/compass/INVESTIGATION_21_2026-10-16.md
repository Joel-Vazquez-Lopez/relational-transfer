# Compass Investigation 21 — Exact-position operand shielding and the limits of behavioral selectivity

**16 October 2026 | Independent Compass workstream | Completed frozen-checkpoint development audit.**

## Core question
Can the models distinguish when the *same relation-table operand*, at precisely the *same token position*, is semantically needed by a final instruction versus irrelevant? Earlier Compass 19–20 irrelevant-source controls changed **different** cells/positions, so attention to position could explain apparent relevance effects.

This study fixes that mismatch with an **exact-cell 2 × 2 factorial**. It does **not** attempt to locate a latent causal circuit.

## Causal design
Original program uses final operation `B`; its third-step input comes from a specific B-table cell that **does not appear anywhere in the first two operation paths**. The gold value of that cell is initially `old_gold = B(u)`. We construct a new B-cell content equal to `new_gold = A(u)` and choose only instances with `new_gold != old_gold`. We can also flip the final operation to A. Thus, while preserving the A-bijective held-out graph family:

| Condition | Last operation | B[u] table cell | Mathematical gold |
|---|---|---|---|
| **B0** | B | Original | old_gold |
| **B1** | B | Changed to A[u] | new_gold |
| **A0** | A | Original | new_gold |
| **A1** | A | Changed to A[u] | new_gold |

The **same cell at the exact same table position** is edited in B0→B1 and A0→A1. Its substitution is identical, and the first two operations remain identical. Under correct computation, it should affect answers only with final-B, since final-A does not read B[u]. Thus the *four-way semantic law* is `B0 != B1 = A0 = A1` in gold-answer space.

We measure, for `p = P(new_gold)`:

`Delta_B = p(B1) - p(B0)`

`Delta_A = p(A1) - p(A0)`

`Selective interaction = Delta_B - Delta_A`.

This is **behavioral input selectivity**, not native reuse or a hidden-state intervention. All stimuli selected post hoc to force a gold-changing operand edit; reported metrics are **conditional on that selection**.

## Source and execution

Frozen, unmodified Investigation 18 models (three causal-attention blocks, width 48, 1,600 updates, seeds 19/29/41), each trained on table-first and instructions-first presentations. **No training** in Investigation 21. The same 188 selected cases from **95 synthetic held-out A-bijective worlds** were evaluated in both layouts, yielding **1,128 model × layout × stimulus quadruples**.

All `B0`, `B1`, and `A0` predictions were **exactly replayed and matched** against the existing Investigation 19 JSONL. `A1` is the new double-intervention forward pass. The model inference was performed on CPU PyTorch, model state dictionaries loaded with `weights_only=True`. Saved output probabilities and argmax for all four conditions per row. The 95 source worlds—not 1,128 rows—define the primary bootstrap units. All 5 new tests pass. This is development-only: worlds, constructions, and diagnoses are not preregistered sealed confirmation.

## Frozen model findings

| Model-averaged outcome across all 1,128 dependent quadruples | Value |
|---|---:|
| `Delta_B`: target-probability change when B operand is relevant | **+19.15 percentage points**, conditional cluster interval **[+14.97,+23.64]** |
| `Delta_A`: *same exact B-cell edit* when final instruction A makes it irrelevant | **+3.00 pp**, conditional interval **[+1.26,+4.86]** |
| **Selective probability interaction** | **+16.15 pp**, conditional interval **[+12.91,+19.55]** |
| Total-variation distribution change under B-cell edit, final B | **25.81%** |
| Total-variation distribution change under same edit, final A | **13.77%** |
| Argmax invariant under A0→A1 (should be invariant) | **970/1,128 = 86.0%** |
| B1 answers the correct new gold | **457/1,128 = 40.5%** |
| A0 and A1 both answer correct new gold | **238/1,128 = 21.1%** |
| **All four conditions semantically correct** | **116/1,128 = 10.3%** |

**Every model-seed/layout condition exhibits positive `Delta_B - Delta_A`**, between +11.55 and +22.91 percentage points. That is some conditional input selectivity, despite the poor four-condition correctness.

The 95-world paired bootstrap intervals in the summary file are **exploratory and conditional**; seeds and layouts reuse the same worlds, and no held-out experimental hypothesis test was preregistered. These should not be presented as broader statistical confidence about real LLMs.

## Strong shortcut falsification

We executed 188/188 matched quadruples on transparent computational programs and measured **correct-new-gold *indicator* interactions**, the deterministic analog of the probability score. They are not numerically calibrated against neural probabilities in a common output distribution, but they reveal what behaviors can induce a positive conditional interaction.

| Known program | Correctness-based selective interaction | All four semantically correct |
|---|---:|---:|
| Full three-step composition | **100.0 pp** | **100.0%** |
| **Only final two operations** | **35.6 pp** | **34.0%** |
| **Only final operation / local one-hop lookup** | **25.5 pp** | **25.5%** |
| Only first two operations | **0.0 pp** | **0.0%** |
| Last-operation token but no relation-table access | **0.0 pp** | **0.0%** |

The final-one-operation shortcut shows conditional sensitivity to precisely the relevant source B-cell **without computing the first two operations at all**. Its 25.5% full quadruple correctness exceeds the neural 10.3% on this selected panel. A positive conditional interaction therefore **does not identify complete composition, internal event binding, or a native relational cache**.

## Interpretation and limitations

**Supported:** The tested models respond more to the exact same operand substitution when the final instruction refers to its relation than when it does not. This removes **source-cell token-position mismatch** from this narrow contrast. They show significant behavioral conditional source relevance but low correctness and nonzero sensitivity to a semantically shielded operand.

**Not supported:** Genuine three-step relational algebra, spontaneous pre-question relation caching, a specific hidden-state binding representation, internal source routing or real pretrained-model generalization. Even the unchanged-cell/same-position comparison may involve operation-conditioned attention; it cannot adjudicate algorithm identity. Question-time reconstruction, local lookups and shortcut rules remain viable.

**Selection/provenance:** The 188 cases were selected with the true semantics to guarantee an equal-new-gold pair; observations were clustered over only 95 unique worlds from a single synthetic grammar. Within-world and across-seed re-evaluation are correlated. Selection biases preclude a population-level interpretation.

**Team interface:** No overlap with Astra's Qwen prompt-field-order study or Mira/Sentry's authentic source intake. As of this check, human multi-event annotation, corpus rights and model-overlap checks remain unresolved; no natural-language inference was run.

## Decision gate

Before another mechanistic claim, a competent model should pass **all four** semantically correct conditions, including A0→A1 invariance; beat both last-one and suffix-two symbolic shortcuts on independent structures; transfer across graph families and human-verified natural events; and survive native-circuit interventions against competence-matched reconstruction controls. **Do not scale these weak models solely because they exhibit a selective interaction.**

## Reproduction

The full self-contained package contains:
- `exp21_factorial.py`: exact frozen forward-pass reproduction, 4-way outputs, world-cluster bootstrap.
- `shortcut_controls.py`: known algorithmic policies against the same cases.
- `standalone_certificate.py`: **stdlib-only, dependency-free** exact-case generator + known-shortcut controls.
- `test_exp21.py`: five passing scientific integrity and strict replay tests.
- `reference18/`: original saved model source, frozen protocol and all 3 checkpoints.
- `reference19/results/paired_predictions.jsonl`: previously frozen triple outputs used for readback.
- `results/four_way_predictions.jsonl`, `results/summary.json`, `results/shortcut_controls.json`, training-free CPU log, SHA-256 file manifest and verifier.

Run `python exp21_factorial.py`, `python shortcut_controls.py`, and `python -m unittest -v test_exp21` after extraction, or `python standalone_certificate.py` without PyTorch.

**Conclusion: exact-position conditional input selectivity is observed, but it fails as a sufficient discriminator of systematic relational computation.**

**Verified full artifact:** `COMPASS_INVESTIGATION_21_EXACT_POSITION_SHIELDING_2026-10-16.zip` (SHA-256 `48f622d675e123125de537ef84b42a7529ef9ea658e5d242af85f1deb8f49d35`; 23 ZIP entries; 22 SHA-256-authenticated files). Conversation attachment; not GitHub-hosted.
