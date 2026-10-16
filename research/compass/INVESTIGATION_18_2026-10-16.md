# Compass Investigation 18 — Instruction–operand dissociation under reversed source presentation

**16 October 2026 · Compass branch `research/compass-reuse` · Synthetic open-development only**

## Question and preregistered contrast

Investigation 17 found that, for equal-gold edits, neural answers changed more often when the **final operation instruction** was replaced than when the **referenced relation-table value** was replaced. The final instruction token was closer to the prediction token. Is this asymmetry explained simply by which item appears later?

This experiment trains **one ordinary causal Transformer per seed on both presentation layouts** and then evaluates precisely the *same* worlds and paired edits in each layout:

- **Table first:** eight function-table values; start entity; three program instructions; answer. Last instruction occurs after the operand.
- **Instructions first:** start entity; three instructions; eight function-table values; answer. Operand occurs after the last instruction.

Identical token identities, vocabulary, program semantics and 13-token length across layouts. No oracle state, engineered cache or hard-coded operator is supplied to the network. Both layouts are sampled equally from the same world/short-program training generator. Crucially, **the reversal also changes whether intermediate operation tokens can attend to the table**. Therefore this is a stronger presentation-order counterfactual, **not a pure isolated recency intervention**.

The protocol was frozen in `runs/frozen_protocol.json` before model fitting. Three random seeds (19/29/41), three causal-attention blocks, width 48, 1,600 AdamW updates, 128 training samples/update, LR .003. Training examples consist of one or two active operations among three slots (including shuffled NOOP locations), from 9,000 non-bijective-A worlds. Test: 96 disjoint non-bijective-A worlds and 96 disjoint **bijective-A worlds absent from training**. Questions within world are correlated. These results are development-only, not sealed confirmation.

## Held-out graph-family task competence

Each model/layout: 768 one-hop, 1,536 two-hop, 3,072 three-hop predictions on the same 96 withheld bijective-A worlds.

| Seed | Layout | 1-hop | 2-hop | 3-hop | Complete 32-answer three-hop worlds |
|---|---|---:|---:|---:|---:|
| 19 | Table first | 669/768 | 1,522/1,536 | 1,292/3,072 | 1/96 |
| 19 | Instructions first | 768/768 | 1,528/1,536 | 1,232/3,072 | 0/96 |
| 29 | Table first | 768/768 | 1,536/1,536 | 1,232/3,072 | 1/96 |
| 29 | Instructions first | 768/768 | 1,536/1,536 | 1,212/3,072 | 2/96 |
| 41 | Table first | 768/768 | 1,517/1,536 | 1,207/3,072 | 0/96 |
| 41 | Instructions first | 768/768 | 1,530/1,536 | 1,059/3,072 | 0/96 |

All three models **fail the whole-world compositionality gate** in both layouts. Seed 19 table-first also **fails the short-task competence gate on one-step queries**, though its two-step score is high. The other two seeds approximately or perfectly master the short queries. Do not infer a learned general composition operator.

For reference on the same 3,072 held-out three-hop queries: an exact symbolic interpreter scores 100%; ignoring the first operation scores **1,324/3,072 (43.10%)**; a direct final lookup using the original start scores **1,413/3,072 (46.00%)**. Mean learned three-hop accuracy is **40.5% table-first** versus **38.0% instructions-first** across three seeds; both below the direct final-lookup shortcut.

## Matched counterfactual design

From 95/96 held-out bijective-A worlds, select two eligible examples per world where the final program operation is B, the originally consulted B-table cell was not used earlier, and changing the final B operation to A changes the correct answer. Construct a second counterfactual by **altering that B-table cell to the same new correct answer**. Therefore each pair contains:

- Untouched world/question, original gold label.
- Final **B→A instruction edit**; new gold label.
- Original B instruction but **referenced B-table value edited**; **identical new gold**.

The complete A table stays unchanged (bijective) and both counterfactuals are valid. Compare *changes in the original model prediction* and separately whether a counterfactual prediction is actually correct. There are **188 paired questions from 95 world clusters per seed/layout**. Raw predictions for every intervention are retained.

| Seed | Table-first instruction change | Table-first operand change | Instructions-first instruction change | Instructions-first operand change |
|---|---:|---:|---:|---:|
| 19 | 93/188 | 81/188 | 118/188 | 66/188 |
| 29 | 93/188 | 24/188 | 125/188 | 39/188 |
| 41 | 143/188 | 33/188 | 118/188 | 59/188 |

**Instruction minus operand change-rate gap** (percentage points):

| Seed | Table first | Instructions first | Interaction: instructions-first minus table-first | Conditional world-cluster bootstrap interval for interaction |
|---|---:|---:|---:|---|
| 19 | +6.38 | +27.66 | +21.28 | [+8.42,+34.39] pp |
| 29 | +36.70 | +45.74 | +9.04 | [+1.05,+17.99] pp |
| 41 | +58.51 | +31.38 | −27.13 | [−37.84,−16.49] pp |

Bootstraps resample 95 world clusters **within a fixed model seed**, do not correct for adaptive protocol selection, and are not population-level inference about Transformers. Direction of the layout interaction **is inconsistent across seeds**. Yet the within-layout instruction-response gap is positive in all six contrasts, including all three instruction-first (table-late) presentations. Accordingly, simple “the instruction is always closer to the answer” is *insufficient* to account for the observed patterns in these trained models; this does **not** reject more general positional, token-type, attention-access or heuristic mechanisms.

**Correctness caveat:** On cases where each layout's untouched response was correct, both interventions should switch to the same new gold. However the model's correct-response rates under the two edits remain weak. Considering only cases with a correct untouched answer in *both* layouts, counts are n=47,44,54 for the three seeds, and no consistent positive benefit to instruction edits emerges. Change-rate asymmetry is **not** a demonstration of semantic update competence.

## Adversarial computational shortcuts

The identical 188 matched examples were evaluated under deterministic non-generalizing programs. A complete interpreter changes correctly on **188/188 instruction edits and 188/188 operand edits**. But:

| Policy | Instruction edits changing answer | Operand edits changing answer | Full three-step computation? |
|---|---:|---:|---|
| Ignore first operation, execute last two | 152/188 | 68/188 | No |
| Perform only final function lookup from original start | 155/188 | 48/188 | No |
| Return arbitrary function of last instruction symbol | 188/188 | 0/188 | No |
| Ignore final operation, execute first two | 0/188 | 0/188 | No |

**Therefore the behavioral gap is not diagnostic of a native relational binding circuit or completed composition.** The first two strong shortcuts reproduce it without executing the intended three-step program.

## Scientific decision

**Supported:** a response asymmetry survives reversing the instruction–table order within one shared learned model, across three seeds, on family-preserving equal-gold counterfactuals. The direction of the reversal effect itself is heterogeneous by seed.

**Not supported:** a uniquely relational or causal neural mechanism, capacity to execute complete three-hop programs, authentic-language transfer, or an explanation isolated to token recency. The layouts differ in causal token access as well as position; semantic competence remains weak; this was developed after examining prior open-development failures. The rich alternative of a two-operation shortcut is explicitly viable.

**Next gate:** develop a new controlled architecture/dataset where *input positions can be moved independently of token type and causal access* (e.g., fixed-position tagged table entries, masked state buffers or externally equated attention interfaces) and obtain strong whole-world competence before labeling a native circuit. Focus on evidence from **correct changed answers**, not sensitivity alone. Pending independent validation, do not claim ACL novelty.

## Reproduction

`python -m unittest -v test_exp18` (4 tests), `python exp18.py --out runs --steps 1600 --seeds 19 29 41`, `python replay_checkpoints.py --root runs` (all 3 frozen checkpoints and exact per-case predictions verified), `python analyze_pairs.py --root runs` (paired and per-world bootstrap), `python shortcut_controls.py` (known algorithm controls). Dependencies: Python, CPU PyTorch. The original `exp16_reference.py` and `matched_family_preserving.py` are bundled unchanged except their relocation. All data synthetic/open; no original sealed datasets or paid inference used.

GitHub branch will retain the scientific summary and executable stimulus/analysis controls. Complete `.pt` checkpoints, raw paired predictions, logs and SHA-256 archive manifest remain in the Compass-conversation ZIP, not in GitHub.

**Complete archive SHA-256:** `1357c0f08466ca39b826a15eb0dd54ffa5c1df2e2224489275405faa9e3d2bc8`. Contains 25 hashed files and a manifest; archived in the Compass conversation, not GitHub.
