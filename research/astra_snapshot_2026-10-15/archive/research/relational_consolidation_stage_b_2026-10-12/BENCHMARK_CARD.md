# Stage B benchmark card — generated 12 October 2026

**Status: automated validity checks pass; human grammar review pending; no model training.** This is a new source-only synthetic benchmark. It did not read the reserved active/passive challenge, EWT human-review sheet, or existing model predictions. The generator and its frozen [protocol](PROTOCOL.json) are separate from existing experiments.

## What is in it

- **8 independent graph-family instances**, 18 entities each, 6 training neighbors per entity, 54 training unordered pairs per family. Degree-preserving graph switches vary triangle counts (25–33) and focal-pair common neighbors (1–3). These are independent *instances* but are **not yet a matched high/low rewiring causal contrast**.
- Each family has **324 training events** (54 pairs × 3 reversible verbs × 2 directions), **90 development events** and **90 confirmation events** (15 withheld unordered pairs × 3 verbs × 2 directions in each split). Both directions of each withheld pair stay out of training. Across all families there are 720 confirmation event records, but within each family the two directions share the same pair and verb; they are not 720 independent graph replications.
- The wake-phase train material contains symbolic L0 and canonical L1 forms. The L2 evaluator renders canonical, passive, object cleft, subject cleft and a true role-reversed event. The first four preserve the original actor/patient IDs; reversal exchanges them. S2 derivations are emitted **only from training events**, with source IDs and role maps.
- Development and confirmation families are fixed in `PROTOCOL.json`: families 0–1 for development, 2–7 for later confirmation. The present automated audit generates and checks confirmation items but does not train or choose a model against them.

## Automated validity results

Five generator tests pass. The [audit](AUDIT.json) verifies deterministic artifacts and SHA-256 hashes, exact sentence and pair nonoverlap across splits, equal train entity degree and actor count, correct transformation role maps, and no withheld pair in S2-derived training pairs. `benchmark.py verify` independently regenerates the expected audit and checks saved file hashes. A source-only sanity check of three verbs (`chased`, `helped`, `followed`) shows grammatical passive and cleft templates, but this is **AI inspection**, not the required independent human review.

| Heuristic on 3,600 confirmation form-rows | Actor accuracy | Whole-event accuracy (all five forms) |
| --- | ---: | ---: |
| First named participant | 60% | 0% |
| Last named participant | 40% | 0% |
| Grammatical-subject surface proxy | 60% | 0% |
| `by` phrase, otherwise first participant | **80%** | 0% |
| Exact lexical memorization, then a fixed name | 10% | 0% |
| **Construction-aware hand-coded parser** | **100%** | **100%** |

The predeclared shallow-heuristic gate was `<85%` actor and `<25%` whole-event, so it passes. The 100% construction-aware parser is a **material scientific limit**: this finite grammar cannot establish that a model forms an abstract internal relation. It can still make the planned *offline versus interleaved* comparison interpretable, provided conclusions remain about transfer across held-out pairs and these specified constructions. A paper-level abstraction claim will require more constructions, lexical diversity, and authentic human-verified tests.

## Human review gate

The [blind 20-item grammar sheet](BLIND_GRAMMAR_REVIEW.csv) draws **four development examples per form** and contains no model predictions or gold answers. A human reviewer should fill `actor_written`, `patient_written`, `grammatical_yes_no` (`yes` or `no`), and any comments while viewing only that sheet. `python3 review.py check` then checks completeness and agreement against the deterministic latent event labels. As of this card: **0/20 reviewed**. No training should begin until review passes or errors are corrected and a new version is frozen.

## Reproduction

From the project root:

```sh
python3 -m unittest discover -s research/relational_consolidation_stage_b_2026-10-12 -p 'test_*.py' -v
python3 research/relational_consolidation_stage_b_2026-10-12/benchmark.py verify
python3 research/relational_consolidation_stage_b_2026-10-12/review.py check
```

`benchmark.py build` recreates generated artifacts from the protocol. It does **not** overwrite the human review sheet. The generator uses Python standard library only; no model checkpoint, GPU, or network is required. Exact artifacts and digests are in `generated/` and `AUDIT.json`.

## Next decision

After human review, run a tiny training calibration **on development family 0 only** to measure headroom, seconds per update and memory. The planned S2 pilot must compare identical examples and losses offline versus interleaved, with augmentation and extra-update controls as specified in the Stage A [experiment plan](../relational_consolidation_stage_a_2026-10-12/EXPERIMENT_PLAN.md). Do not interpret a model beating the shallow heuristics as proof of a role circuit; the grammar-aware parser already provides a counterexample.
