# Three-event stress development probe

2026-10-16. This is a development result, not a claim about an internal event representation.

## Question

Can a model keep Agent and Patient attached to the requested event when a natural passage contains three nearby instances of the same relation? Does reversing arbitrary event labels make the answers move to the wrong event?

## Frozen design

Four MAVEN-Arg passages were selected and reviewed before this probe. Each contains three `defeated` triggers, with distinct annotated Agent and Patient entities for all three. The passages and gold are in [STRESS_ITEMS.json](STRESS_ITEMS.json); the selection is in [STRESS_SELECTION.json](STRESS_SELECTION.json). The selection reused some source documents from earlier development work. There was one reviewer and no independent referent adjudication.

Each passage received three cue conditions: no trigger markers with first/second/third wording; A/B/C markers in source order; and C/B/A markers in source order. Each event was queried separately for Agent and Patient with fixed defeat-specific wording. That gives 4 documents × 3 events × 2 roles × 3 conditions = 72 responses per model. The [protocol](STRESS_PROTOCOL.json) was frozen before model inference (SHA-256 `b3fa5b50e08f0c937a8d236501aacb568017a62602b29ab8c5610f6f4e143530`). The primary score was normalized exact match to the annotated local mention. The secondary unique-alias rule did not add any correct answers in this run.

Gemma 4 31B IT was queried through Berget at temperature zero. The existing pinned Qwen2.5 7B Instruct MLX 4-bit model ran locally with greedy decoding. Both used the same questions and 24-token cap. Gemma consumed an estimated €0.00245550, within the €0.10 hard cap; the separate budget ledger conservatively accounts double estimated cost.

## Results

| Model | Unmarked ordinal | A/B/C markers | Reversed C/B/A markers | All |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 31B | 22/24 | 24/24 | 24/24 | 70/72 |
| Qwen2.5 7B | 16/24 | 24/24 | 23/24 | 63/72 |

For the 24 matched event-role queries, Gemma's answers were identical under A/B/C and C/B/A. Qwen's answers were identical for 23/24. Qwen's sole marked-condition miss answered the first match's Patient query with its Agent, Daniel Bryan, under reversed labels. It was a role error, not an event swap.

The unmarked SummerSlam passage also says Daniel Bryan later lost a title to Randy Orton between the first and second explicit `defeated` triggers. Therefore “second defeat described” is semantically ambiguous, despite the builder's exact three-trigger check. Both Gemma errors and three Qwen errors were on that passage. A sensitivity count excluding all six SummerSlam unmarked questions leaves Gemma **18/18** and Qwen **13/18** unmarked exact answers. The excluded answers stay in the raw record; this is an after-result diagnostic, not a revised primary score.

Among the other unmarked Qwen errors, the second Extreme Rules match was answered with **both** participants from the third match (Sheamus/Daniel Bryan). The second Bad Blood match was answered with Goldberg for both roles, so it was not a coherent wrong-event pair. One additional Extreme Rules answer swapped Agent into Patient. Thus there is one complete wrong-event tuple in a small development sample. It does not distinguish a shared event selector from local lexical and position-based alternatives.

## Decision

The stronger hosted model solves all unambiguous source cases in this panel, and both models solve almost all explicitly marked cases. That is encouraging for reliable event-role extraction, but it deprives this panel of the error variation needed to test the proposed event-first account. The label reversal establishes behavioral robustness to arbitrary markers on these passages; it does not locate an internal event variable.

Do not treat ordinal errors as mechanism evidence. The next discriminating experiment needs independently reviewed natural passages with multiple same-type events and no implicit extra events, or an open-weight causal intervention with matched lexical, position, and answer-copy controls. A result deserves a mechanistic claim only if it predicts held-out joint responses and intervention effects better than those rivals. Further Gemma prompt repetitions on this four-document panel are low value.

## Reproducibility and limits

The [runner](stress_run.py) contains generation and scoring code. Raw outputs and scored details are in [Gemma analysis](STRESS_GEMMA_ANALYSIS.json) and [Qwen analysis](STRESS_QWEN7B_ANALYSIS.json); each analysis records the raw-file SHA-256. The source archive SHA-256 is recorded in the frozen protocol. The sample is tiny, relation-specific, manually selected with labels, and partially overlaps earlier development documents. The Berget model name is a provider alias without a pinned weight revision. These are descriptive counts, not statistical evidence of a general model ranking or a novel mechanism.
