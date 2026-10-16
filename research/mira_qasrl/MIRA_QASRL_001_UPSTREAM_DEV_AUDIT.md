# MIRA-QASRL-001 — Independent upstream development shortcut audit

Date: 2026-10-16. Status: **observed corpus metadata, not pretrained-model inference**.

## Provenance
QA-SRL Gold upstream `plroit/qasrl-gs`, pinned commit `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83`.
Read `data/gold/{wikinews,wikipedia}.dev.gold.csv` and `data/sentences/{wikinews,wikipedia}.dev.full.csv` via connected GitHub. No test partition accessed. Original sentence text and worker IDs are not reproduced here.

## Independently computed results

| Count | Wikinews dev | Wikipedia dev | Combined |
|---|---:|---:|---:|
| Gold QA rows | 3714 | 3474 | 7188 |
| Sentence reference rows | 917 | 917 | 1834 |
| Exact eligible single-span QA rows | 3239 | 2794 | 6033 |
| Multi-span QA rows excluded | 474 | 680 | 1154 |
| Surface disagreements excluded | 1 | 0 | 1 |
| Answer token sequence occurs at 2+ positions | 55 | 78 | 133 |
| Eligible sentence clusters | 499 | 494 | 993 |
| Eligible sentence clusters with 2+ predicate indices | 345 | 322 | 667 |
| Single-token answer at sentence start | 180 | 134 | 314 |
| Single-token answer immediately before predicate | 158 | 159 | 317 |
| Single-token answer immediately after predicate | 165 | 118 | 283 |

All counts derived by parsing CSV quoted fields and joining `qasrl_id`; a single accepted `answer_range` must have valid half-open coordinates and its token slice must exactly match the `answer` surface. Multiple accepted answer spans were conservatively excluded, not scored wrong. Counts are metadata of this selection, not model accuracies.

## Interpretation and limitations

1. **133/6033 = 2.20%** of eligible rows have the exact answer-token sequence at multiple sentence positions. Correct text alone cannot certify which occurrence was selected. This is a *surface-ambiguity risk set*, not 133 verified event-collision cases or 133 model errors.
2. **667/993** eligible sentence clusters contain annotations on multiple predicate indices. This is not evidence that those predicates share a lemma, denote distinct events, or support event-specific disambiguation. Do not use this count as an event-collision sample.
3. First-token and adjacent-token diagnostic counts are *weak position heuristics*. They do not compete with dependency parsing, syntax-aware role extraction, or the official QA-SRL metric.
4. The `tokens` column was whitespace-split following the upstream format. The one observed exact surface disagreement was not repaired.
5. QA-SRL human-authored questions and answer spans are not universal binary actor/patient labels. Some questions request duration, location or other semantic arguments.
6. Sentence clusters are not verified independent documents. Source rights and model-training contamination remain open.
7. No test set, model inference, statistical significance, or mechanistic causal evidence is claimed.

## Next discriminating gate

Construct a **human-reviewed subset** of same-lemma, distinct-event examples with explicit target predicate indices and admissible answer spans. Partition it into (a) identical answer surface with multiple occurrences, (b) unique surface with multiple events, and (c) no genuine event collision. Compare answer-string scoring to index- or source-span-grounded scoring. If the model cannot output source indices reliably, label event identity **unobservable**, not wrong. Evaluate syntax-aware and last-mentioned-event baselines on exactly the same cases. Coordinate with Brain-X BX-006, Comput and Nova; avoid duplicate runners.

## Reproduction

`python research/mira_qasrl/shortcut_diagnostics.py --gold path/to/wikinews.dev.gold.csv --sentences path/to/wikinews.dev.full.csv --partition wikinews.dev`

Run separately for Wikipedia development. Script is development-only and outputs aggregate counts. Upstream data not vendored.
