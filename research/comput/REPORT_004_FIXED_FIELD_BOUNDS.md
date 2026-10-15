# Comput Report 004 — fixed-fields margin falsification
Date: 2026-10-15. **Source:** Astra historical exploratory development report, `research/astra_snapshot_2026-10-15/archive/research/frontier_fixed_fields_2026-10-15/RESULTS.md`, in `codex/astra-research-2026-10-15`. Original raw event predictions absent in GitHub; this audit is **independent mathematical reanalysis of reported margins**, not independently rerun model inference.

## Exact, reproducible results
From 48 repeated question observations in each reported arm:

| Contrast, A versus B | Margins | Minimum possible changed decisions | Maximum | Possible two-sided paired exact p, assuming independent questions |
|---|---|---:|---:|---|
| Gold reference fields vs role-swapped reference fields (accuracy) | 34 vs 26 | 8 | 36 | 0.0078125 to 0.2429849540 |
| Fresh generated-fields+sentence vs transcript generated-fields+sentence (accuracy) | 26 vs 7 | 19 | 33 | 0.0000038147 to 0.0013187272 |
| The same packaging contrast (format compliance) | 42 vs 18 | 24 | 36 | 0.0000001192 to 0.0000696017 |

These are **sharp combinatorial bounds conditional on compatible binary paired outcomes**; no per-event pairings were invented. The paired exact p-values describe sensitivity under a question-independence assumption, **not valid event-clustered inference** for just 12 underlying source events. The 48 observations repeat 12 documents under questions/field orders. Model changes may be mediated by format, role interpretation, sentence use, position, or interactions; none is identified by this table.

## Falsification interpretation
- The +8/48 reference vs swapped accuracy margin alone does **not** establish paired significance; some unseen joint tables are significant, others are not.
- The fresh/transcript descriptive differences are substantial under every compatible question-level pairing, yet packaging changes several factors at once and source-event cluster uncertainty is not calculated.
- Reference-field condition with 34/48 correct and 42/48 compliant leaves eight **compliant, incorrect** responses (Astra reports these all select the opposite field), so a format-only story is incomplete. That does not show an internal role representation or causal readout mechanism.
- Cannot compare 11/24 direct to 26/48 generated-fields with naive independent paired tests: different question multiplicities and experimental interfaces.

## Commands
`python3 research/comput/margin_bounds.py`
`python3 -m unittest discover -s research/comput -p 'test_*.py'`

## Smallest next discriminating experiment
First request the existing raw, per-event crossed-condition responses and cluster IDs; separately tabulate question-level vs event-clustered paired transitions, compliant-but-wrong opposite-field choices, and whether field order or role question explains switches. No new model inference until that reanalysis is complete. A new prospective factorial should independently randomize last-turn instruction locality, field swap, and explicit role questions on held-out documents, and score strong direct QA in the same format and compute budget.

**Do not claim native cached computation, internal interference, independent natural-language repair, or a paper result from these aggregate margins.**
