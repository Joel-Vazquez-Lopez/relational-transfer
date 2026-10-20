# Frozen analysis plan — 5 October 2026, before full follow-up training

This is an internal pre-run analysis lock for the D5 indirect-support follow-up. It is not a public preregistration. The code has passed graph and six-step smoke audits; no 6,000-step follow-up model has been trained or evaluated as of this lock.

## Design

- Source: `run.py`, SHA-256 `b12e4bf3e65770bdd39d1fb2640627eb76f6b96e6903f7efcb72f4c44c1ee03d`.
- Conditions: `near` (two shared training neighbours) and `far` (zero); same labelled held-out directed pair `(0,1)` and `(1,0)`, same isomorphism class of training graph, 12 observed undirected edges, endpoint degree three.
- Seeds: 7201–7206, each trained once in both conditions from identical seed IDs.
- Training: 6,000 updates, batch 128, AdamW learning rate 0.0008, fixed final checkpoint. No selection by held-out performance.
- Evaluation: 1,024 independently rendered familiar examples and 1,024 held-out examples per model. The held-out scene schedule is identical across conditions for a paired seed.

## Decision rule

The single primary outcome is the per-seed difference in mean correct-minus-reverse candidate score margin on held-out examples: `near - far`. The directional hypothesis is positive. Report all six paired differences, their mean and median, an exact sign-flip test over all 64 sign assignments for the paired mean (two-sided), and a paired bootstrap 95% interval as descriptive uncertainty. Call the directional result supported only if all twelve models attain familiar four-way accuracy at least 0.95, the mean contrast is at least +0.20 score units, and the exact two-sided p-value is at most 0.05. If the familiar gate fails, classify the run as inconclusive for indirect support. If the gate passes but the primary criteria fail, report no support for the directional hypothesis. There are no model exclusions for low held-out performance.

Held-out four-way accuracy, correct-minus-best-foil margin, query separation, and training loss are secondary descriptions. No secondary outcome can rescue a failed primary result. A negative contrast will be reported as such.

## Limits of the claim

This contrast jointly changes shared-neighbour count and shortest path length; it does not isolate which graph statistic causes a difference. It tests one isomorphic graph family in one synthetic oracle-channel task. Publication-level claims about a general motif law, internal mechanism, or natural language require further experiments.
