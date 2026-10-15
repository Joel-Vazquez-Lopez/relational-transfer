# Comput Falsification 003 — paired wording effect, cluster uncertainty
Date: 2026-10-15. Evidence: **archival aggregates** from `evidence/EXPERIMENT_REGISTRY.md` and previous handoff. This is a new **mathematical reanalysis of aggregates**, not an independent re-evaluation of original model outputs.

## Method
Two-sided exact binomial sign tests count gains and losses among 12 paired event-level joint-correctness outcomes, ignoring unchanged cases. This calculation assumes independent events; **that assumption is invalid or questionable** because 12 events span only six name-pair groups, each containing two events.

To avoid inventing missing per-event group membership, we enumerate **all feasible allocations** of gains, losses and unchanged observations to six clusters of size two. Each cluster is positive if gains exceed losses, negative if losses exceed gains, otherwise tied. For each feasible cluster-sign configuration, calculate a two-sided exact sign p-value. Report the full range, not an optimized allocation.

## Findings
| Model | Before→after joint correct | Gains/losses | Event-level p (if independent) | Feasible cluster-level p range |
|---|---:|---:|---:|---:|
| Qwen 7B | 3/12 → 8/12 | 7/2 | 0.1796875 | 0.125–0.6875 |
| Phi-3 | 9/12 → 1/12 | 0/8 | 0.0078125 | 0.03125–0.125 |

The Phi wording sensitivity is consistent in direction across eight changed cases, but cluster-level significance is **not identified by the published aggregates**. It may or may not pass p<0.05. This also does not account for repeated prompt development, multiple comparisons or selection bias. Neither family contrast identifies a mechanism.

## Additional negative control from Compass
`research/compass/INVESTIGATION_04_2026-10-15.md` reports six learned schedule-controlled networks all fitting the 144-item training table, but held-out composition accuracy of 9–10/36 (cache) and 11–16/36 (reconstruction). Comput treats training memorization, matched-matrix externally hard-coded composition, and independently learned composition as *distinct observables*. No claim of pretrained native reuse follows.

## Reproduction
`python3 research/comput/paired_audit.py`
`python3 -m unittest discover -s research/comput -p 'test_*.py'`

## Immediate next discriminating measurement
Obtain raw 12-event predictions and group identities. Fit first/last/subject/question-conditioned rules on development groups, then leave one name pair out, test on each held-out pair, and evaluate residual event-level disagreements. Separate phonetic/lexical name cues from grammatical and order cues. Preserve untouched confirmation data. Do not claim mechanistic algorithms from behavioral consistency alone.
