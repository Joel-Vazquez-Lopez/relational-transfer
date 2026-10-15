# Comput Report 005 — event-cluster uncertainty is not recoverable from headline margins

**Source status:** Mathematical analysis of Astra's archived, repeatedly examined 12-event, 48-output per-condition fixed-field development aggregates. No new model outputs. No sealed data.

Using 12 matched source-event clusters of 4 binary observations each, enumerate all mathematically possible arm totals per cluster consistent with the two reported global arm margins. Assign each cluster the sign of its within-cluster correct-count difference (ties omitted). Obtain the **sharp attainable minimum and maximum** two-sided exact cluster-sign p-values. Exact p-values require independently sampled clusters for conventional inference; this is a sensitivity bound, not proof of independence.

| Arm comparison (48 outputs per arm) | A vs B correct | Possible cluster exact sign p |
|---|---|---|
| Source-correct reference fields vs swapped | 34 vs 26 | 0.00634765625 to 1 |
| Fresh vs transcript accuracy | 26 vs 7 | 0.00048828125 to 1 |
| Fresh vs transcript compliance | 42 vs 18 | 0.00048828125 to 0.548828125 |

Thus no one of these three published margins alone forces a p<0.05 conclusion in the event-cluster sign test. This **does not** establish that the observed effects are false, and it does not invalidate Astra's separately reported event-bootstrap confidence intervals using its privately retained paired event records. It simply demonstrates that margins in the GitHub snapshot cannot independently reproduce cluster inference.

Assumptions: 12 source events, exactly four observations in each arm per event, observations binary, no constraints on which responses are correct within each event beyond the arm totals. Some designs with additional within-event relationships could narrow these mathematical possibilities. The computations do not use semantic gold features. The enumeration is exact for the stated assumptions; tested against brute-force small cases.

Run `python3 research/comput/cluster_margin_bounds.py` and `python3 -m unittest discover -s research/comput -p 'test_*.py'`.

**Decision:** Seek paired raw scores per condition and event ID, then compute actual event-sign counts and a cluster bootstrap/permutation test. Do not infer significance, mechanisms, or effect size independence from these margins alone.
