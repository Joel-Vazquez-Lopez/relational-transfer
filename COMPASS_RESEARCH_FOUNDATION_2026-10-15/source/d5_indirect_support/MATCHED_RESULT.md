# Exact-pairing replication result — 5 October 2026

The fresh six-pair run specified in `MATCHED_PROTOCOL.md` is complete. All 12 final checkpoints and metrics exist; `analyze.py --matched` verified their SHA-256 hashes and the frozen source hashes. The graph, 768 canonical training pool entries, paired scene rendering, and identical held-out evaluation passed the pre-run audit.

All 12 familiar four-way accuracies are at least 99.12%, passing the 95% interpretation gate. The pre-run reverse-direction replication rule **passes**: mean near-minus-far held-out correct-minus-reverse score margin is −10.5300, exceeding the frozen −0.20 magnitude requirement, with exact two-sided sign-flip p = 0.03125. All six paired margins are negative. The paired bootstrap 95% interval is [−11.8700, −9.1528]. This is a prospective replication of an effect first seen in the feasibility run, not an independent discovery.

| Seed | Near held accuracy | Far held accuracy | Near−far score margin |
|---:|---:|---:|---:|
| 8201 | 0.4863 | 0.9941 | −12.4191 |
| 8202 | 0.4912 | 0.9980 | −8.4386 |
| 8203 | 0.4980 | 1.0000 | −12.3074 |
| 8204 | 0.4678 | 0.9980 | −10.9953 |
| 8205 | 0.4150 | 0.9756 | −10.8832 |
| 8206 | 0.6406 | 0.9883 | −8.1366 |

Mean held-out four-way accuracy was 0.4998 near and 0.9924 far. In this controlled synthetic task, the unseen relation is much easier when its endpoints occupy the far graph positions. The near condition also has much lower final query separation in all six matched seeds, a secondary observation.

## What this does and does not establish

The exact-pairing design rules out a different rendered training schedule as the explanation for the observed contrast. It does **not** isolate common-neighbour count: near and far differ in shortest path and potentially other local pair properties. It does not show an internal mechanism, a general graph law, a language-model effect, or transfer to natural language. The original positive-direction hypothesis failed; the reverse-direction pattern was selected from the feasibility result and then replicated prospectively on fresh seeds.

The next decisive experiment is a matched graph family containing nonedges with one versus two common neighbours at the same shortest path length, plus multiple graph instances. A second task would be needed for a broader publication claim.

Machine-readable results: `results/matched/analysis.json`; all per-model metrics and checkpoints are in `results/matched/{near,far}/seed_*/`.
