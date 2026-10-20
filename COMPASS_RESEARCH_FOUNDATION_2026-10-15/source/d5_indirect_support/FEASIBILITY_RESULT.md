# Six-pair feasibility result — 5 October 2026

The original six-pair run completed: 12/12 models trained 6,000 updates, saved checkpoints, and passed SHA-256 verification. All familiar four-way accuracies were at least 99.90%, passing the frozen 95% gate. The frozen directional hypothesis expected `near > far` on the held-out correct-minus-reverse score margin. It **failed**.

| Seed | Near held accuracy | Far held accuracy | Near−far margin |
|---:|---:|---:|---:|
| 7201 | 0.5781 | 0.9922 | −9.7817 |
| 7202 | 0.4609 | 0.9951 | −11.3991 |
| 7203 | 0.5215 | 0.9893 | −11.0304 |
| 7204 | 0.9531 | 0.9629 | −0.4397 |
| 7205 | 0.4922 | 0.9912 | −11.2524 |
| 7206 | 0.5811 | 1.0000 | −12.0684 |

Mean held accuracy was 0.5978 near and 0.9884 far. The mean primary near-minus-far margin was −9.3286; all six paired differences were negative. The exact two-sided sign-flip p-value was 0.03125, and the paired bootstrap 95% interval was [−11.5363, −5.6254]. These statistics describe an **unexpected reverse-direction association** in this implementation; the preregistered positive result is unsupported.

The graph structures are isomorphic and exposure-balanced, but the initial generator did not make the entire rendered training schedule an exact label-permuted tensor pair. The conditions also differ jointly in common-neighbour count and shortest path length. The result therefore does not identify a specific local graph statistic or establish a publication-level causal mechanism. A fresh exact-pairing replication is specified in `MATCHED_PROTOCOL.md` and uses new seeds.

Raw metrics, checkpoints, hashes, and exact analysis are under `results/`; run `analyze.py` to verify them. The result remains labeled feasibility evidence regardless of the replication outcome.
