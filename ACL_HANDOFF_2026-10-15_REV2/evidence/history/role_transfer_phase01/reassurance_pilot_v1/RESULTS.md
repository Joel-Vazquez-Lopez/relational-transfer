# Development-only reassurance pilot, 6 October 2026

Question and fixed design: see `PROTOCOL.json`. This used `new1`, a graph
already studied historically but not used in the six-start auxiliary
development block, with new training seed 261003. It is not one of the eight
sealed fresh confirmation graphs. All three arms used the same corrected
generator and 6,000-update schedule as the earlier development runs. The
pilot's method and success check were recorded before these outcomes.

| Arm | Unseen binary role | Unseen four-choice | Unseen event | Seen four-choice | Weak unseen pairs (<80%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced baseline | 93.52% | 92.48% | 98.79% | 99.84% | 2/25 |
| Unordered identity auxiliary | 86.08% | 85.54% | 99.29% | 99.93% | 8/25 |
| Ordered role auxiliary | **98.00%** | **96.59%** | 98.20% | 99.95% | **0/25** |

Primary evaluation scene seed: 260107, 512 crossed renderings per pair.
Ordered roles improved unseen binary role accuracy by **4.48 points over
baseline** and **11.92 points over unordered identity**. Its event score was
0.59 points below baseline and familiar four-choice was above 99%. The
prespecified pilot check passed.

Scene seed 260108 gave 93.14%, 86.09%, and 97.93% unseen binary role accuracy
for baseline, unordered, and ordered arms. Weak-pair counts were again 2, 8,
and 0. All three checkpoint hashes match their saved metrics.

This supports feasibility on one further development graph. It does not
increase the number of untouched confirmation graphs, justify a population
significance claim, or remove the need for the sealed 72-model confirmation.
