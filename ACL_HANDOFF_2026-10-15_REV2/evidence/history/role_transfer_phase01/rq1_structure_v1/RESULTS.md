# RQ1 network arrangement contrast: final result

Completed 8 October 2026. The frozen directional prediction **did not pass**.

## Design and integrity

The [protocol](PROTOCOL.json) and [graph manifest](GRAPH_MANIFEST.json) were
fixed before this study's training. Eight pairs of new graph layouts were
selected without looking at model outcomes. Each graph had ten entities,
twenty observed unordered pairs, four partners per entity, and 1,280 balanced
training configurations. One graph in each pair had at most four triangles and
the other at least eight. The same entity labeling and three training starts
were used within each graph pair. The primary test used only the 13–17 entity
pairs that were unobserved in **both** graphs of each comparison, so the tested
identities and rendered test cases matched within a graph pair.

All **48/48** planned models completed. The final graph/source/protocol audit
passed, all 48 saved checkpoint hashes matched their metrics files, and no
graph or start was excluded. The lowest observed-pair four-choice accuracy
among the 48 models was **99.67%**, so poor learning of familiar cases does not
explain the transfer comparison. The [interim reads](EARLY_PEEK.md) were
requested by the user and recorded; neither changed training or analysis.

## Frozen primary analysis

The primary measure was binary actor/patient accuracy on pairs absent from
both networks, averaged over those pairs and three starts within each network
pair. Network pair was the independent analysis unit. Positive differences
favor the lower-triangle network.

| Network pair | Shared unseen pairs | Lower triangles | Higher triangles | Lower − higher |
|---|---:|---:|---:|---:|
| pair0 | 15 | 90.78% | 89.57% | +1.21 points |
| pair1 | 14 | 74.57% | 85.60% | −11.03 points |
| pair2 | 15 | 84.80% | 90.71% | −5.91 points |
| pair3 | 13 | 83.65% | 85.25% | −1.60 points |
| pair4 | 17 | 87.88% | 81.18% | +6.70 points |
| pair5 | 14 | 83.03% | 88.48% | −5.45 points |
| pair6 | 14 | 95.11% | 89.37% | +5.74 points |
| pair7 | 15 | 89.23% | 93.47% | −4.24 points |

The equal-weight network-pair mean was **−1.82 percentage points** (lower minus
higher triangles). The prespecified exact two-sided sign-flip test gave
**p = 0.4375**; the graph-pair t-based 95% interval was **−6.91 to +3.27
points**. Three pairs favored the lower-triangle network and five favored the
higher-triangle network. The frozen rule required a mean gain of at least
+3 points for lower triangles with p ≤ 0.05, so it failed. The observed
opposite-direction mean is also not statistically persuasive under the planned
test.

## Secondary checks

The independent second scene rendering gave a mean shared-pair difference of
**−1.84 points**, near the primary rendering's −1.82. Averaging each graph's
entire set of 25 unseen pairs, rather than only matched shared pairs, gave
**−3.02 points** on the primary rendering and **−3.20 points** on the second.
This broader comparison is secondary because the tested entity pairs differ
between the two graphs. The mean sparse-minus-clustered difference in unseen
event accuracy was −0.27 points and in familiar four-choice accuracy was
+0.009 points on the primary rendering. These are descriptive secondary
checks, not additional independent replications.

## Interpretation

This experiment does **not** establish the proposed rule that lower clustering
improves role transfer. It also does not establish the reverse rule. Effects
varied substantially among the eight graph pairs. The chosen triangle contrast
changes other network features as well, so even a consistent difference would
have required another experiment to identify triangle count as the cause.

The result is limited to this coded-message, generated-scene architecture,
ten-node degree-four networks, and the specified graph-selection rule. It does
not show that training-network structure is generally irrelevant. The separate
fresh-graph finding that ordered-role auxiliary supervision improves transfer
remains intact; this experiment used the ordinary training objective in both
conditions and tests a different question.

Full per-start data and the frozen test result are in [SUMMARY.json](SUMMARY.json).
The reproduction code is `../rq1_structure_v1.py` and
`../run_rq1_structure_v1.py`.
