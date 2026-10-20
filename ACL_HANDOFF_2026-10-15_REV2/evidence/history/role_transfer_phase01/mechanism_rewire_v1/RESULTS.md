# Fresh rewiring models: internal role-state check (9 October 2026)

The [protocol](PROTOCOL.json) was fixed before inspecting internal states of the 96 already-trained models from the independent [repair study](../rq2_rewire_repair_v1/RESULTS.md). The eight graph families are the independent units; three random starts and 128 focal-pair episodes per model are repeated measurements within each family. No model weights were updated for this analysis.

## Question and method

Can actor and patient identity be read from the model's final candidate-scene state on pairs it never saw together? More importantly, does swapping the role-related part of that state change a held-out decision more than an equally large swap in random directions? A fixed linear readout was fitted to scenes from observed training pairs only, then assessed on all unseen pairs and the focal pair. Its actor-minus-patient directions defined a nine-dimensional intervention subspace. The comparison used five random nine-dimensional subspaces for each model, with each perturbation scaled to the role-swap perturbation's size in that episode.

The pre-specified primary contrast was the additional correct-versus-reversed score-margin loss under the role swap relative to the random controls, in ordered-role versus unordered training, in the high-rewiring condition. Positive values indicate that the ordered model's decision depends more on the measured role directions.

## Results

| High-rewiring models | Unseen-pair probe joint actor/patient accuracy | Focal-pair probe joint accuracy | Ordinary focal binary role accuracy | After role swap | After matched random swap | Extra margin loss, role versus random |
|---|---:|---:|---:|---:|---:|---:|
| Unordered training | 63.3% | 60.8% | 74.0% | 67.8% | 72.0% | 0.155 |
| Ordered-role training | 68.9% | 64.8% | 97.3% | 85.0% | 94.9% | 1.363 |

The primary ordered-minus-unordered contrast was **+1.208 score-margin units**, with an eight-family 95% t interval of **[+0.842, +1.573]** and exact two-sided family-level sign-flip **p = 0.0078125**. All eight family contrasts were positive: +1.363, +1.554, +1.334, +0.641, +0.535, +1.098, +1.826, and +1.311. For context, the same extra margin loss in the sham condition was 0.003 for unordered and 0.836 for ordered training. The [complete summary](SUMMARY.json) contains all model records, raw intervention outcomes, probe scores, checkpoint hashes, and graph-level contrasts.

## Interpretation and limits

The repaired models use a role-related component of the candidate state more strongly on the withheld pair. The linear probe improvement is modest; it does not by itself explain the large behavioral repair. The intervention supplies stronger functional evidence because a targeted state change impairs the decision more than matched random changes. It still does not identify a unique circuit or prove that this state is the only route to the answer. The role directions were estimated from the same trained models being tested, though from observed pairs and independent scene renderings. This is a mechanistic check within the controlled Transformer task, not a result about pretrained LLMs or natural sentences.
