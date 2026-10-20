# Same-distance graph control result — 5 October 2026

The fixed three-world, three-seed, two-condition run specified in
[PATH2_PROTOCOL.md](PATH2_PROTOCOL.md) is complete: **18 final checkpoints**.
All stored checkpoint hashes verified, and all 18 familiar four-way accuracies
were at least 99.80%, passing the 95% interpretation gate. The audit verified
three nonisomorphic connected graphs, matched degree and focal triangle
membership, shortest path two for both withheld pairs, absent direct training
edges, exact pairing of canonical rendered training episodes, and identical
held-out scenes between conditions.

## Locked primary outcome

The primary contrast is the shared-two minus shared-one mean held-out
correct-minus-reverse score margin. Negative favours the pair with one shared
training neighbour.

| World | Seed | Shared-one held accuracy | Shared-two held accuracy | Margin contrast |
|---|---:|---:|---:|---:|
| w0 | 9201 | 0.974 | 0.815 | -5.656 |
| w0 | 9202 | 0.669 | 0.995 | +5.872 |
| w0 | 9203 | 0.479 | 0.511 | +0.003 |
| w1 | 9201 | 0.979 | 0.854 | -6.017 |
| w1 | 9202 | 0.985 | 0.561 | -8.290 |
| w1 | 9203 | 0.465 | 0.503 | +0.298 |
| w2 | 9201 | 0.410 | 0.396 | -0.415 |
| w2 | 9202 | 0.772 | 0.504 | -2.456 |
| w2 | 9203 | 0.846 | 0.938 | -0.283 |

World mean contrasts were **+0.073** (w0), **-4.670** (w1), and **-1.051**
(w2). Six of nine paired contrasts were negative. The locked criterion
required all three world means below -0.20 and at least seven of nine paired
contrasts negative. **It did not pass.** The across-pair mean was -1.883, but
that descriptive average does not override the predeclared failure rule.
There is no cross-world inferential p-value from only three worlds.

The result narrows the prior eight-entity finding. Removing the original
shortest-path difference does not yield a stable rule that two shared
neighbours impair transfer more than one. It does show major held-out
variation despite near-perfect familiar learning in all conditions. Other
pair motifs, initialization, and training dynamics remain plausible
contributors. Common-neighbour count alone is not identified because the
matched pairs can differ in other local properties.

## Exploratory analyses, devised after training began

We evaluated **both** withheld pairs in each final model, rather than only
the one mapped to literal labels `(0,1)`. This eliminates between-model
training differences for a pair comparison. The shared-two pair had a lower
correct-minus-reverse margin in **10 of 18** within-model comparisons. The
direction is therefore not stable even in this analysis. Machine-readable
results are in `results/path2/within_model.json`.

Across the 18 models, the existing endpoint query-separation measure had
Spearman correlation **0.806** with held-out correct-minus-reverse margin
(within-world correlations: w0 0.886, w1 0.943, w2 0.771). This is a
post-training, selected exploratory association; it is not evidence that
query separation causally mediates the graph effect. The two measures share
the same learned model and related role distinctions. It is a candidate
prospective predictor or intervention target for a later study.

## Consequence for the research plan

The next claim should focus on **predicting when a model learns a
transferable role distinction**, rather than claiming a universal
one-versus-two-common-neighbour rule. A future experiment should predeclare
a measure of the training relation support graph, test it on new graph
families and seeds, and compare it against simpler pair-motif and
initialization explanations. A language-model or authentic-sentence study
remains a separate bridge; this result does not establish it.

Source: `path2_control.py`; exploratory analysis: `path2_within.py`.
All checkpoint and metrics files are in `results/path2/`.
