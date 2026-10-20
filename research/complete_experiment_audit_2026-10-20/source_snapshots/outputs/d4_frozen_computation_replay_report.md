# D4 — Frozen D1 Computation Replay Diagnostic

## 1. Executive classification

**D4-B — the actual D1 computation degrades on H–H relations, but the preregistered all-corruption score-margin analysis does not show the predicted endpoint-collapse/event-preservation dissociation.**

D4 is valid. It trained zero parameters and exactly reproduced all 64,704 archived D1 predictions. The literal extracted scores matched `model.listen` with maximum absolute error **0.0**. The checkpoint, full model, message path, and candidate path hashes were unchanged.

The result is more nuanced than either “endpoint information disappears” or “nothing factor-specific happens.” Across all four legitimate endpoint corruptions, H–H source and target margins weakened substantially but remained positive in **83.75–88.74%** of comparisons. Their medians remained 44–75% of the familiar reference medians, so the frozen D4-A collapse thresholds were not met. Event margins also lost roughly half their magnitude, although their signs remained positive in **94.70–98.56%** of H–H comparisons.

A prespecified candidate-set replay and a descriptive inventory breakdown expose the behaviorally relevant failure more sharply: for the remaining **H-identity** endpoint alternative actually present in an H–H Family-S candidate set, endpoint margins are approximately zero and the alternative beats the original candidate approximately 50% of the time. Corruptions to G identities, which leave H–H support, retain large positive margins and dominate the all-corruption average. This post-classification breakdown does not revise D4-B.

![Frozen D1 score-margin summary](../results/d4_frozen_replay_seed_1701/factor_margin_fraction_positive.svg)

## 2. Exact computation recovered

The recovered oracle input is the literal one-hot message

`[source, target, 6 + event, PAD=0]`, with vocabulary size 8.

The frozen message path is:

1. multiply each one-hot token by the learned `sym` matrix;
2. prepend learned `mcls` and add learned `mpos`;
3. apply both frozen message-transformer blocks;
4. take the CLS output through `msg_norm`;
5. L2-normalize the resulting 48-dimensional vector to obtain `q_D1(s,t,e)`.

The candidate path is the frozen three-object `candidate` encoder, including its final LayerNorm, followed by L2 normalization to obtain `h_D1(x)`. The exact score is

`10 × dot(q_D1, h_D1)`.

The secondary descriptive condition computes all six object-slot permutations independently, L2-normalizes each literal candidate output, and takes their arithmetic mean without post-average normalization. Its score is therefore the mean of the six literal D1 scores.

No one-hot semantic query, auxiliary probe, learned compatibility function, layer selection, or checkpoint selection was used.

## 3. Reconstruction validation

| Check | Result |
|---|---:|
| checkpoint SHA-256 | `69e9843ba14ca7d6cd527fbe5daf408fb1b446f7871b58e236c11147a443a77e` |
| checkpoint hash match | pass |
| direct-vs-extracted score maximum absolute error | **0.0** |
| archived prediction agreement | **64,704/64,704 = 100%** |
| archived joint-accuracy maximum absolute error | **0.0** |
| reverse-order slot-marginalization error | **1.1920928955078125e-07** |
| newly optimized parameters | **0** |
| trainable parameters during replay | **0** |

The same full-state hash, `8a326019…ceb9`, occurred before and after D4. The message-state hash remained `80d21192…1ed7`; the candidate-state hash remained `ca7def07…95a`.

## 4. Frozen-data validation

D4 rebuilt the exact D1 evaluation plans and phases: IID S/R phases 301/302; common H–H 311/312; new H–H 321/322; full H–H 331/332. These namespaces are disjoint from D1 training phase 1.

The 48 D1 training meanings contain zero H–H relations; the held support contains exactly 12 H–H meanings. All 64,704 unique candidate-scene realizations had distinct SHA-256 hashes across evaluation blocks and conditions. All 582,336 semantic corruptions were legitimate ordered source–target meanings and valid binary events. No H–H result selected a query, threshold, layer, or checkpoint.

## 5. Exact D1 behavioral replay

| Condition | Original correct/n | Original joint | Six-slot mean correct/n | Six-slot joint | Prediction agreement |
|---|---:|---:|---:|---:|---:|
| familiar S | 8,150/8,160 | 99.8775% | 8,152/8,160 | 99.9020% | 99.9755% |
| familiar R | 8,156/8,160 | 99.9510% | 8,156/8,160 | 99.9510% | 100.0000% |
| common H–H S | 3,863/8,064 | 47.9043% | 3,860/8,064 | 47.8671% | 97.3958% |
| common H–H R | 4,013/8,064 | 49.7644% | 4,012/8,064 | 49.7520% | 97.4330% |
| new H–H S | 4,051/8,064 | 50.2356% | 4,055/8,064 | 50.2852% | 95.9201% |
| new H–H R | 4,107/8,064 | 50.9301% | 4,097/8,064 | 50.8061% | 96.0813% |
| full H–H S | 3,895/8,064 | 48.3011% | 3,903/8,064 | 48.4003% | 97.4454% |
| full H–H R | 4,011/8,064 | 49.7396% | 4,016/8,064 | 49.8016% | 97.7803% |

The original-path replay is exact, not merely within sampling tolerance.

## 6. Learned query geometry

The 60 literal normalized D1 query vectors and all 1,770 pairwise comparisons were preserved. Same-factor labels are associated with somewhat greater geometric similarity, but no clean factorized geometry follows:

| Pairwise condition | Same mean cosine | Different mean cosine |
|---|---:|---:|
| source | .14755 | .04472 |
| target | .25939 | .02459 |
| event | .08082 | .04067 |

Single-factor substitutions had mean pairwise query cosine .36923, compared with −.02206 for two-factor changes and .04194 for three-factor changes.

The displacement-vector test directly rejects context-independent constituent translations. Mean cross-context cosine for a fixed identity substitution was **.3340 for source**, **.5029 for target**, and only **.0715 for event**. Individual groups included negative pairwise displacement cosines.

Endpoint identity class strongly affected displacement magnitude:

| Substitution | Source mean norm | Target mean norm |
|---|---:|---:|
| H↔H identity | **.03122** | **.04254** |
| H↔G identity | 1.14095 | 1.26639 |
| G↔G identity | 1.16781 | 1.23264 |

Thus changing one H identity to another barely moves the final D1 query compared with changes involving G identities. This is a property of the learned query geometry; it is not evidence for a symbolic register or for a particular internal component.

## 7. Familiar counterfactual margins

The primary endpoint reference includes every corruption whose resulting query belongs to the 48-meaning D1 training support. Event reversal uses all examples.

| Factor | Positive/n | Fraction positive, 95% block CI | Mean | Median |
|---|---:|---:|---:|---:|
| source | 50,403/55,488 | 90.8359% [90.4818, 91.1030] | 5.6039 | 6.6094 |
| target | 50,725/55,488 | 91.4162% [91.0104, 91.7182] | 6.0777 | 7.2940 |
| event | 16,291/16,320 | 99.8223% [99.7549, 99.8836] | 10.3935 | 10.7996 |

All-corruption familiar source/target rates were 60,188/65,280 = 92.1998% and 60,511/65,280 = 92.6945%. The preregistered 95% familiar endpoint gate nevertheless failed.

The aggregate masks substantial directional heterogeneity:

| Familiar stratum | Source mean / positive | Target mean / positive | Event mean / positive |
|---|---:|---:|---:|
| H→G | 3.4794 / 74.9489% | 7.7330 / 99.3056% | 10.0084 / 99.4485% |
| G→H | 7.1557 / 99.4894% | 3.8932 / 77.0425% | 10.6829 / 99.9796% |
| G→G | 6.6154 / 99.5060% | 7.0954 / 99.2379% | 10.4651 / 99.9847% |

The changed endpoint is weaker when it occupies the H side of a cross-group relation, despite near-perfect familiar four-way behavior.

## 8. H–H counterfactual margins: all legitimate corruptions

Each endpoint row contains four corruptions per example; event has one.

| Split | Factor | Positive/n | Fraction positive, 95% block CI | Mean | Median |
|---|---|---:|---:|---:|---:|
| common | source | 54,027/64,512 | 83.7472% [83.5069, 83.9674] | 2.5102 | 2.9144 |
| common | target | 55,327/64,512 | 85.7623% [85.5422, 85.9763] | 3.7907 | 4.7314 |
| common | event | 15,273/16,128 | 94.6987% [94.3514, 95.0397] | 4.1730 | 4.6927 |
| new | source | 56,842/64,512 | 88.1107% [87.9325, 88.3076] | 3.4934 | 4.2214 |
| new | target | 57,248/64,512 | 88.7401% [88.5494, 88.9478] | 4.4249 | 5.4706 |
| new | event | 15,895/16,128 | 98.5553% [98.3629, 98.7413] | 5.0765 | 5.4656 |
| full | source | 54,925/64,512 | 85.1392% [84.9113, 85.3547] | 2.7762 | 3.2938 |
| full | target | 55,763/64,512 | 86.4382% [86.2258, 86.6443] | 3.9384 | 5.0212 |
| full | event | 15,405/16,128 | 95.5171% [95.2009, 95.8395] | 4.4092 | 4.9803 |

Endpoint margins clearly weaken, but their all-corruption distributions are not centered near zero. Source-positive fractions drop only 2.7–7.1 percentage points from the frozen familiar reference; target drops 2.7–5.7 points. These miss the preregistered ≥20-point collapse requirement. Event-positive fractions drop 1.3–5.1 points and pass the event sign-preservation threshold, but event mean margins also decline by 51–60%.

## 9. Endpoint corruption inventory breakdown

For an H–H meaning, exactly one endpoint replacement is the remaining H identity and three replacements are G identities. The following descriptive breakdown was computed from every preserved row and was not used to alter the category:

| Split | Factor | Alternative inventory | Mean | Median | Positive/n |
|---|---|---|---:|---:|---:|
| common | source | H | **.0010** | **.0004** | 8,237/16,128 = 51.0727% |
| common | target | H | **.0049** | **.0050** | 8,769/16,128 = 54.3713% |
| common | source | G | 3.3466 | 3.6472 | 45,790/48,384 = 94.6387% |
| common | target | G | 5.0527 | 5.5532 | 46,558/48,384 = 96.2260% |
| new | source | H | **.0021** | **.0007** | 8,458/16,128 = 52.4430% |
| new | target | H | **.0030** | **.0032** | 8,864/16,128 = 54.9603% |
| new | source | G | 4.6571 | 4.7078 | 48,384/48,384 = 100% |
| new | target | G | 5.8989 | 5.8477 | 48,384/48,384 = 100% |
| full | source | H | **.0016** | **.0006** | 8,354/16,128 = 51.7981% |
| full | target | H | **.0039** | **.0042** | 8,777/16,128 = 54.4209% |
| full | source | G | 3.7011 | 3.9871 | 46,571/48,384 = 96.2529% |
| full | target | G | 5.2499 | 5.6600 | 46,986/48,384 = 97.1106% |

The within-H endpoint contrast is approximately centered at zero in every H–H condition. Its block-bootstrap positive-fraction intervals are 50.36–53.19% for source and 53.65–55.74% for target across the three inventories. The positive all-corruption averages arise because three quarters of endpoint corruptions substitute a G identity and remain strongly discriminated.

## 10. Counterfactual candidate-set replay

This analysis uses only cases where the corresponding semantic alternative is one of the unchanged four candidates.

| Split | Factor | Alternative beats original after query change | Final choice redirects to alternative |
|---|---|---:|---:|
| familiar | source | 4,074/4,080 = 99.8529% | 4,072/4,080 = 99.8039% |
| familiar | target | 4,077/4,080 = 99.9265% | 4,074/4,080 = 99.8529% |
| familiar | event | 12,236/12,240 = 99.9673% | 12,232/12,240 = 99.9346% |
| common H–H | source | 2,018/4,032 = 50.0496% | 1,343/4,032 = 33.3085% |
| common H–H | target | 2,027/4,032 = 50.2728% | 1,353/4,032 = 33.5565% |
| common H–H | event | 11,539/12,096 = 95.3952% | 6,520/12,096 = 53.9021% |
| new H–H | source | 2,028/4,032 = 50.2976% | 1,348/4,032 = 33.4325% |
| new H–H | target | 2,037/4,032 = 50.5208% | 1,404/4,032 = 34.8214% |
| new H–H | event | 12,082/12,096 = 99.8843% | 6,824/12,096 = 56.4153% |
| full H–H | source | 2,017/4,032 = 50.0248% | 1,352/4,032 = 33.5317% |
| full H–H | target | 2,021/4,032 = 50.1240% | 1,335/4,032 = 33.1101% |
| full H–H | event | 11,565/12,096 = 95.6101% | 6,539/12,096 = 54.0592% |

Changing the H endpoint query fails to reorder the two semantically corresponding H–H candidates. Event reversal usually reverses that pairwise ordering, but endpoint ambiguity among the other candidates prevents reliable final selection. This is the actual frozen D1 decision rule, not an auxiliary decoder.

## 11. Primary factor-specific comparison

The broad preregistered hypothesis is only partly supported:

- Familiar candidate-set source, target, and event redirection is essentially perfect.
- H–H endpoint discrimination against the in-inventory H alternative collapses to chance and its raw margin centers near zero.
- H–H endpoint discrimination against G alternatives remains strong, so the preregistered all-valid-corruption endpoint distribution does **not** collapse.
- H–H event-margin signs remain strongly positive, but event magnitude is not stable: its mean falls from 10.3935 to 4.1730–5.0765.

This is why the result is D4-B rather than D4-A. The earlier D1–D3 factor-specific account needs refinement from “endpoint discrimination globally disappears” to “within-H endpoint contrasts fail while out-of-inventory endpoint contrasts and event ordering remain much stronger.”

## 12. Original path versus slot marginalization

Six-slot averaging barely changes the score geometry:

| Split | Source margin-sign agreement | Target agreement | Event agreement |
|---|---:|---:|---:|
| familiar | 99.7381% | 99.6431% | 99.9877% |
| common H–H | 99.29% | 99.49% | 99.55% |
| new H–H | 99.5164% | 99.5939% | 99.7892% |
| full H–H | 99.3707% | 99.4249% | 99.5846% |

Margin correlations are .9976–.9997. Aggregate original-versus-marginal fraction-positive differences are below .09 percentage points. Four-way prediction agreement is 95.92–100%, and joint accuracy changes by at most .124 percentage points. The within-H endpoint collapse, event sign preservation, and D4-B classification therefore do not depend materially on object-slot realization.

## 13. Controls and validity gates

All critical gates passed:

- exact checkpoint, config, and D1 metadata;
- zero trainable and optimized parameters;
- exact direct-score reconstruction;
- exact archived-prediction replay;
- unchanged full/message/candidate parameter hashes;
- valid oracle messages for every corruption;
- no H–H training meanings;
- disjoint deterministic evaluation namespaces;
- no cross-block candidate-geometry duplicates;
- verified six-permutation accounting and aggregation-order invariance.

The first shell launch selected a broken system Python and failed while importing PyTorch, before model loading or output creation. The unchanged script was then launched once with the same Python 3.12/PyTorch 2.8 runtime used by D1–D3. This operational failure produced no scientific result and triggered no result-contingent change.

## 14. Statistical uncertainty

All reported confidence intervals are percentile intervals from 2,000 deterministic block-cluster bootstrap replicates. The Task-v2 candidate block—not an individual corruption row—is the resampling unit. The raw bootstrap distributions are preserved.

The key within-H endpoint estimates are precise: common-H–H source 51.07% [50.36, 51.83], target 54.37% [53.65, 55.08]; new source 52.44% [51.70, 53.19], target 54.96% [54.16, 55.74]; full source 51.80% [51.05, 52.49], target 54.42% [53.69, 55.15]. These intervals exclude the all-corruption endpoint-positive rates by large margins.

## 15. What D4 establishes—and does not

D4 establishes properties of the actual trained D1 score computation:

- D1’s familiar four-way behavior and semantic corruption response are genuine and exactly replayable.
- The final learned query barely separates H identities when one H endpoint replaces another.
- Corresponding H–H endpoint alternatives are not reliably reordered, while event pairwise ordering remains strong.
- Broad endpoint corruptions to G identities remain discriminable, making “global endpoint information loss” false.
- Slot marginalization is not the source of the phenomenon.
- Learned constituent substitutions are strongly context-dependent in final-query geometry.

D4 does **not** establish causal necessity, locate the responsible layer/head/MLP, identify neurons or registers, demonstrate symbolic variables or grammar, or generalize beyond this checkpoint, seed, architecture, and synthetic task.

## 16. Implications for D1, D2-R, and D3

D1’s near-chance H–H joint accuracy is compatible with a sharply selective score-geometry failure, not with message neglect. D2-R was correct that independent endpoint probes do not capture the operative geometry. D3 correctly exposed near-zero H–H endpoint discrimination for the candidate-matched alternative, but its broad “endpoint collapse” language was too coarse: D4 shows that the collapse is specific to within-H substitutions, whereas H→G/G→H counterfactuals remain strongly separated.

The query geometry supplies a direct descriptive correlate: H↔H endpoint substitutions move `q_D1` by only .03–.04 in norm, compared with roughly 1.14–1.27 for substitutions involving G identities. This is not yet a causal explanation.

## 17. Exactly one recommended next experiment

Run one preregistered **frozen D1 message-path activation-patching experiment** using matched familiar and H–H oracle-query pairs. Patch only the source-token or target-token contribution at prespecified residual-stream boundaries, include event-token and sham-position controls, and measure restoration or disruption of the exact candidate-pair score difference. Correct across all declared layers and positions; do not select components from D4 geometry or tune donors after observing outcomes.

This experiment was **not run**.

## 18. Reproducibility artifacts

The D4 bundle contains the frozen checkpoint, preregistration, executable, source hashes, pre-replay and final validity records, raw normalized and pre-normalized query vectors, all candidate score inputs for both paths, all original and counterfactual scores, every margin, four-way scores and predictions, candidate-set redirection rows, semantic mappings, query pairwise geometry, displacement vectors, stratified summaries, 2,000-replicate bootstrap outputs, the figure, logs, and a SHA-256 manifest.

Environment: Python 3.12.14, PyTorch 2.8.0, NumPy 2.3.5, macOS arm64, CPU. The workspace is not a Git checkout, so reproducibility uses exact SHA-256 source-file hashes rather than a commit identifier.
