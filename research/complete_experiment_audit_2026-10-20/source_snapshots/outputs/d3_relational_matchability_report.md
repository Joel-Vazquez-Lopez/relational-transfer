# D3 — Oracle-Conditioned Relational Matchability Diagnostic

## Executive classification

**D3-D — SIMPLE RELATIONAL PROBES DO NOT EXPLAIN D1.**

D3 is valid. The exact D1 checkpoint and D2-R permutation-marginalized candidate representation were frozen; the query was a zero-parameter constituent concatenation; training contained all and only the 48 Task-v2 meanings and no H–H query or positive; H–H evaluation contained query-support membership equally for MATCH and NON-MATCH; every declared marginal was exactly balanced; all shortcut, replacement, and null controls passed; and no train/test or ranking/base scene overlap occurred.

The 673-parameter primary bilinear model reached **92.2108% familiar balanced MATCH accuracy** but only **76.8627% Family-S** and **60.3922% Family-R** familiar four-way ranking—far below D1's approximately 100%. Its common H–H balanced accuracy was 63.6533%; ranking was 43.3532% S and 46.9742% R.

The 1,025-parameter secondary MLP came close to explaining familiar ranking (96.2745% S; 99.7549% R) and achieved 96.8424% familiar balanced accuracy. It nevertheless missed the preregistered familiar source-only and target-only mismatch floors: **87.8255% and 88.3464%**, versus 90% required. It also collapsed on common H–H ranking to 44.4444% S and 46.8254% R. Because neither fixed compatibility model passed every familiar gate, the prespecified classification is D3-D, not D3-C. No threshold was relaxed.

This valid result does not show that relational information is absent. It shows that the two generic compatibility functions do not fully account for the exact trained D1 computation under the frozen decision rules.

## 1. Scientific question

D3 tested whether a candidate scene's frozen pooled representation can be matched to a deterministic constituent query `(source,target,event)` without requiring those identities to be independently decoded. The primary model was bilinear and the sole secondary model was a fixed 16-unit MLP. This is a recoverability test, not evidence that D1 uses these variables internally.

## 2. Frozen D1 verification

| Check | Result |
|---|---|
| D1 checkpoint SHA-256 | `69e9843ba14ca7d6cd527fbe5daf408fb1b446f7871b58e236c11147a443a77e` |
| exact hash match | pass |
| D1 encoder trainable | no |
| speaker or emergent message used | no |
| Task v2 modified | no |
| candidate layer searched | no |

The candidate encoder state hash was checked before and after extraction/training; no D1 parameter entered an optimizer.

## 3. Candidate representation

For all six object-slot permutations, D3 computed

`h_pi(x) = F.normalize(model.candidate(pi(x)), dim=-1)`

and used the unrenormalized 48-dimensional mean

`h_bar(x) = mean_pi h_pi(x)`.

Underlying-slot and aggregation-order maximum discrepancies were both `5.960464477539063e-08`, below the frozen D2-R tolerance of `1e-6`.

## 4. Oracle query representation

The query vector was the deterministic 14-dimensional concatenation

`[one_hot(source,6), one_hot(target,6), one_hot(event,2)]`.

It had zero trainable parameters, no whole-meaning ID, no pair lookup, and no H–H-specific component. The bilinear score was `q^T W h + b`; only the `14×48` matrix and scalar bias were trained, for **673 parameters**.

The secondary model was exactly `[q,h] (62) -> Linear(62,16) -> GELU -> Linear(16,1)`, with **1,025 parameters**.

## 5. MATCH/NON-MATCH construction

Every candidate scene generated seven paired positive/negative comparisons. For each negative type, one exact positive query was duplicated with the same type tag, yielding equal MATCH/NON-MATCH counts and making mismatch type itself uninformative.

| Split | Base scenes | Pair rows | MATCH | NON-MATCH |
|---|---:|---:|---:|---:|
| train | 1,536 | 21,504 | 10,752 | 10,752 |
| validation | 768 | 10,752 | 5,376 | 5,376 |
| familiar | 1,536 | 21,504 | 10,752 | 10,752 |
| common H–H | 576 | 8,064 | 4,032 | 4,032 |
| new H–H | 576 | 8,064 | 4,032 | 4,032 |
| full H–H | 576 | 8,064 | 4,032 | 4,032 |

The seven negatives were source-only, target-only, event-only, source+target, source+event, target+event, and full mismatch. Each corruption was a deterministic bijection over the relevant 48-meaning or 12-meaning support. Therefore the complete query-meaning distribution, not merely individual marginals, was identical between labels.

Training negative queries were restricted to the same 48 non-H–H meanings as positive queries. Training contained no H–H query under either label. Every H–H evaluation query was H–H under both labels. Thus the classifier could not learn “H–H query implies NON-MATCH.”

## 6. Leakage and shortcut audit

The following MATCH and NON-MATCH distributions were exactly identical in every split: candidate source, target, event, H/G roles, family; query source, target, event, H/G roles; and paired mismatch type. Candidate scenes appeared equally often with each label.

The preregistered restricted semantic marginal oracle, which received candidate/query identity and event marginals, distractor, family, H/G bits, and mismatch tag without interactions, achieved:

- balanced accuracy: **50.2046%**;
- AUROC: **0.498290**.

This passed the 55% ceiling before scientific training. There were zero base-scene hash overlaps across splits and zero overlaps between ranking candidate scenes and binary-match scenes.

## 7. Familiar results by semantic regime

The primary bilinear familiar aggregate was **19,829/21,504 = 92.2108%**, with balanced accuracy 92.2108%, AUROC 0.950495, positive accuracy 96.8750%, and negative accuracy 87.5465%.

| Model | Stratum | Correct/n | Balanced accuracy | AUROC | MATCH accuracy | NON-MATCH accuracy |
|---|---|---:|---:|---:|---:|---:|
| bilinear | H→G | 7,447/8,064 | 92.3487% | .971632 | 96.7014% | 87.9960% |
| bilinear | G→H | 7,446/8,064 | 92.3363% | .980904 | 96.0069% | 88.6657% |
| bilinear | G→G | 4,936/5,376 | 91.8155% | .866281 | 98.4375% | 85.1935% |
| secondary | H→G | 7,696/8,064 | 95.4365% | .988864 | 95.8333% | 95.0397% |
| secondary | G→H | 7,799/8,064 | 96.7138% | .994390 | 97.9167% | 95.5109% |
| secondary | G→G | 5,330/5,376 | 99.1443% | .998712 | 98.9583% | 99.3304% |

No semantic stratum was hidden by the aggregate. The bilinear model missed the 95% familiar balanced-accuracy gate in all three strata; the secondary model exceeded it in each stratum, but its specific source-only/target-only counters remained below their separate 90% gates.

## 8. Common/new/full H–H results

| Model | Split | Correct/n | Balanced accuracy | AUROC | MATCH accuracy | NON-MATCH accuracy |
|---|---|---:|---:|---:|---:|---:|
| bilinear | common | 5,133/8,064 | 63.6533% | .675711 | 84.7222% | 42.5843% |
| bilinear | new | 5,017/8,064 | 62.2148% | .690085 | 100.0000% | 24.4296% |
| bilinear | full | 5,053/8,064 | 62.6612% | .671608 | 87.6736% | 37.6488% |
| secondary | common | 5,206/8,064 | 64.5585% | .693129 | 72.5694% | 56.5476% |
| secondary | new | 5,835/8,064 | 72.3586% | .759653 | 87.5000% | 57.2173% |
| secondary | full | 5,500/8,064 | 68.2044% | .722701 | 79.5139% | 56.8948% |

The H–H failure is mainly counterfactual rejection: true H–H matches often score positive, but wrong endpoint queries are also accepted.

## 9. Mismatch-type results

Each negative type has `n=1,536` familiar and `n=576` in each H–H split. Exact-match counts aggregate seven replicated positive comparisons per base scene.

### Primary bilinear

| Class | Familiar | Common | New | Full |
|---|---:|---:|---:|---:|
| exact MATCH | 10,416/10,752 = 96.8750% | 3,416/4,032 = 84.7222% | 4,032/4,032 = 100.0000% | 3,535/4,032 = 87.6736% |
| source-only | 1,311/1,536 = 85.3516% | 92/576 = 15.9722% | 1/576 = 0.1736% | 74/576 = 12.8472% |
| target-only | 1,347/1,536 = 87.6953% | 94/576 = 16.3194% | 2/576 = 0.3472% | 71/576 = 12.3264% |
| event-only | 685/1,536 = 44.5964% | 360/576 = 62.5000% | 240/576 = 41.6667% | 319/576 = 55.3819% |
| source+target | 1,532/1,536 = 99.7396% | 105/576 = 18.2292% | 3/576 = 0.5208% | 70/576 = 12.1528% |
| source+event | 1,501/1,536 = 97.7214% | 344/576 = 59.7222% | 248/576 = 43.0556% | 335/576 = 58.1597% |
| target+event | 1,506/1,536 = 98.0469% | 367/576 = 63.7153% | 242/576 = 42.0139% | 326/576 = 56.5972% |
| full mismatch | 1,531/1,536 = 99.6745% | 355/576 = 61.6319% | 249/576 = 43.2292% | 323/576 = 56.0764% |

### Secondary MLP

| Class | Familiar | Common | New | Full |
|---|---:|---:|---:|---:|
| exact MATCH | 10,472/10,752 = 97.3958% | 2,926/4,032 = 72.5694% | 3,528/4,032 = 87.5000% | 3,206/4,032 = 79.5139% |
| source-only | 1,349/1,536 = 87.8255% | 145/576 = 25.1736% | 79/576 = 13.7153% | 120/576 = 20.8333% |
| target-only | 1,357/1,536 = 88.3464% | 179/576 = 31.0764% | 111/576 = 19.2708% | 148/576 = 25.6944% |
| event-only | 1,518/1,536 = 98.8281% | 456/576 = 79.1667% | 513/576 = 89.0625% | 473/576 = 82.1181% |
| source+target | 1,536/1,536 = 100.0000% | 156/576 = 27.0833% | 100/576 = 17.3611% | 139/576 = 24.1319% |
| source+event | 1,524/1,536 = 99.2188% | 450/576 = 78.1250% | 517/576 = 89.7569% | 479/576 = 83.1597% |
| target+event | 1,536/1,536 = 100.0000% | 439/576 = 76.2153% | 478/576 = 82.9861% | 465/576 = 80.7292% |
| full mismatch | 1,533/1,536 = 99.8047% | 455/576 = 78.9931% | 509/576 = 88.3681% | 470/576 = 81.5972% |

## 10. Counterfactual score margins

Margins are bilinear logits for correct query minus the indicated single-factor corruption.

| Split/stratum | Factor | Mean ± SD | Median | 2.5%, 97.5% | Positive/n |
|---|---|---:|---:|---:|---:|
| familiar H→G | source | 3.426 ± 2.514 | 3.536 | -0.902, 8.426 | 545/576 |
| familiar H→G | target | 5.854 ± 1.132 | 5.877 | 3.349, 7.629 | 573/576 |
| familiar H→G | event | 2.295 ± 1.093 | 2.385 | -0.486, 4.058 | 553/576 |
| familiar G→H | source | 5.261 ± 1.096 | 5.506 | 2.168, 6.618 | 575/576 |
| familiar G→H | target | 4.044 ± 2.307 | 5.031 | 0.406, 7.252 | 564/576 |
| familiar G→H | event | 2.188 ± 1.131 | 2.428 | -0.716, 3.855 | 551/576 |
| familiar G→G | source | 6.833 ± 1.597 | 6.871 | 3.396, 9.611 | 384/384 |
| familiar G→G | target | 7.019 ± 1.424 | 7.161 | 3.580, 9.087 | 383/384 |
| familiar G→G | event | -2.088 ± 1.189 | -2.167 | -4.044, 0.516 | 18/384 |
| common H–H | source | 0.013 ± 3.043 | 0.057 | -5.113, 5.127 | 294/576 |
| common H–H | target | 0.098 ± 3.518 | -0.026 | -5.773, 6.095 | 287/576 |
| common H–H | event | 3.119 ± 1.816 | 3.403 | -1.079, 6.052 | 530/576 |
| new H–H | source | -0.040 ± 3.647 | 0.031 | -5.619, 5.442 | 288/576 |
| new H–H | target | 0.016 ± 4.581 | -0.109 | -6.599, 6.661 | 287/576 |
| new H–H | event | 3.808 ± 1.750 | 3.781 | 0.291, 6.584 | 564/576 |
| full H–H | source | 0.077 ± 3.178 | 0.105 | -5.285, 5.306 | 293/576 |
| full H–H | target | 0.001 ± 3.832 | -0.097 | -6.032, 6.304 | 280/576 |
| full H–H | event | 3.283 ± 1.774 | 3.389 | -0.905, 6.040 | 550/576 |

Endpoint margins collapse to distributions centered almost exactly at zero on H–H. Event margins remain strongly positive on most H–H scenes. On familiar G→G, the bilinear event margin reverses sign, explaining the poor event-only mismatch result despite strong multiple-factor rejection.

## 11. Four-way D1-style ranking

No ranking-specific training occurred. Each compatibility model scored all four frozen candidate representations for the oracle query.

| Model | Split | Family S | Family R |
|---|---|---:|---:|
| bilinear | familiar | 1,568/2,040 = 76.8627% | 1,232/2,040 = 60.3922% |
| bilinear | common | 874/2,016 = 43.3532% | 947/2,016 = 46.9742% |
| bilinear | new | 992/2,016 = 49.2063% | 1,058/2,016 = 52.4802% |
| bilinear | full | 912/2,016 = 45.2381% | 955/2,016 = 47.3710% |
| secondary | familiar | 1,964/2,040 = 96.2745% | 2,035/2,040 = 99.7549% |
| secondary | common | 896/2,016 = 44.4444% | 944/2,016 = 46.8254% |
| secondary | new | 989/2,016 = 49.0575% | 994/2,016 = 49.3056% |
| secondary | full | 910/2,016 = 45.1389% | 949/2,016 = 47.0734% |

The secondary model nearly reconstructs D1's familiar ranking but reproduces its H–H collapse. It is still not a D3-C pass because the separately preregistered source-only and target-only familiar mismatch gates failed.

## 12. Secondary compatibility model

The secondary model's familiar balanced accuracy was **20,825/21,504 = 96.8424%**, AUROC .994453, MATCH accuracy 97.3958%, and NON-MATCH accuracy 96.2891%. Common H–H balanced accuracy was 64.5585%, and common S/R ranking was 44.4444%/46.8254%.

Thus modest nonlinearity is enough to approximate the familiar four-way behavior, but not to compose H–H endpoints. The narrowly failed constituent-specific familiar gates prevent a confirmatory “nonlinear relational matchability” classification.

## 13. Query-only, candidate-only, marginal, and replacement controls

| Control | Familiar BA/AUROC | Common BA/AUROC | New BA/AUROC | Full BA/AUROC |
|---|---:|---:|---:|---:|
| query-free (`h` only) | 50.0000% / .499999 | 50.0000% / .499999 | 50.0000% / .500001 | 50.0000% / .499988 |
| candidate-free (`q` only) | 50.0000% / .500002 | 50.0000% / .500000 | 50.0000% / .500000 | 50.0000% / .500000 |
| additive semantic marginals | 52.3251% / .491831 | 52.6786% / .486332 | 54.4643% / .485154 | 53.1250% / .486961 |
| random query replacement | 49.5768% / .493111 | 50.4588% / .504429 | 50.1984% / .494951 | 50.5704% / .508685 |

Every balanced accuracy and AUROC remained below the preregistered 55% ceiling.

## 14. Randomized-label null distribution

Thirty-two predetermined bilinear label randomizations completed per split. Empirical upper-tail probability was `(1 + count(null >= observed))/33`.

| Split | Observed BA | Null mean ± SD | Null 2.5%, median, 97.5% | Null ≥ observed | p |
|---|---:|---:|---:|---:|---:|
| familiar | 92.2108% | 50.2614 ± 3.0290% | 45.1511, 50.9161, 55.6131% | 0/32 | .0303 |
| common | 63.6533% | 49.5005 ± 1.9835% | 45.7543, 49.3738, 53.4217% | 0/32 | .0303 |
| new | 62.2148% | 49.2230 ± 2.8680% | 44.1295, 49.7892, 54.5626% | 0/32 | .0303 |
| full | 62.6612% | 49.4811 ± 2.1093% | 45.4312, 49.5288, 52.8224% | 0/32 | .0303 |

The bilinear signal is real relative to its null, but it is not strong enough to meet the task-level thresholds.

## 15. Statistical confidence

Bootstrap units were base candidate scenes for binary matching and four-target candidate blocks for ranking. Common-H–H primary intervals use two-sided Bonferroni correction across binary balanced accuracy, S ranking, and R ranking.

| Endpoint | Point | 95% interval | Clusters |
|---|---:|---:|---:|
| familiar binary balanced | 92.2108% | [91.6805, 92.7178] | 1,536 |
| common binary balanced | 63.6533% | [62.0864, 65.2948] | 576 |
| familiar S ranking | 76.8627% | [74.5588, 78.9706] | 510 |
| familiar R ranking | 60.3922% | [56.3725, 64.0699] | 510 |
| common S ranking | 43.3532% | [41.3678, 45.0893] | 504 |
| common R ranking | 46.9742% | [45.0397, 48.7934] | 504 |
| new S/R ranking | 49.2063% / 52.4802% | [47.8175, 50.6944] / [50.6944, 54.2163] | 504 each |
| full S/R ranking | 45.2381% / 47.3710% | [43.7500, 46.8762] / [45.9821, 48.6607] | 504 each |

Exact bootstrap indices are preserved.

## 16. Validity gates

| Gate | Result |
|---|---|
| exact D1 checkpoint | pass |
| frozen candidate encoder | pass |
| exact D2-R representation and tolerance | pass |
| zero-parameter constituent query, no pair lookup | pass |
| all and only 48 training meanings | pass |
| zero H–H query/positive training | pass |
| H–H support balanced across evaluation labels | pass |
| every declared marginal identical across labels | pass |
| pretraining restricted shortcut ≤55% | pass |
| query-free, candidate-free, marginal controls ≤55% | pass |
| random replacement ≤55% | pass |
| no geometry overlap | pass |
| 32 null randomizations complete | pass |
| no H–H selection | pass |

D3 is valid.

## 17. Scientific classification

The primary model failed familiar strong matching and ranking. The secondary model passed aggregate familiar matching and both ranking thresholds, but its source-only and target-only rejection accuracies—87.8255% and 88.3464%—fell below the frozen 90% floor. Therefore neither model satisfied the complete familiar criterion, mapping the result to **D3-D**.

This is a deliberately conservative threshold result. The secondary model's near-D1 familiar ranking is scientifically informative but does not change the classification.

## 18. Comparison with D1 and D2-R

| Experiment | Familiar | H–H | Interpretation |
|---|---|---|---|
| D1 | ≈100% four-way | ≈50% joint; event 98–99% | exact trained listener succeeds familiar, fails H–H endpoints |
| D2-R | independent endpoint probes <95% familiar | endpoints poor; event not uniformly strong | independent constituent accessibility insufficient |
| D3 bilinear | 76.86% S / 60.39% R ranking | 43.35% / 46.97% common | simple bilinear relational match insufficient even familiar |
| D3 secondary | 96.27% S / 99.75% R ranking | 44.44% / 46.83% common | weak nonlinear model nearly explains familiar ranking but remains pair-specific |

Relational compatibility is a much better descriptive account of D1 than independent identity decoding when modest nonlinearity is allowed. It still does not fully pass the familiar counterfactual gates, and neither compatibility model composes H–H endpoints.

## 19. What is now ruled out

D3 rules out query-only, candidate-only, additive marginal, answer-position, H–H-support membership, random-query pairing, label-randomization, object-slot realization, and geometry-overlap shortcuts as explanations of its results. It also rules out the hypothesis that a single constituent-linear bilinear interaction is sufficient to reproduce D1's near-perfect familiar behavior under these thresholds.

## 20. What remains unresolved

The exact source of the D1/D3 gap remains unresolved. Possibilities include the actual D1 oracle-message transformer producing a more useful learned query geometry than the frozen one-hot constituent query; the exact cosine score and slot-specific candidate path differing from permutation-marginalized matching; or distributed pair-specific structure that the fixed secondary model only approximates. D3 does not establish which component is responsible, and it does not establish an internal mechanism.

## 21. Exactly one recommended next experiment

Run one preregistered **frozen D1 computation replay diagnostic**. Without training any parameters, extract the exact normalized D1 oracle-message CLS query and exact candidate score inputs, reproduce the final listener's dot-product scores numerically, and evaluate familiar/H–H four-way ranking plus source-, target-, and event-corruption score margins under both the original slot-specific path and the already validated six-slot marginalization. This interrogates the actual D1 message–candidate computation, as required by D3-D, while separating learned query geometry from slot marginalization. Do not perform architecture search, activation patching, or causal interpretation.

That experiment was not run.

## Reproducibility record

The bundle contains the frozen D1 checkpoint; D3 config and executable; base scenes, representations, query-pair membership, exact balance audits, real and null checkpoints, raw binary and ranking predictions, mismatch tables, margin distributions, familiar-stratum tables, controls, null distributions, bootstrap indices, learning traces, aggregate metrics, run log, research log, and SHA-256 manifest.

Environment: Python 3.12.14, PyTorch 2.8.0, NumPy 2.3.5, macOS 26.6.2 arm64, CPU. Scientific training/evaluation wall time after validation was 31.8397 seconds.
