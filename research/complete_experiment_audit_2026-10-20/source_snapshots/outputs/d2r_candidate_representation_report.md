# D2-R — Candidate Representation Validity-Repair Replication

## Executive classification

**D2R-D — FAMILIAR REPRESENTATION NOT PROBE-ACCESSIBLE.**

D2-R is valid. The exact D1 checkpoint and 48-dimensional representation boundary were preserved; all 48 and only the Task-v2 training meanings appeared in probe training; active H–H leakage and scene overlap were zero; the encoder remained frozen; six-permutation marginalization was invariant within the declared tolerance; all 96 predetermined null probes completed; and no H–H result selected a checkpoint or hyperparameter.

Despite the repair, primary linear decoding on genuinely familiar Task-v2 meanings was only **76.8663% for source (1,771/2,304)** and **87.4566% for target (2,015/2,304)**, both below the preregistered 95% requirement. Event was **78.7760% (1,815/2,304)**. The fixed width-16 MLP reached 76.9097%, 93.7500%, and 98.4375%, so it also failed to make both familiar endpoints strongly recoverable.

Common H–H linear decoding was source **14.9306%**, target **43.8657%**, and event **68.1713%**. Only the common target result clearly exceeded its corresponding randomized-label null; common source and event did not. This does not establish absence of endpoint information. It establishes that independent source/target identity probes are insufficient for localizing how the near-perfect D1 listener solves familiar candidate discrimination.

No Task-v2, D1, or D2 model was changed or rerun. No speaker, message, layer search, encoder fine-tuning, architecture search, mechanistic interpretation, or result-contingent restart was used.

## 1. Scientific question

D2-R asked whether source identity, target identity, and event are recoverable from the frozen D1 candidate representation by a linear classifier trained on all Task-v2 training meanings and evaluated on unseen H–H relations. The 16-unit MLP was a fixed secondary decoder. Decodability is evidence only about the selected representation and decoder class, not internal use or semantic variables.

## 2. Frozen candidate representation

The extraction boundary was exactly:

`h_pi(x) = F.normalize(model.candidate(pi(x)), dim=-1)`

followed by the preregistered nuisance marginalization:

`h_bar(x) = (1/6) * sum_pi h_pi(x)`.

`h_bar` was not renormalized after averaging.

| Property | Result |
|---|---|
| D1 checkpoint SHA-256 | `69e9843ba14ca7d6cd527fbe5daf408fb1b446f7871b58e236c11147a443a77e` |
| checkpoint hash match | pass |
| representation dimension | 48 |
| vectors per semantic scene | one averaged pooled CLS vector |
| slot permutations enumerated | 6 exactly |
| aggregation-order maximum error | `5.960464477539063e-08` |
| underlying-slot-permutation maximum error | `5.960464477539063e-08` |
| declared absolute tolerance | `1e-6` |
| encoder state hash before/after validation | identical: `0a0660c0eee13575834c195cb169ec2bb444544fa8ab7d3d4a0916fa99139f54` |

Each enumerated permutation contained the same three object trajectories and unchanged semantic labels. The averaged representation therefore removed the D2 slot-realization validity failure.

## 3. Dataset integrity and frequencies

| Split | Scenes | Meanings | Inventory |
|---|---:|---:|---|
| train | 4,608 | all 48 Task-v2 training meanings | ordinary |
| validation | 1,536 | all 48 training meanings | ordinary |
| familiar test | 2,304 | all 48 training meanings | ordinary |
| common H–H | 864 | all 12 held-out meanings | G distractors |
| new H–H | 864 | all 12 held-out meanings | remaining-H distractor |
| full H–H | 864 | all 12 held-out meanings | ordinary |

Training contained exactly 192 scenes per directed endpoint pair, 2,304 per event, and 2,304 per family. Relation strata were H→G 1,728, G→H 1,728, and G→G 1,152. Source and target counts were 576 for each H identity and 960 for each G identity, the unavoidable structural asymmetry when all G–G meanings are included and all H–H meanings excluded. Fixed inverse-frequency weights were `[1.333333, 1.333333, 1.333333, 0.8, 0.8, 0.8]` for both endpoint probes and `[1,1]` for event.

There were no missing or extra training meanings, no active H–H training meaning, and no duplicate or cross-split scene hash. Exact per-split source, target, event, pair, family, stratum, and distractor tables are in `pretraining_validation.json`; every scene seed and hash is in `dataset_membership.csv`.

## 4. Probe training

Linear probes were one affine layer with softmax: 294 parameters for a six-class endpoint and 98 for event. Secondary probes were `48 -> 16 -> GELU -> output`: 886 and 818 parameters. All used AdamW, learning rate 0.01, no weight decay, batch size 256, inverse-frequency class weights, clipping at 1.0, exactly 1,500 updates, and final-checkpoint evaluation only.

At step 1,400, linear train/validation accuracy was 77.89/76.76% source, 88.50/85.74% target, and 78.23/76.89% event. MLP accuracy was 79.12/77.02%, 95.53/94.34%, and 99.28/98.57%. Training was not extended.

## 5. Primary linear results

Common-H–H intervals are simultaneous two-sided familywise 95% Wilson intervals after Bonferroni correction across source, target, and event. Other intervals are ordinary 95% Wilson intervals.

| Target | Familiar | Common H–H | New H–H | Full H–H |
|---|---:|---:|---:|---:|
| source | 1,771/2,304 = 76.8663% [75.1006, 78.5426] | 129/864 = 14.9306% [12.2594, 18.0639] | 286/864 = 33.1019% [30.0449, 36.3084] | 175/864 = 20.2546% [17.7092, 23.0634] |
| target | 2,015/2,304 = 87.4566% [86.0415, 88.7470] | 379/864 = 43.8657% [39.8778, 47.9345] | 353/864 = 40.8565% [37.6262, 44.1677] | 382/864 = 44.2130% [40.9343, 47.5429] |
| event | 1,815/2,304 = 78.7760% [77.0592, 80.3971] | 589/864 = 68.1713% [64.2684, 71.8347] | 560/864 = 64.8148% [61.5714, 67.9271] | 575/864 = 66.5509% [63.3378, 69.6176] |

Both familiar endpoint gates fail decisively. The classification is therefore D2R-D before attempting to interpret H–H endpoint failure as a compositional representation result.

## 6. Familiar semantic strata

| Probe | Target | H→G | G→H | G→G |
|---|---|---:|---:|---:|
| linear | source | 363/864 = 42.0139% | 845/864 = 97.8009% | 563/576 = 97.7431% |
| linear | target | 856/864 = 99.0741% | 589/864 = 68.1713% | 570/576 = 98.9583% |
| linear | event | 829/864 = 95.9491% | 804/864 = 93.0556% | 182/576 = 31.5972% |
| MLP | source | 364/864 = 42.1296% | 846/864 = 97.9167% | 562/576 = 97.5694% |
| MLP | target | 856/864 = 99.0741% | 737/864 = 85.3009% | 567/576 = 98.4375% |
| MLP | event | 846/864 = 97.9167% | 855/864 = 98.9583% | 567/576 = 98.4375% |

The familiar aggregate does not reflect uniformly weak identity information. G identities are highly recoverable as source or target, while an H source paired with G is poorly decoded and an H target paired with G is weaker, particularly linearly. The linear event failure is concentrated in G→G and is almost completely repaired by the fixed MLP.

## 7. Independent joint endpoint decoding

Source and target predictions were combined without further training.

| Probe | Split | Source | Target | Both / ordered pair |
|---|---|---|---|---|
| linear | familiar | 1,771/2,304 | 2,015/2,304 | 1,491/2,304 = 64.7135% |
| linear | common | 129/864 | 379/864 | 23/864 = 2.6620% |
| linear | new | 286/864 | 353/864 | 80/864 = 9.2593% |
| linear | full | 175/864 | 382/864 | 34/864 = 3.9352% |
| MLP | familiar | 1,772/2,304 | 2,160/2,304 | 1,635/2,304 = 70.9635% |
| MLP | common | 78/864 | 469/864 | 28/864 = 3.2407% |
| MLP | new | 261/864 | 315/864 | 94/864 = 10.8796% |
| MLP | full | 121/864 | 458/864 | 43/864 = 4.9769% |

Both-correct accuracy was below the product of marginal endpoint accuracies in every condition, by 1.14–5.02 percentage points.

## 8. Secondary MLP results

| Target | Familiar | Common H–H | New H–H | Full H–H |
|---|---:|---:|---:|---:|
| source | 1,772/2,304 = 76.9097% | 78/864 = 9.0278% | 261/864 = 30.2083% | 121/864 = 14.0046% |
| target | 2,160/2,304 = 93.7500% | 469/864 = 54.2824% | 315/864 = 36.4583% | 458/864 = 53.0093% |
| event | 2,268/2,304 = 98.4375% | 726/864 = 84.0278% | 849/864 = 98.2639% | 757/864 = 87.6157% |

The MLP recovers familiar event information and improves target decoding, but it does not make both familiar endpoints strongly recoverable and does not rescue H–H endpoints.

## 9. Randomized-label null distributions

For each target, 32 predetermined global training-label permutations were fitted with the identical weighted linear-probe procedure. This produced 96 null probes and 384 target/randomization/split evaluations. Empirical upper-tail probability was `(1 + number of null accuracies >= observed) / 33`; minimum possible p was 0.0303. No permutation was rerun.

| Target/split | Observed | Null mean ± SD | Null 2.5%, median, 97.5% | Null ≥ observed | p |
|---|---:|---:|---:|---:|---:|
| source familiar | 76.8663% | 17.1604 ± 5.5921% | 7.9470, 17.2309, 26.7556% | 0/32 | .0303 |
| source common | 14.9306% | 18.7536 ± 7.1773% | 6.0301, 17.1296, 30.1591% | 20/32 | .6364 |
| source new | 33.1019% | 20.8370 ± 11.4929% | 1.8461, 19.7338, 36.3947% | 5/32 | .1818 |
| source full | 20.2546% | 19.1334 ± 8.0977% | 6.2211, 19.0394, 31.0185% | 16/32 | .5152 |
| target familiar | 87.4566% | 16.3045 ± 6.5386% | 7.9915, 15.8203, 30.5664% | 0/32 | .0303 |
| target common | 43.8657% | 17.4913 ± 8.1357% | 2.6997, 16.9560, 28.4751% | 0/32 | .0303 |
| target new | 40.8565% | 17.9217 ± 11.8828% | 0.4485, 18.6921, 33.2581% | 0/32 | .0303 |
| target full | 44.2130% | 17.6432 ± 8.9804% | 2.0573, 18.2870, 29.8409% | 0/32 | .0303 |
| event familiar | 78.7760% | 51.0444 ± 7.0134% | 40.1931, 50.4557, 64.2719% | 0/32 | .0303 |
| event common | 68.1713% | 54.8213 ± 15.9290% | 25.0145, 53.5301, 86.1661% | 6/32 | .2121 |
| event new | 64.8148% | 55.4290 ± 18.4906% | 20.3704, 50.1736, 93.1337% | 6/32 | .2121 |
| event full | 66.5509% | 54.5862 ± 15.9456% | 21.3715, 53.6458, 88.0237% | 6/32 | .2121 |

All three familiar true-label probes exceed their nulls, so their below-threshold accuracies are not simply random-label behavior. On common H–H, target exceeds the null, while source and event do not.

## 10. Majority and class-prior baselines

Training-prior majority classes for endpoints were G identity 3 (ties broken deterministically). Their familiar accuracy was 20.8333%; because H–H truth contains only identities 0–2, their H–H accuracy was 0%. The expected class-prior baseline was 17.7083% on familiar and 12.5% on each H–H split. Event majority and prior baselines were 50% everywhere.

The randomized-label distributions, not these textbook baselines, are the inferential comparator under train/test shift.

## 11. Complete confusion matrices

Rows are true classes and columns predicted classes. Matrices are ordered by labels 0–5 for endpoints and 0–1 for event. Zero H–H rows for true labels 3–5 are retained in `confusion_matrices.csv`.

### Linear source

- familiar: `[[151,44,86,3,3,1],[101,94,85,2,4,2],[114,47,118,5,4,0],[0,1,1,474,2,2],[2,0,0,6,468,4],[0,2,0,5,7,466]]`
- common: `[[44,80,2,65,54,43],[31,84,1,81,50,41],[44,60,1,72,62,49],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- new: `[[178,106,0,4,0,0],[179,108,0,1,0,0],[209,77,0,2,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- full: `[[80,72,1,55,37,43],[70,93,1,59,34,31],[82,68,2,56,41,39],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`

### Linear target

- familiar: `[[221,37,25,1,0,4],[71,183,27,2,2,3],[52,51,185,0,0,0],[0,3,0,477,0,0],[0,1,2,3,473,1],[1,0,2,1,0,476]]`
- common: `[[131,133,0,9,6,9],[23,244,0,8,5,8],[65,198,4,6,6,9],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- new: `[[67,221,0,0,0,0],[2,286,0,0,0,0],[29,259,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- full: `[[124,150,0,2,7,5],[22,250,0,2,6,8],[54,211,8,5,8,2],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`

### Linear event

- familiar: `[[904,248],[241,911]]`
- common: `[[387,45],[230,202]]`
- new: `[[431,1],[303,129]]`
- full: `[[400,32],[257,175]]`

### MLP source

- familiar: `[[134,68,82,2,1,1],[95,139,48,2,1,3],[118,73,91,5,1,0],[0,0,1,475,1,3],[1,1,0,6,469,3],[0,3,0,5,8,464]]`
- common: `[[27,31,19,91,80,40],[29,30,20,95,71,43],[19,21,21,84,96,47],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- new: `[[135,81,22,26,24,0],[135,94,23,17,19,0],[150,77,32,15,14,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- full: `[[58,30,15,73,70,42],[59,37,28,81,53,30],[55,37,26,77,62,31],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`

### MLP target

- familiar: `[[246,21,16,1,0,4],[24,222,36,2,2,2],[5,14,269,0,0,0],[1,4,0,475,0,0],[1,3,1,3,472,0],[0,0,2,1,1,476]]`
- common: `[[134,123,3,8,10,10],[5,260,4,7,4,8],[13,181,75,2,8,9],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- new: `[[9,279,0,0,0,0],[0,288,0,0,0,0],[1,269,18,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`
- full: `[[115,155,3,2,8,5],[9,260,6,1,5,7],[6,187,83,3,8,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]`

### MLP event

- familiar: `[[1147,5],[31,1121]]`
- common: `[[368,64],[74,358]]`
- new: `[[428,4],[11,421]]`
- full: `[[381,51],[56,376]]`

## 12. Validity decision

| Validity requirement | Result |
|---|---|
| exact D1 checkpoint/boundary | pass |
| frozen encoder | pass |
| all and only 48 training meanings | pass |
| zero active H–H training | pass |
| zero scene overlap | pass |
| six-permutation mean invariant within `1e-6` | pass |
| 32 × 3 null probes completed exactly | pass |
| no H–H checkpoint or hyperparameter selection | pass |
| required controls/artifacts completed | pass |

D2-R is therefore valid. The failed familiar source/target thresholds map it to **D2R-D**, not D2R-A/B/C or INVALID.

## 13. What D2-R establishes

The selected 48-dimensional permutation-marginalized representation does not expose both endpoint identities strongly to the preregistered independent linear or small-MLP probes, even across semantic meanings present during D1 training. The failure is structured by relation regime rather than uniform: G endpoint identities are readily decoded, while H endpoint identities in cross-group relations are substantially less accessible. Familiar true-label probes nevertheless exceed the randomized-label nulls.

This makes ordinary independent-identity probing insufficient for deciding whether D1's H–H behavioral failure originates in candidate encoding or later comparison. The valid result narrows the methodological conclusion, not the mathematical information content.

## 14. What D2-R does not establish

D2R-D does not mean source or target information is absent. It does not show that D1 lacks endpoint information, because D1 identifies familiar candidates almost perfectly from the same underlying candidate pathway. It does not establish causal use, semantic variables, or a particular internal format. Information could be relational, distributed, query-dependent, or accessible only through interactions not modeled by independent identity probes.

## 15. Comparison with D1

D1 had approximately 100% familiar behavior, approximately 50% H–H joint behavior, 98–99% held-out event accuracy, and approximately 50% held-out ordered endpoints. D2-R does not reproduce that same dissociation cleanly at the independent-representation-probe level:

- endpoints are not strongly probe-accessible even on familiar support;
- common H–H source/target decoding is poor, but familiar weakness prevents interpreting the decrease as a pure composition effect;
- linear common event is 68.17%, and MLP common event is 84.03%, rather than D1's 98–99%; only new-inventory MLP event reaches 98.26%.

Therefore the D1 factor pattern is likely produced by message-conditioned candidate comparison or another distributed relational computation, not by three uniformly accessible independent labels in the pooled candidate vector. This is a representational-accessibility statement, not a mechanistic claim.

## 16. Exactly one recommended next experiment

Run one preregistered **oracle-conditioned relational matchability diagnostic** with the D1 encoder still frozen: train a single controlled bilinear compatibility readout between the constituent oracle query `(source,target,event)` and the permutation-marginalized candidate representation, using all 48 training meanings and testing H–H without independent source/target classification. This asks whether distributed relation information supports query–candidate matching even when absolute endpoint identities are not independently probe-accessible. If it succeeds IID but fails H–H, the representation/comparison interface is pair-specific; if it succeeds both, D1's existing comparison implementation is implicated.

Do not run that experiment from D2-R. No activation or causal interpretation is authorized.

## 17. Reproducibility artifacts

The bundle contains the D1 checkpoint, frozen configs and source, generated scenes, scene seeds and hashes, permutation-marginalized representations, six real probe checkpoints, 96 null-probe checkpoints, every real and null prediction, complete confusion matrices, exact joint/stratum/baseline/null tables, training traces, validation audit, aggregate metrics, run log, research log, and SHA-256 manifest.

Environment: Python 3.12.14, PyTorch 2.8.0, NumPy 2.3.5, macOS 26.6.2 arm64, CPU. Scientific training and evaluation wall time after validation was 37.1193 seconds.
