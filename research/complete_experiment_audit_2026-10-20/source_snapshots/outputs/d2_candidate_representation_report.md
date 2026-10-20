# D2 — Candidate Representation Sufficiency Diagnostic

## Executive classification

**D2-INVALID.**

The frozen checkpoint, representation boundary, data separation, balancing, and probe training all passed their implementation checks. However, two preregistered validity gates failed: object-slot prediction agreement fell below 99% for several probes, and the one fixed label-permutation control was not approximately chance on all held-out distributions. The scientific run was not repeated or tuned.

The preserved results show a strong **descriptive D2-D-like pattern**, but this is not promoted to a valid D2-D conclusion: the primary linear probes achieved only 66.6667% source, 86.8056% target, and 73.6979% event accuracy on the familiar test, and 4.3981%, 36.6898%, and 93.7500% respectively on common H–H. The small MLP did not rescue endpoint decoding. Because the diagnostic failed its own controls, these numbers localize what must be repaired in D2 rather than supporting a definitive claim about representational sufficiency.

No Task-v2 or D1 model was rerun, changed, or fine-tuned. No layer search, architecture search, held-out checkpoint selection, speaker training, emergent communication, activation intervention, or mechanistic interpretation was performed.

## 1. Scientific question

D2 asked whether the final D1 candidate representation permits a decoder trained without active H–H relations to recover source identity, target identity, and event on unseen H–H candidate scenes. A linear affine-softmax probe was primary; a fixed one-hidden-layer MLP was secondary. Probe success would establish recoverability only, not internal use or semantic variables.

## 2. Frozen candidate representation

The selected tensor is the only candidate-encoded/message-unconditioned boundary in `Agents.listen`:

`F.normalize(model.candidate(candidate_scene), dim=-1)`.

| Property | Exact definition |
|---|---|
| source code location | `pilot.Agents.listen` |
| encoder output | candidate encoder CLS token after both transformer blocks and final `Encoder.norm` |
| pooling | one pooled vector per candidate scene, not one vector per object |
| dimension | 48 |
| final transformation | L2 normalization |
| message interaction | none before this tensor |
| layer search | none |

The representation includes the candidate encoder input affine map, learned object-position embeddings, two transformer blocks, CLS pooling, final LayerNorm, and the same L2 normalization used by D1 scoring. Reconstructing the D1 listener score from this tensor and the message vector had maximum absolute error **0.0**.

The loaded checkpoint SHA-256 was exactly `69e9843ba14ca7d6cd527fbe5daf408fb1b446f7871b58e236c11147a443a77e`. Encoder parameters had gradients disabled; probe checkpoints contain only probe parameters.

## 3. Dataset integrity

Independent scenes were generated with the frozen Task-v2 renderer. Each split used a distinct seed phase, and every stored scene hash was unique.

| Split | Scenes | Active relations | Distractor inventory |
|---|---:|---|---|
| probe train | 5,184 | 18 directed H–G/G–H pairs, both events | ordinary |
| familiar validation | 1,728 | same balanced training stratum | ordinary |
| familiar test | 2,304 | all 48 Task-v2 training meanings | ordinary |
| common H–H | 864 | all 12 held-out meanings | G |
| new-inventory H–H | 864 | all 12 held-out meanings | remaining H |
| full held-out H–H | 864 | all 12 held-out meanings | ordinary |

There were **0 H–H training scenes**, **0 duplicate scene hashes**, and **0 train/test scene overlaps**. The probe-training stratum used all cross-group edges but no G–G edges. This was preregistered because positive G–G mass and zero H–H mass make exact six-identity source/target marginal balance mathematically impossible. The familiar test still covered all Task-v2 training meanings; its G–G subset is therefore a distributional stratum not seen by the probe.

Training balance was exact:

- each identity: 864 source occurrences and 864 target occurrences;
- each event: 2,592 occurrences;
- each distractor identity: 864 occurrences;
- source and target slots: 1,728 occurrences in each of the three slots;
- each source had three possible targets and each target three possible sources;
- every allowed distractor appeared 72 times conditional on each active pair.

The complete scenes, representations, scene seeds, labels, split membership, object slots, distractors, and SHA-256 scene hashes are preserved.

## 4. Probe specification and training

Three independent probes were trained for each decoder class. All used AdamW, learning rate 0.01, weight decay 0, batch size 256, gradient clipping at 1.0, exactly 1,500 updates, and the final checkpoint only.

| Probe | Source/target parameters | Event parameters |
|---|---:|---:|
| linear affine-softmax | 294 | 98 |
| one-hidden-layer MLP, width 16, GELU | 886 | 818 |
| D1 trainable listener, reference | 77,664 | 77,664 |

At step 1,400, linear train/validation accuracy was 71.78/71.06% for source, 86.40/85.24% for target, and 97.51/97.16% for event. MLP accuracy was 74.15/73.15%, 95.47/94.68%, and 99.40/99.19%. The curves were not used for checkpoint selection.

## 5. Linear probe results

The common-H–H confidence intervals use a two-sided Bonferroni correction over the three primary endpoints; all other intervals are ordinary 95% Wilson intervals.

| Target | Familiar | Common H–H | New H–H | Full H–H |
|---|---:|---:|---:|---:|
| source | 1,536/2,304 = 66.6667% [64.7155, 68.5624] | 38/864 = 4.3981% [3.0072, 6.3901] | 61/864 = 7.0602% [5.5354, 8.9651] | 37/864 = 4.2824% [3.1227, 5.8469] |
| target | 2,000/2,304 = 86.8056% [85.3622, 88.1264] | 317/864 = 36.6898% [32.8642, 40.6909] | 288/864 = 33.3333% [30.2699, 36.5443] | 315/864 = 36.4583% [33.3155, 39.7211] |
| event | 1,698/2,304 = 73.6979% [71.8618, 75.4552] | 810/864 = 93.7500% [91.4757, 95.4477] | 845/864 = 97.8009% [96.5910, 98.5877] | 808/864 = 93.5185% [91.6766, 94.9752] |

The preregistered familiar ≥95% precondition failed for all three linear targets. Common-H–H source predictions show major identity collapse: only 38/864 are correct, and true H sources are predominantly predicted as G identities. Common target predictions are strongly biased toward identity 1. Event generalizes much better than either endpoint but misses the ≥95% common threshold.

## 6. Primary linear confusion matrices

Rows are true labels and columns predicted labels in order 0–5. Event matrices use order 0–1.

### Source

| Split | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| familiar true 0 | 92 | 96 | 100 | 0 | 0 | 0 |
| familiar true 1 | 46 | 151 | 91 | 0 | 0 | 0 |
| familiar true 2 | 61 | 88 | 139 | 0 | 0 | 0 |
| familiar true 3 | 0 | 8 | 49 | 416 | 5 | 2 |
| familiar true 4 | 78 | 1 | 55 | 2 | 341 | 3 |
| familiar true 5 | 0 | 37 | 41 | 2 | 3 | 397 |
| common true 0 | 18 | 12 | 0 | 102 | 76 | 80 |
| common true 1 | 13 | 20 | 0 | 104 | 79 | 72 |
| common true 2 | 15 | 23 | 0 | 102 | 81 | 67 |
| new true 0 | 48 | 8 | 0 | 129 | 97 | 6 |
| new true 1 | 57 | 13 | 0 | 114 | 98 | 6 |
| new true 2 | 48 | 11 | 0 | 121 | 102 | 6 |

Rows for true identities 3–5 in H–H splits are structurally zero.

### Target

| Split | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| familiar true 0 | 206 | 34 | 47 | 0 | 0 | 1 |
| familiar true 1 | 29 | 209 | 48 | 2 | 0 | 0 |
| familiar true 2 | 23 | 47 | 218 | 0 | 0 | 0 |
| familiar true 3 | 0 | 2 | 22 | 454 | 0 | 2 |
| familiar true 4 | 1 | 0 | 23 | 3 | 452 | 1 |
| familiar true 5 | 8 | 0 | 8 | 1 | 2 | 461 |
| common true 0 | 55 | 196 | 0 | 12 | 11 | 14 |
| common true 1 | 3 | 255 | 0 | 12 | 11 | 7 |
| common true 2 | 21 | 236 | 7 | 5 | 12 | 7 |
| new true 0 | 0 | 287 | 0 | 1 | 0 | 0 |
| new true 1 | 0 | 288 | 0 | 0 | 0 | 0 |
| new true 2 | 0 | 288 | 0 | 0 | 0 | 0 |

### Event

| Split | true 0 → pred 0/1 | true 1 → pred 0/1 |
|---|---:|---:|
| familiar | 843 / 309 | 297 / 855 |
| common | 407 / 25 | 29 / 403 |
| new | 431 / 1 | 18 / 414 |

The machine-readable confusion table contains every cell for every linear, MLP, and control probe across familiar, common, new, and full splits.

## 7. Independent joint endpoint analysis

Ordered-pair predictions were formed without additional training by combining the independent source and target outputs.

| Probe | Split | source correct | target correct | both/ordered pair correct |
|---|---|---:|---:|---:|
| linear | familiar | 1,536/2,304 | 2,000/2,304 | 1,244/2,304 = 53.9931% |
| linear | common | 38/864 | 317/864 | 3/864 = 0.3472% |
| linear | new | 61/864 | 288/864 | 21/864 = 2.4306% |
| linear | full | 37/864 | 315/864 | 8/864 = 0.9259% |
| MLP | familiar | 1,463/2,304 | 1,892/2,304 | 1,219/2,304 = 52.9080% |
| MLP | common | 67/864 | 490/864 | 10/864 = 1.1574% |
| MLP | new | 148/864 | 188/864 | 3/864 = 0.3472% |
| MLP | full | 82/864 | 431/864 | 8/864 = 0.9259% |

For common linear decoding, both-correct accuracy was 1.2665 percentage points below the product of the two marginal accuracies; endpoint errors were not independent.

## 8. Small-MLP secondary probe

| Target | Familiar | Common H–H | New H–H | Full H–H |
|---|---:|---:|---:|---:|
| source | 1,463/2,304 = 63.4983% | 67/864 = 7.7546% | 148/864 = 17.1296% | 82/864 = 9.4907% |
| target | 1,892/2,304 = 82.1181% | 490/864 = 56.7130% | 188/864 = 21.7593% | 431/864 = 49.8843% |
| event | 2,117/2,304 = 91.8837% | 739/864 = 85.5324% | 856/864 = 99.0741% | 765/864 = 88.5417% |

The fixed 16-unit MLP did not recover source or target strongly on common H–H, and it did not meet the familiar ≥95% precondition. This is descriptive only because D2 is invalid.

## 9. Object-slot permutation

Each scene received an independent three-object permutation; predictions were compared example by example.

| Probe/target | Familiar agreement | Common agreement | New agreement | Minimum over all splits |
|---|---:|---:|---:|---:|
| linear source | 2,240/2,304 = 97.2222% | 833/864 = 96.4120% | 824/864 = 95.3704% | 95.3704% |
| linear target | 2,275/2,304 = 98.7413% | 842/864 = 97.4537% | 862/864 = 99.7685% | 97.4537% |
| linear event | 2,291/2,304 = 99.4358% | 858/864 = 99.3056% | 860/864 = 99.5370% | 99.3056% |
| MLP source | 2,257/2,304 = 97.9601% | 834/864 = 96.5278% | 855/864 = 98.9583% | 96.5278% |
| MLP target | 2,271/2,304 = 98.5677% | 840/864 = 97.2222% | 827/864 = 95.7176% | 95.3704% |
| MLP event | 2,296/2,304 = 99.6528% | 856/864 = 99.0741% | 864/864 = 100.0000% | 99.0741% |

Accuracy changes were small, but aggregate accuracy is not the preregistered criterion: several endpoint probes changed 2–5% of individual predictions. The ≥99% agreement gate therefore failed. This is consistent with learned positional encodings being present before CLS pooling; pooling did not make the selected representation exactly invariant.

## 10. Label-permutation control

One deterministic training-label permutation was used per target; it was not repeated.

| Target | Familiar | Common H–H | New H–H | Full H–H |
|---|---:|---:|---:|---:|
| source | 199/2,304 = 8.6372% | 231/864 = 26.7361% | 256/864 = 29.6296% | 238/864 = 27.5463% |
| target | 391/2,304 = 16.9705% | 127/864 = 14.6991% | 99/864 = 11.4583% | 113/864 = 13.0787% |
| event | 1,090/2,304 = 47.3090% | 290/864 = 33.5648% | 324/864 = 37.5000% | 282/864 = 32.6389% |

The preregistered control ranges were ≤25% for six-class common-H–H accuracy and 40–60% for event accuracy on every test split. Source exceeded the ceiling on common/new/full, while event fell below the interval on common/new/full. These are not misleadingly high task results, but they violate the declared adequacy criterion and show that one global random-label fit can acquire a distribution-dependent decision boundary under this probe-train/test shift.

## 11. Statistical confidence

The primary corrected intervals are:

| Common-H–H linear endpoint | Correct/n | Point | Bonferroni-adjusted 95% familywise Wilson interval |
|---|---:|---:|---:|
| source | 38/864 | 4.3981% | [3.0072%, 6.3901%] |
| target | 317/864 | 36.6898% | [32.8642%, 40.6909%] |
| event | 810/864 | 93.7500% | [91.4757%, 95.4477%] |

The intervals quantify the preserved sample but do not cure the failed validity gates.

## 12. What D2 establishes

D2 validly establishes only the audit facts: the requested D1 boundary was extracted exactly; the checkpoint remained frozen; the dataset contained no H–H training relation or scene overlap; training balance was exact; and the fixed probes, controls, and permutation tests produced the preserved counts above.

It also establishes that this particular D2 specification cannot support the intended representation-sufficiency conclusion. Endpoint probe performance was weak, but the familiar test mixed the balanced cross-group training stratum with G–G relations absent from probe training, individual predictions were slot-sensitive, and the random-label control behaved differently across held-out nuisance inventories.

## 13. What D2 does not establish

D2-INVALID does not show that source or target information is absent from the candidate representation. It does not support D2-A, D2-B, D2-C, or D2-D as a confirmatory classification. It does not show that D1 used, failed to use, or internally represented any decoded constituent. The poor probes could reflect the selected decoder classes, the probe-training relation stratum, position sensitivity, or their interaction.

## 14. Localization relative to D1

D1 remains unchanged: perfect oracle semantics yielded nearly perfect familiar behavior but approximately 50% H–H composition. D2 was meant to decide whether the downstream bottleneck is already present in the candidate representation or arises later in message–candidate comparison. Because D2 is invalid, that branch remains unresolved. Descriptively, the endpoint collapse is compatible with a candidate-representation/accessibility problem, but the evidence is not clean enough to rule out a probe-design/distribution problem.

## 15. Exactly one recommended next experiment

Run one freshly preregistered **D2 validity-repair replication** on the same frozen D1 checkpoint and same single representation boundary. Include all 48 Task-v2 training meanings in probe training, use declared inverse-frequency weighting/stratified reporting instead of dropping G–G relations, and train/evaluate each probe on the average of all six object-slot permutations. Use a fixed randomization distribution declared in advance for the negative-label control rather than accepting/rejecting a single shifted test accuracy. Keep the linear probe primary, the same 16-unit MLP secondary, and prohibit checkpoint or hyperparameter selection using H–H results.

Do not proceed to message–candidate comparison diagnostics until that validity repair passes. The recommended replication was not run.

## 16. Reproducibility record

The bundle preserves:

- `configs/d2_preregistered.json`;
- `src/d2_candidate_probe.py` and the serialization-only recovery `src/d2_finalize_existing.py`;
- the immutable D1 checkpoint hash and exact extraction definition;
- `probe_dataset.npz`, `representations.npz`, `dataset_membership.csv`, and all scene seeds/hashes;
- nine final probe checkpoints: linear, MLP, and label-permuted source/target/event;
- `label_permutations.npz`;
- `probe_results.csv`, `raw_predictions.csv`, and `confusion_matrices.csv`;
- `independent_joint_endpoints.csv`;
- `permutation_summary.csv` and every example-level `permutation_predictions.csv` row;
- `training_traces.csv`, `pretraining_validation.json`, and `metrics.json`;
- run log and SHA-256 manifest.

The scientific run completed all training and prediction exports, then failed only while serializing a NumPy boolean into the aggregate JSON. The summary was reconstructed from the already saved tables without retraining, resampling, or recomputing predictions. This repair is recorded explicitly rather than treated as a second run.
