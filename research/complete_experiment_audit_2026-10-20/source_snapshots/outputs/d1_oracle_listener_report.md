# D1 — Oracle Semantic Channel Diagnostic for Task v2

## Executive result

**D1-B — IID PASS, COMPOSITION FAIL.**

A single preregistered seed-1701 listener was trained for exactly 8,000 updates. With a deterministic, injective oracle message containing the correct source, target, and event, the unchanged Task-v2 listener reached **99.8775% Family-S IID joint accuracy (8,150/8,160)** and **99.9510% Family-R IID joint accuracy (8,156/8,160)**. It did not compose those familiar constituents into held-out H–H endpoint relations: common-inventory joint accuracy was **47.9043% for S (3,863/8,064)** and **49.7644% for R (4,013/8,064)**. Thus D1 rules out the strong claim that the listener or rendered task is generally unlearnable when perfect semantics are supplied, but it shows that the emergent channel is not sufficient to explain the original compositional failure.

No failed Task-v2 model was rerun or tuned. No speaker, emergent protocol, straight-through estimator, auxiliary loss, mechanistic interpretation, checkpoint selection, extra seed, rescue run, or post-result modification was used.

## 1. Oracle codebook and validation

The zero-parameter oracle encoder was

`oracle(s,t,e) = [source_code(s), target_code(t), event_code(e), PAD]`,

with the exact constituent codebook:

| Constituent | Value | Token |
|---|---:|---:|
| source | 0, 1, 2, 3, 4, 5 | 0, 1, 2, 3, 4, 5 respectively |
| target | 0, 1, 2, 3, 4, 5 | 0, 1, 2, 3, 4, 5 respectively |
| event | 0 | 6 |
| event | 1 | 7 |
| PAD | constant | 0 |

This gives all 60 directed non-self endpoint/event meanings unique messages. The complete 60-row mapping is saved in `oracle_codebook.csv`.

| Validation | Exact result |
|---|---:|
| meanings checked | 60 |
| unique messages | 60 |
| collisions | 0 |
| source substitutions changed only position 0 | pass |
| target substitutions changed only position 1 | pass |
| event substitutions changed only position 2 | pass |
| generator/message correspondences checked | 3,840 |
| correspondence errors | 0 |
| held-out H–H messages used familiar constituent tokens | pass |

Examples: `(0,1,0) -> [0,1,6,0]`, `(0,1,1) -> [0,1,7,0]`, and `(2,1,1) -> [2,1,7,0]`. These are experimental inputs, not claims about natural semantic registers or the trained listener's internal variables.

## 2. Frozen-design verification

The D1 generator retained Task v2's H/G split (`H={0,1,2}`, `G={3,4,5}`), 48 training meanings, 12 held-out H–H meanings, S/R families, 50:50 family mixture, ten eligible triples, online nuisance geometry, renderer and validity rules, candidate construction, distractor schedule, object-slot and candidate-position randomization, optimizer, learning rate, batch size, cross-entropy loss, seed, 8,000-update budget, and evaluation-block construction.

The frozen generator audit contained 3,840 examples: 1,920 per family and 1,920 per event. Every ordered assignment appeared 64 times; each selected member appeared four times conditionally; each answer position appeared once conditionally. All Task-v2 shortcut and leakage audits passed. The only substantive intervention was replacing the learned speaker and hard emergent message with the fixed oracle message.

## 3. Training

Only the unchanged listener path was optimized: symbol embedding, message transformer, and candidate-scene encoder. The speaker scene encoder and emission head were instantiated but frozen and unused.

| Quantity | Value |
|---|---:|
| seed | 1701 |
| updates | 8,000 exactly |
| total model parameters | 118,208 |
| trainable listener parameters | 77,664 |
| wall time | 346.394 s |
| throughput | 23.095 updates/s |
| device | CPU |
| final checkpoint SHA-256 | `69e9843ba14ca7d6cd527fbe5daf408fb1b446f7871b58e236c11147a443a77e` |

At step 0, monitored training/IID-S/IID-R/common-S/common-R joint accuracies were 22.66%, 25.42%, 26.25%, 27.50%, and 26.88%. By step 1,500 they were 100.00%, 99.79%, 100.00%, 52.29%, and 51.67%. At the final logged point (step 7,900) they were 100.00%, 99.58%, 100.00%, 48.33%, and 49.79%. The curve therefore shows rapid IID convergence and a long composition plateau near 50%, not continuing improvement at the budget boundary. The monitor was diagnostic only; the saved model is the final 8,000-update checkpoint.

![D1 learning curve](d1_learning_curve.svg)

## 4. IID and compositional behavior

All rows use the same hard four-token message path, candidate scoring, and all-four-target block evaluation. Counts in the factor columns are correct predictions for that factor; `pair` means ordered `(source,target)`.

| Split/family | n | source | target | event | pair | joint |
|---|---:|---:|---:|---:|---:|---:|
| IID S | 8,160 | 8,153 (99.9142%) | 8,157 (99.9632%) | 8,159 (99.9877%) | 8,150 (99.8775%) | 8,150 (99.8775%) |
| IID R | 8,160 | 8,160 (100.0000%) | 8,160 (100.0000%) | 8,156 (99.9510%) | 8,160 (100.0000%) | 8,156 (99.9510%) |
| common H–H S | 8,064 | 5,318 (65.9474%) | 5,325 (66.0342%) | 7,881 (97.7307%) | 3,926 (48.6855%) | 3,863 (47.9043%) |
| common H–H R | 8,064 | 4,035 (50.0372%) | 4,035 (50.0372%) | 8,021 (99.4668%) | 4,035 (50.0372%) | 4,013 (49.7644%) |
| new-inventory H–H S | 8,064 | 5,427 (67.2991%) | 5,390 (66.8403%) | 8,056 (99.9008%) | 4,055 (50.2852%) | 4,051 (50.2356%) |
| new-inventory H–H R | 8,064 | 4,108 (50.9425%) | 4,108 (50.9425%) | 8,062 (99.9752%) | 4,108 (50.9425%) | 4,107 (50.9301%) |
| full held-out S | 8,064 | 5,316 (65.9226%) | 5,358 (66.4435%) | 7,910 (98.0903%) | 3,949 (48.9707%) | 3,895 (48.3011%) |
| full held-out R | 8,064 | 4,043 (50.1364%) | 4,043 (50.1364%) | 8,002 (99.2312%) | 4,043 (50.1364%) | 4,011 (49.7396%) |

The primary common-inventory result is not an inventory-novelty effect: new-inventory and full held-out scores are also approximately 50%. Event transfer is strong in every held-out condition, but ordered endpoint-pair transfer is not.

## 5. Family-S ambiguity tests

Each contrast has two alternatives and chance accuracy 50%.

| Split | Contrast | Correct / n | Accuracy |
|---|---|---:|---:|
| IID S | source | 4,072 / 4,080 | 99.8039% |
| IID S | target | 4,074 / 4,080 | 99.8529% |
| IID S | event | 4,076 / 4,080 | 99.9020% |
| common H–H S | source | 1,343 / 4,032 | 33.3085% |
| common H–H S | target | 1,353 / 4,032 | 33.5565% |
| common H–H S | event | 2,507 / 4,032 | 62.1776% |

Perfectly specified factor messages therefore produce nearly perfect familiar contrasts but do not yield compositional source/target discrimination. These are behavioral results only.

## 6. Every held-out pair × event × family cell

Each cell has `n=672`; the table reports `correct (accuracy)` for the common-inventory primary split.

| Family | source→target | event 0 | event 1 |
|---|---|---:|---:|
| R | 0→1 | 334 (49.7024%) | 323 (48.0655%) |
| R | 0→2 | 344 (51.1905%) | 340 (50.5952%) |
| R | 1→0 | 334 (49.7024%) | 346 (51.4881%) |
| R | 1→2 | 326 (48.5119%) | 350 (52.0833%) |
| R | 2→0 | 322 (47.9167%) | 336 (50.0000%) |
| R | 2→1 | 339 (50.4464%) | 319 (47.4702%) |
| S | 0→1 | 340 (50.5952%) | 302 (44.9405%) |
| S | 0→2 | 303 (45.0893%) | 331 (49.2560%) |
| S | 1→0 | 352 (52.3810%) | 331 (49.2560%) |
| S | 1→2 | 300 (44.6429%) | 320 (47.6190%) |
| S | 2→0 | 324 (48.2143%) | 341 (50.7440%) |
| S | 2→1 | 303 (45.0893%) | 316 (47.0238%) |

The minimum cell was S, `1→2`, event 0: **300/672 = 44.6429%**, far below the preregistered 90% floor. No cell approached that floor.

## 7. Oracle-message controls

Controls combine IID S and R (`n=16,320`) unless stated otherwise.

| Condition | source | target | event | pair | joint |
|---|---:|---:|---:|---:|---:|
| normal oracle | 16,313 (99.9571%) | 16,317 (99.9816%) | 16,315 (99.9694%) | 16,310 (99.9387%) | 16,306 (99.9142%) |
| fixed neutral message | 9,388 (57.5245%) | 8,834 (54.1299%) | 9,054 (55.4779%) | 7,071 (43.3272%) | 4,080 (25.0000%) |

Neutral-message prediction-position counts were `[3,928, 4,000, 4,284, 4,108]` for positions 0–3, so its exactly-chance joint result was not produced by a single fixed answer position.

The exact independent-message product crossed every IID recipient example with every one of 48 distinct IID oracle messages. It produced **66,585,600 / 266,342,400 = 25.0000%** joint accuracy. Donor-recipient balance over the product was exact for event, orientation, and family: 133,171,200 same and 133,171,200 different for each. There were 6,214,656 identical-meaning pairs and 260,127,744 different-meaning pairs. Candidate-block bootstrap intervals were [25.0000%, 25.0000%] for both neutral and independent replacement.

Normal behavior being nearly perfect while neutral and independent replacement are exactly chance rules out a candidate-only or message-free solution in this evaluation. The balanced candidate permutations below further argue against answer-position leakage.

## 8. Factor-corruption experiments

Counts combine IID S/R. `corrupted target present` is the subset where the candidate corresponding to the corrupted oracle meaning is actually available; redirection is evaluated only there.

| Corruption | n | original selected | corrupted alternative selected | other candidate | corrupted target present | redirected / present |
|---|---:|---:|---:|---:|---:|---:|
| wrong source | 16,320 | 9,813 | 4,072 | 2,435 | 4,080 | 4,072/4,080 (99.8039%) |
| wrong target | 16,320 | 10,033 | 4,074 | 2,213 | 4,080 | 4,074/4,080 (99.8529%) |
| wrong event | 16,320 | 96 | 12,232 | 3,992 | 12,240 | 12,232/12,240 (99.9346%) |

When the factor-corrupted semantic alternative is present, the listener is redirected to it almost perfectly. This verifies behavioral use of the supplied oracle on familiar IID patterns; it is not evidence about internal neural variables.

## 9. Permutation invariance

All comparisons are example-level, with permutations undone before comparing predicted semantic candidates.

| Test | Agreement | Accuracy before | Accuracy after |
|---|---:|---:|---:|
| candidate permutation | 16,320/16,320 (100.0000%) | 99.9142% | 99.9142% |
| object-slot permutation | 16,314/16,320 (99.9632%) | 99.9142% | 99.9387% |

Candidate accounting is exact. Object-slot agreement exceeds the preregistered 99% sufficiency threshold; six predictions changed, with no accuracy degradation.

## 10. Statistical confidence results

The candidate block, not the four target examples, was the bootstrap unit. The table reports the preregistered simultaneous one-sided lower confidence bound after Bonferroni adjustment.

| Endpoint | Point | Simultaneous lower bound | Blocks |
|---|---:|---:|---:|
| IID S joint | 99.8775% | 99.7672% | 2,040 |
| IID R joint | 99.9510% | 99.8775% | 2,040 |
| common S joint | 47.9043% | 47.4702% | 2,016 |
| common R joint | 49.7644% | 49.5164% | 2,016 |
| IID S source ambiguity | 99.8039% | 99.6078% | 2,040 |
| IID S target ambiguity | 99.8529% | 99.6814% | 2,040 |
| IID S event ambiguity | 99.9020% | 99.7549% | 2,040 |
| common S source ambiguity | 33.3085% | 31.9692% | 2,016 |
| common S target ambiguity | 33.5565% | 32.2170% | 2,016 |
| common S event ambiguity | 62.1776% | 60.6396% | 2,016 |

The IID pass and common-composition failure are not borderline threshold calls.

## 11. Preregistered decision gates

| Gate | Result |
|---|---|
| IID S joint ≥95% | pass |
| IID R joint ≥95% | pass |
| common S joint ≥95% | fail |
| common R joint ≥95% | fail |
| all IID/common S ambiguity contrasts ≥95% | fail |
| every held-out cell ≥90% | fail |
| neutral/independent controls appropriate | pass |
| candidate permutation exact | pass |
| object-slot invariance sufficient | pass |
| generator/oracle integrity | pass |

This combination maps unambiguously to **D1-B**, not D1-A, D1-C, or D1-INVALID.

## 12. Scientific localization

D1 rules out the strongest downstream failure hypothesis: the unchanged listener can learn the rendered candidate task extremely reliably when supplied perfect semantic messages for familiar endpoint relations. It also rules out nominal channel capacity, oracle collision, candidate-only solution, answer-position leakage, and serious candidate/object-slot dependence as explanations of this D1 result.

D1 does **not** show that the original learned speaker extracted semantics, that the hard channel was optimizable, or that speaker-listener coordination was successful. Those upstream causes remain possible explanations of the original Task-v2 IID failure.

Crucially, D1 also establishes an independent downstream compositional limitation. Even a perfectly factorized source/target/event message did not cause the listener to recombine endpoint constituents for unseen H–H relations. The strong held-out event accuracy with near-50% ordered-pair accuracy localizes the D1 failure primarily to endpoint relation matching, not event decoding. Because common-, new-, and full-inventory joint performance are all near 50%, the failure is not specifically caused by novel distractor inventory.

These observations are behavioral. They do not warrant interpreting token positions or activations as internal semantic registers.

## 13. What remains unresolved

D1 does not separate three closely related downstream explanations: (1) the rendered candidate-scene encoder fails to represent endpoint identity compositionally outside trained endpoint pairs; (2) the listener's message–candidate comparison architecture encourages pair-specific matching; or (3) the training distribution identifies a high-performing pair-specific IID solution without pressure for H–H constituent recombination. It also does not distinguish the upstream causes of the original end-to-end IID failure.

## 14. Exactly one next diagnostic

Run a new preregistered **candidate-representation sufficiency diagnostic**: freeze the D1/Task-v2 data generator and train a separately supervised readout from the unchanged candidate-scene representation to the candidate's source, target, and event labels, evaluating training endpoint pairs and held-out H–H pairs with the same inventory and permutation controls. This isolates whether the candidate encoder itself carries endpoint constituents across H–H composition before changing the listener comparison mechanism or returning to speaker diagnostics.

Do not run that diagnostic from this report. Under D1-B, a speaker diagnostic is premature.

## 15. Reproducibility and raw artifacts

The bundle contains the complete executable and inspection record:

- `configs/d1_preregistered.json`: frozen D1 declaration.
- `src/d1_oracle_listener.py`: training, evaluation, controls, bootstrap, and exports.
- `src/render_d1_learning_curve.py`: dependency-free plot renderer.
- `results/d1_oracle_listener_seed_1701/checkpoint.pt`: final 8,000-update checkpoint.
- `metrics.json`: all reported aggregates, gates, environment, and classification.
- `oracle_codebook.csv`, `oracle_audit.json`, `oracle_substitution_checks.csv`: complete oracle mapping and pretraining checks.
- `generator_audit.json`, `shortcut_oracles.json`, `candidate_blocks.csv`: frozen generator and leakage evidence.
- `heldout_cells.csv`: all 24 held-out cells and exact counts.
- `raw_predictions.csv`: every normal prediction for every reported split/family.
- `factor_corruption_predictions.csv`: every corrupted-message prediction.
- `independent_replacement_by_message.csv`, `independent_recipient_block_accuracy.csv`: exact independent-product tables.
- `permutation_mappings.csv`: example-level original and inverse-mapped permutation predictions.
- `bootstrap_indices.npz`: preserved block-bootstrap resamples.
- `learning_curve.csv`: all 80 monitoring records plus step 0.
- `work_d1.log`: run terminal record.
- `manifest.sha256`: artifact hashes.

Software environment: Python 3.12.14, PyTorch 2.8.0, NumPy 2.3.5, macOS 26.6.2 arm64, CPU.
