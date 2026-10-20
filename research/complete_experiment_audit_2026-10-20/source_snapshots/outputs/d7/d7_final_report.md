# D7-R final report — Candidate-Only Predictability Diagnostic

## 1. Executive D7 classification

**D7-R-B — candidate-only signal exists but is insufficient.** Candidate-only information predicts the correct D6 answer at 0.5078125 under grouped held-out evaluation. It is reliable by every frozen criterion, but it does not reproduce the large REQUIRED/REDUNDANT asymmetry in D6 neutral-message performance. The repeated-PAD versus literal-zero comparison does not meet the distinct-path threshold.

D6 remains mechanically **D6-INVALID**. Original D7 remains **D7-INVALID** because its semantic decoder failed. D7-R is the separately frozen validity-repair replication.

## 2. Diagnostic question

Can the four candidates alone predict the answer, and does that explain why D6 listeners scored above chance when given the repeated token-0 control?

## 3. Frozen preregistration and hashes

- D7-R machine preregistration: `d784cb0a29b81f3ff4fddf4f33e03b4d2c4accafb401ceb0f57a48c749ae3c7f`
- D7-R scientific implementation: `0adf3f57bfc2fad368c16d005bc709d9c1dea2b73c85df65e5075a27c7957c86`
- Base D7 preregistration: `85c0d8fbc685c726ed508d95494045afb2be8f3ef4c620c2df8860cad8d29996`
- Base D7 implementation: `590ab18a6e22d8e4cf726eecab3d836ccb012d3a1225cd826259839b6cc039d1`
- D6 preregistration: `ed5423c7779699181bbd3fda6fe476eafc9a69bd5b52ed5f1479e32e85ec2657`
- D6 implementation: `019e7de6a40bcb5ca7af8f3990171a830751ff6b3ddf61d39932cb761fce4b3c`

## 4. D6 artifact verification

All 12 D6 runs are complete. Every final checkpoint matches its frozen hash and status record. The D6 aggregate and pretraining-audit hashes match. Native candidate blocks have exactly one frozen construction per condition; demand and transfer diagnostics are shared across conditions and seeds. D6 artifacts were read only.

## 5. Candidate-only feature provenance

Allowed features were candidate position; candidate-local source, target, and event decoded solely from its scene; within-set equality and frequency relations; fixed event/count interactions; and, for one secondary linear model, identity-sorted rendered coordinates. Condition and seed were used only for reporting.

Forbidden and unused features were the answer, query, query semantics, message/query representation, logical candidate role, `position_members`, targeted role, context/style as inputs, listener scores, predictions, and activations. The complete provenance table is `results/d7r_candidate_only/feature_provenance.csv`.

## 6. Split and generalization design

Six folds held out complementary unordered three-identity inventories. Each example was tested once after fitting on the other ten inventories in both conditions. There were zero exact candidate-set duplicates crossing folds. D6 seed copies were not duplicated or treated as candidate-only replicates.

## 7. Position-only baseline

The training-fold position-frequency baseline scored exactly **0.250000**. Answers were exactly balanced: 4,032 examples at each of four positions.

## 8. Structural heuristics

| Heuristic | Accuracy |
|---|---:|
| Maximum ordered-pair multiplicity, lowest-position tie | 0.498388 |
| Maximum ordered-pair multiplicity, event-0 tie | 0.500000 |
| Maximum ordered-pair multiplicity, event-1 tie | 0.500000 |

The transparent shortcut is therefore already visible without a learned model: candidates belonging to the repeated ordered endpoint pair isolate a two-candidate subset containing the answer.

## 9. Learned candidate-only baseline

The primary symbolic conditional logistic model achieved **8,190/16,128 = 0.5078125**. Cross-entropy was **0.694324**. Its inventory-group bootstrap 95% interval was **[0.501240, 0.513517]**. All six numerical fits had final gradient norms below `7.15e-7`.

Confusion matrix, rows true and columns predicted:

| | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
|---|---:|---:|---:|---:|
| True 0 | 2774 | 0 | 0 | 1258 |
| True 1 | 1324 | 1164 | 258 | 1286 |
| True 2 | 1324 | 1190 | 220 | 1298 |
| True 3 | 0 | 0 | 0 | 4032 |

The secondary rendered-linear model scored **0.503844**, cross-entropy **0.696429**. Rendered nuisance features did not improve on symbolic candidate structure.

## 10. Permutation and null results

Both learned models had permutation `p = 1/10001 = 0.00009999`. The primary accuracy exceeded `.30`, its grouped-bootstrap lower bound exceeded `.25`, and permutation `p<=.01`; the frozen reliable-candidate-signal gate passed.

## 11. REQUIRED versus REDUNDANT

Candidate-only accuracy was identical:

| Condition | Correct / N | Accuracy |
|---|---:|---:|
| REQUIRED | 4095 / 8064 | 0.5078125 |
| REDUNDANT | 4095 / 8064 | 0.5078125 |

Thus condition label was neither needed nor available to the predictor.

## 12. Seed and stratum breakdown

Candidate-only evaluation sets are identical across the six D6 model seeds, so there is no independent candidate-only seed variation. Repeating the same value six times would be pseudoreplication.

| Stratum | Accuracy |
|---|---:|
| Native demand | 0.504712 |
| Native transfer | 0.510913 |
| Shared diagnostic demand | 0.502108 |
| Shared diagnostic transfer | 0.501736 |

Fold accuracies ranged from 0.498512 to 0.519345. Inventory accuracies ranged from 0.483631 to 0.522321. Exact counts are in `accuracy_breakdowns.csv`.

## 13. Comparison with D6 neutral-message performance

D6 repeated-P-token-0 means were **0.313017 REQUIRED** and **0.448681 REDUNDANT**, pooled mean **0.380849**. Candidate-only accuracy was 0.5078125 in both conditions.

The candidate excess ratio was **1.970305**, but the pooled absolute gap was **0.126963**, exceeding the frozen `.05` boundary. The candidate condition contrast was zero, whereas the D6 neutral REQUIRED-minus-REDUNDANT contrast was **−0.135665**, also exceeding the `.05` discrepancy boundary. “Substantial portion” therefore failed mechanically. A seed-cell covariance is not scientifically meaningful because candidate sets do not vary by D6 model seed.

## 14. Zero-message computational-path audit

D6's implemented neutral input was four repeated one-hot token-0/PAD vectors, not a zero tensor. It generated a learned constant query using `sym[0]`, message positions, CLS, and the message transformer. It contained no example-specific query semantics, mask signal, cache, condition metadata, or answer.

Across 12 frozen listeners:

- Repeated PAD mean: **0.380849**
- Literal-zero tensor mean: **0.379309**
- Trained-pad-pattern-only mean: **0.384869**
- Repeated-PAD minus literal-zero mean: **0.001540**
- Positive differences: **5/12**
- Bootstrap 95% CI: **[−0.020451, 0.022869]**
- Exact one-sided sign-flip `p=.443359`

The frozen distinct-path criterion failed.

## 15. Observation

Candidate-only structure supports approximately 50% accuracy across conditions, held-out identity inventories, and evaluation strata. The simplest repeated-pair heuristics already attain approximately 50%. Rendered coordinates add no meaningful gain. D6 neutral accuracy is above chance but condition-dependent and lower on average. Changing repeated PAD to literal zeros does not systematically reduce it.

## 16. Inference

The D6 candidate design contains a robust, generalizing candidate-only shortcut. This explains why a query-free scorer can exceed four-way chance and shows that the D6 neutral gate correctly detected a genuine candidate-design problem. Candidate structure alone does not explain how strongly the frozen D6 listeners exploited that shortcut in REQUIRED versus REDUNDANT, so the mechanical outcome is D7-R-B rather than D7-R-A.

## 17. Speculation

The condition asymmetry may arise because the two training objectives shape the candidate encoder and its alignment with any constant query state differently, even though the evaluation candidate-only shortcut is the same. D7-R did not manipulate or decompose that learned alignment, so this remains speculation.

## 18. What caused the D6 neutral-control failure

An identifiable cause is the candidate construction: the repeated ordered endpoint pair narrows the answer to two candidates, permitting approximately 50% performance with a fixed event tie-break. The nominal zero control was also described imprecisely—it used repeated PAD one-hots—but the frozen causal comparison provides no evidence that this distinction materially caused the above-chance accuracy.

## 19. What D7-R establishes

D7-R establishes that candidate-only answer information generalizes across held-out identity inventories and is sufficient for reliable above-chance performance. It also establishes that the D6 repeated-PAD and literal-zero interventions have nearly identical mean behavior under the frozen threshold.

## 20. What D7-R does not establish

It does not validate or rescue D6, identify a transformer mechanism, establish independent symbolic variables, explain the entire condition-specific neutral pattern, or generalize beyond this synthetic renderer and candidate design.

## 21. Implications for a future demand experiment

Any future confirmatory demand experiment must remove the repeated-pair shortcut and pass a frozen candidate-only grouped-holdout gate before listener initialization, while retaining exact exposure matching and genuine endpoint demand.

## 22. Reproducibility artifacts and deviations

Core artifacts are the D7/D7-R preregistrations and implementations, `prefit_audit.json`, `metrics.json`, `message_path_by_model.csv`, `symbolic_oof_predictions.csv`, `accuracy_breakdowns.csv`, `feature_provenance.csv`, `fit_numerics.csv`, and checkpoint manifest.

Operational deviation: the first analysis computed results but failed at JSON serialization of a NumPy boolean. A separate serialization-only runner deterministically replayed the frozen analysis; it wrote complete metrics, then its optional console summary encountered the same type issue. No third run was made. No definition, threshold, fit, D6 artifact, or classification was changed. The event is documented in `operational_log.md`.

Reporting completion regenerated the deterministic frozen OOF fits solely to preserve preregistered per-example and stratum tables; the regenerated primary accuracy exactly matched `0.5078125` and did not affect classification.

## 23. Exactly one recommended next scientific action — NOT RUN

**Preregister a redesigned demand experiment whose four-candidate sets contain no candidate-only repeated-pair or equivalent structural answer cue, and require a grouped candidate-only pretraining gate at chance before initializing any listener. NOT RUN.**
