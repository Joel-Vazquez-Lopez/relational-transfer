# P2.1 — Bounded Predictive-Compression Audit

**Completed 20 September 2026 · VALID bounded discovery comparison, not confirmatory mechanism evidence.** The frozen [machine-readable protocol](../../configs/p2_1_predictive_compression_prereg.json) has SHA-256 `64c9c9ba3731373be0edc9b086be8e6a1a8e11c32e159e562526c991f9d9c65a`; the companion [human preregistration](../../phase2/preregistrations/p2_1_predictive_compression.md) has SHA-256 `c9ffc850b8d808772065e952e42a0141f0e7d7b0cd886ba2fdb4a410eb4f7b51`.

## 1. Executive summary

The strongest simple zero-shot model is a **51-parameter shared-feature/interpolation ridge**. Across the 12 discovery models, its equal-model-weight mean context-holdout score RMSE is **3.553**, versus **4.751** for a constant model; finite-response RMSE is **4.533** versus **6.089**, mixed query–candidate response RMSE **7.350** versus **11.013**, and meaningful finite-response sign accuracy **0.748**. These are useful but far from exact predictions on a score scale bounded by ±10. The same model transfers without target-seed fitting to the two A/B seed-1703 models (context score RMSE **3.486**), but worsens on the six C/D graph-transfer targets (**4.470**). Ordinary shared-feature interpolation therefore remains a serious conservative explanation for a substantial portion of the measured response structure; it has **not** been quantitatively defeated.

State completeness matters for a different, higher-cost family. A matched ridge transition surrogate from post-block-0 **CLS-only** states scores **3.686** RMSE on context holdout, while **CLS plus endpoint-bearing token/object states** scores **3.133**, improving in **12/12** models. The complete all-token input scores **3.371** (better than CLS-only in 12/12, worse than selected-token in 10/12). The selected-token surrogate has **13,920** fitted coefficients, versus **4,704** for CLS-only and **20,832** for the full-state surrogate. Thus omitted token information contributes to the poor CLS-only description; full-state ridge degradation is consistent with nuisance dimensionality/regularization and is **not** evidence that the full transformer state is non-closed. The selected-token surrogate improves raw scores but does **not** beat the 51-parameter interpolation null on mixed-response RMSE (7.466 versus 7.350) or endpoint-redirection effect MAE (3.457 versus 2.599).

No tested class gives a nearly complete compact prediction of held-out finite and mixed effects. The two-anchor response-state candidate is conditional on **two actual network score calls per query** and has worse withheld-column score RMSE than M1 (4.303 versus 3.553 on different score-cell sets); its lower finite/mixed errors cannot be read as zero-shot compression. No low-cardinality functional quotient, reusable operator law, acquisition principle, or grammar was found.

## 2. Validity and operational provenance

All 12 D5 discovery checkpoints (four conditions × seeds 1701–1703) were present, complete at update 8,000, and matched their frozen checkpoint hashes, embedded D5 config, graph specification, condition, and seed. The per-checkpoint hashes are in the [preregistration](../../configs/p2_1_predictive_compression_prereg.json) and [atlas manifest](../../phase2/p2_1/atlas/manifest.json). The 12 final D5 models were only loaded in eval mode; no optimizer, new task network, parameter update, or activation intervention was used. No D5 protected checkpoint for seed 1704, 1705, or 1706 was opened, hashed, evaluated, or used. D6 and D9 remain invalid and supplied no confirmatory evidence.

The atlas has **4,608** selected four-way blocks, **18,432** query examples, and **73,728** query–candidate score cells. Instrumented `10×dot(normalized query, normalized candidate)` scores agreed with direct listener scores to maximum absolute difference **3.814697265625×10⁻⁶**; query/candidate interface vectors agreed exactly. Every selected query row matched its original archived D5 `raw_predictions.csv` score row and prediction: **18,432/18,432**, maximum score difference **0**, prediction mismatches **0**. Symbolic meanings, physical candidate ordering, object-slot identities, and event trajectory signs passed the exhaustive selected-atlas audit with **0 errors**. Final pre-L2 norms were reconstructed in a read-only [normalization sidecar](../../phase2/p2_1/atlas/normalization_manifest.json), with normalized-vector reconstruction error **0** and unchanged model state.

Two pre-interpretation technical repairs are explicitly logged in [technical_repairs.md](../../phase2/p2_1/analysis/technical_repairs.md). The first fixed R-family context-triple metadata before any atlas was produced. The second corrected candidate semantic-member versus physical-position indexing after an initial mechanical analysis; **none of that first pass's scientific metrics was inspected or interpreted**, and its complete files are retained under `analysis/invalid_first_pass/`. The scientific preregistration, checkpoint set, block selection, splits, model classes, hyperparameter spaces, metrics, and thresholds were unchanged. Only the corrected tables are used below. The frozen config and human-preregistration hashes still match. No operational interruption affected model or checkpoint state.

| Frozen validity gate | Result and check |
| --- | --- |
| Exact frozen-network replay | **PASS:** 18,432/18,432 selected archived rows, score maximum difference 0, prediction mismatches 0 |
| Atlas versus direct score | **PASS:** maximum absolute difference 3.814697265625×10⁻⁶, below 10⁻⁵ |
| Legal symbolic substitutions | **PASS:** all selected rows follow D5 `qset`; trajectory/event audit errors 0 |
| Candidate and object ordering | **PASS:** recorded permutation, true position, identities and source/target slots agree with scenes and archived rows |
| Protected D8 seeds | **PASS:** no protected checkpoint access |
| No new task network | **PASS:** eval-only loads, no optimizer, unchanged model states |
| Frozen theory holdouts | **PASS:** preregistration hash fixed before fitting; exact split counts retained |
| Train-only preprocessing | **PASS by code audit:** fit, scales, alpha validation, and refit use only core rows; pooled transfer uses source models only |
| Query/candidate score algebra | **PASS:** final-vector dot products reproduce direct scores within tolerance; finite/mixed calculations use exact score cells |
| D6/D9 evidence boundary | **PASS:** neither invalid study is used in the analysis or interpretation |

## 3. Checkpoint inventory and architecture

All four D5 graph conditions were included without cherry-picking: A original clique hole, B relocated clique hole, C degree-balanced matching holes, and D path-connectivity swap; each contributed seeds 1701, 1702, 1703. Exact paths and hashes are in the manifest. The original model has **118,208** total parameters, of which **77,664** belong to the active oracle-message listener and candidate encoder rather than unused speaker/emitter modules. It has four oracle message positions plus CLS and three candidate object positions plus CLS, each with two 48-wide pre-norm transformer blocks. The exact scoring interface is 10 times the dot product of final L2-normalized message and candidate CLS vectors. The atlas records token-plus-position, post-block-0, and post-block-1 **all-token** states on both sides, final normalized vectors, complete rendered scenes, and a separate pre-L2 norm sidecar.

## 4. Atlas construction and natural targets

For each condition, seed, evaluation kind (familiar, focal A, focal B, withheld), and candidate family (S, R), a frozen hash rule selected **48** of D5's natural evaluation blocks. The four oracle query meanings were crossed with the block's four naturally rendered candidates, without altered activations or invented semantic scenes. Metadata retain meanings, event, triple/context, graph support, correct member, candidate permutation, query/candidate distractors, and source/target object slots. Member 0 is the semantic anchor; source/target/event or orientation/event alternatives follow the original D5 `qset`. Candidate positions are inverse-mapped to semantic members before contrasts.

The prediction targets are direct raw scores, correct-candidate margins, natural finite score responses to legal member substitutions, finite-response signs for |true effect|>0.1, four-cell mixed query–candidate interactions, and Family-S endpoint-redirection score effects. The identity relating a four-cell score matrix to its finite and mixed differences is exact instrumentation, **not** a compressed explanatory theory. The underlying D5 network-training held-out edges are separate from every P2.1 theory-fitting holdout.

## 5. Theory-fitting firewall

The fitting core excluded a union of hashed whole-context triples, withheld query/candidate meaning combinations, all query/candidate event-disagreement cells, and complete candidate configurations. Four corresponding test sets were reported separately. The core contains **1,682–2,079 score cells per model**, with a hash-defined internal fifth for alpha selection; all means, scales, ridge coefficients, and alpha choices were fit on core rows only. The [split table](../../phase2/p2_1/analysis/split_counts.csv) gives every model's exact counts. Complete-block response metrics are available for context and candidate-configuration holdouts; the partial-cell combination and substitution-family holdouts have score metrics, not falsely assembled full-block interactions.

M0/M1 receive oracle symbolic meaning and support descriptors; STATE-1's token/object selection uses atlas meaning metadata to identify the endpoint-bearing slots. Those are declared *inputs to an explanatory surrogate*, not learned semantic detectors from raw scenes. Their side-information advantage must be considered when comparing description costs or proposing an autonomous model of computation.

For cross-seed transfer, semantic/readout models were fitted on A/B seeds 1701–1702 and evaluated without adaptation on A/B seed 1703. For graph transfer, the same fitting pool was evaluated on C/D seeds 1701–1703. Independently initialized hidden-state coordinates were not aligned; their direct cross-model transfer is a descriptive stress test, not the primary within-model state comparison. No target-seed/graph normalizer or calibration was fitted.

## 6. P2.1A — state-closure audit

The fixed ridge class predicted both pathways' final normalized vectors from their post-block-0 states, then formed the predicted score via 10×dot. The state hierarchy was CLS-only (48 input dimensions per side), CLS plus source/target-bearing states (144 per side), and all tokens (240 query / 192 candidate dimensions). Mean context-holdout results, one vote per D5 model:

| Input state | Fitted parameters | Score RMSE | Query-vector MAE | Candidate-vector MAE |
| --- | ---: | ---: | ---: | ---: |
| STATE-0: CLS only | 4,704 | 3.686 | 0.01074 | 0.10563 |
| STATE-1: CLS + relevant token/object states | 13,920 | **3.133** | 0.000291 | 0.08746 |
| STATE-2: all tokens | 20,832 | 3.371 | 0.000102 | 0.09907 |

STATE-1 beats STATE-0 in **12/12** models; STATE-2 beats STATE-1 in only **2/12** under this bounded ridge fit. The all-token transformer state is sufficient for deterministic continuation by architecture. Failure of a regularized linear predictor with more nuisance dimensions does not show non-closure; it limits this particular compact predictor. No short-memory STATE-3/M6 model was run because the preregistered non-closure precondition was absent. [All per-model closure results](../../phase2/p2_1/analysis/state_closure.csv) are preserved.

## 7. P2.1B — predictive-compression tournament

Equal-model-weight means on the **context holdout** (12 models; 10,128 zero-shot score cells except M4's 5,064 conditionally predicted cells). Lower errors are better. Finite/mixed metrics are computed on 633 complete held-out blocks across models. Full per-model, per-holdout tables are in [model_metrics.csv](../../phase2/p2_1/analysis/model_metrics.csv), with the four-holdout aggregate in [aggregate_metrics.csv](../../phase2/p2_1/tables/aggregate_metrics.csv).

| Model | Fitted params | Score RMSE | Finite RMSE | Mixed RMSE | Finite sign acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 mean | 1 | 4.751 | 6.089 | 11.013 | 0.000 |
| M0 additive | 39 | 4.746 | 6.080 | 11.013 | 0.538 |
| **M1 shared-feature/interpolation ridge** | **51** | **3.553** | **4.533** | **7.350** | **0.748** |
| M1 5-NN lookup | mean 95,561 stored values | 4.449 | 5.710 | 8.763 | 0.650 |
| M2 static STATE-0 readout | 145 | 4.636 | 5.855 | 9.958 | 0.607 |
| M2 static STATE-1 readout | 337 | 4.662 | 5.876 | 9.898 | 0.610 |
| M2 static STATE-2 readout | 481 | 4.595 | 5.824 | 10.043 | 0.600 |
| M3 constrained gain | 11 | 4.650 | 6.085 | 10.847 | 0.528 |
| M4 two-anchor response model† | 41 + two score calls/query | 4.303† | 3.502† | 4.044† | 0.859† |
| STATE-0 transition surrogate | 4,704 | 3.686 | 5.390 | 8.908 | 0.680 |
| STATE-1 transition surrogate | 13,920 | 3.133 | 4.655 | 7.466 | 0.755 |
| M5 shared full-state linear map | 20,832 | 3.371 | 4.978 | 8.137 | 0.733 |

† M4 observes the true network scores against semantic candidate members 0 and 1 at test time and predicts only members 2 and 3. Its raw-score RMSE is on **half** the cells; finite/mixed predictions reuse the observed anchor scores. It is not comparable to zero-shot models on the same input budget, and its apparently good finite/mixed errors do **not** establish predictive-state compression. The 1-NN lookup ceiling is also fully preserved in the raw tables and performs worse than 5-NN.

The candidate-configuration holdout tells a similar story: M1 score/finite/mixed RMSE **3.529 / 4.572 / 7.329**, while the 13,920-parameter STATE-1 surrogate gives **3.001 / 4.460 / 7.143**. The latter improvement in score and finite RMSE is modest relative to its much larger fitted description and privileged internal-state input. On context holdout, M1's endpoint-redirection effect MAE is **2.599**, lower than STATE-1's **3.457**. Neither model dominates every decision-relevant response target.

## 8. Complexity and Pareto reading

The [complexity table](../../phase2/p2_1/analysis/complexity.csv) records fitted coefficients, feature dimensions, selected alpha, approximate serialized bytes, and anchor-call cost. The [Pareto figure](../../phase2/p2_1/figures/context_pareto.svg) plots context score RMSE against fitted-parameter count; its M4 point is explicitly marked conditional. M1's 51 fitted parameters are a genuine compact *score approximation* relative to the 77,664-parameter active listener, whereas 5-NN stores about 95,561 training values on average and offers no useful compression. M2's small static linear readouts underperform M1. STATE-1 uses 13,920 fitted coefficients and both pathways' internal token states; it earns a lower score RMSE but cannot be called a simpler raw-input model. M5 is one dense, shared linear map per pathway (48 output dimensions, 20,832 total fitted coefficients), **not** a per-context operator family. No arbitrary grand score or post-hoc winner threshold was introduced.

## 9. Cross-seed and graph generalization

Without target-model fitting, A/B seed-1703 **cross-seed** context RMSE is **3.486** for M1 and **4.621** for M0 additive (two target models). C/D **graph-transfer** context RMSE is **4.470** for M1 and **5.507** for M0 additive (six target models). M1 mixed-response RMSE is **7.358** cross-seed and **8.355** graph-transfer. These scores indicate nontrivial but degraded transfer of the declared semantic interpolation features. The unaligned M2 STATE-0 readout has context score RMSE **7.921** cross-seed and **6.395** graph-transfer; those failures are not evidence against within-model state content because coordinate alignment/adaptation was deliberately absent. Every transfer result and target is in [transfer_metrics.csv](../../phase2/p2_1/analysis/transfer_metrics.csv).

## 10. Failure analysis and strata

M1's context score RMSE is **3.494** on observed-query cells and **3.628** on withheld-query cells; its mean worst evaluation-kind RMSE is **4.031**. The STATE-1 surrogate improves those to **3.353 / 2.651**, but its mixed-response RMSE is **7.466**, slightly worse than M1. M1 endpoint finite-effect MAE is **3.081**, event finite-effect MAE **4.141**; both are non-negligible. On event-disagreement substitution-family cells—excluded entirely from fitting—M1 score RMSE rises to **4.262**; the full-state linear map scores **2.924** but at vastly higher complexity and with internal state access. The per-seed and supported/unsupported/family columns in the raw table should be used instead of inferring uniform generality from means. Context holdout is small in some conditions (e.g. 10 complete blocks per A seed); these are bounded discovery results, not precise population effects.

## 11. What P2.1 establishes

For the declared D5 discovery models, natural evaluation blocks, and structural holdouts, **some decision-relevant computation has a compact partial predictive description**: a 51-parameter semantic interaction model beats trivial/additive baselines and retains substantial cross-seed predictivity. Access to endpoint-bearing internal token/object states improves a learned transition-to-score description relative to CLS alone across all 12 models. A strong simple interpolation account remains viable; additional state helps certain targets but is not uniformly better after accounting for its much larger description and privileged inputs.

## 12. What P2.1 does not establish

It does not identify a causal mechanism, an abstract predictive state, a closed low-dimensional dynamics, a reusable finite operator law, an acquisition explanation, a machine-native grammar, or language transfer. The exact query–candidate dot-product identity is instrumentation, not an explanatory compression. M4 has no demonstrated low-cardinality classes or withheld-test reuse without fresh anchor calls; no state-per-example abstraction is accepted. M5's shared map is not shown to dominate the simple null at comparable complexity. D5 remains the bounded acquisition causal anchor; P2.1 cannot retroactively make D6 or D9 valid. D8 donor transfer remains closed and its protected seeds sealed.

## 13. Surviving and weakened explanatory classes

**Survive:** simple shared-feature/interpolation as a serious partial account; selected-token state plus a shared regularized transition as a higher-cost execution surrogate; full-state shared mapping as a non-dominant possibility. **Weakened in this bounded tournament:** purely additive semantics, the tested small static state readouts, a scalar downstream-gain account, and lookup as an efficient description. **Not demonstrated:** a predictive-response quotient or discrete program/automaton. M6 and M7 were **not run—preregistered preconditions absent**. These judgments compare the tested implementations only, not all members of each broad theory family.

## 14. Explicit unresolved disagreement

The leading disagreement is whether the residual finite/mixed interaction error of the small semantic interpolation model reflects a reusable, compact computation made accessible by a modest augmented internal state, or merely a higher-capacity conditional surrogate exploiting model-specific token states. The current Pareto curves and graph-transfer degradation do not resolve it. This is an empirical disagreement, **not** a P2.2 design.

## 15. Recommended next scientific question—no new study launched

What held-out response family would most sharply separate a genuinely reusable state-dependent rule from shared-feature interpolation while holding input access and description cost comparable? P2.1 stops here; no follow-on protocol, extraction, training, intervention, or language test has begun.

## Reproducibility index

The [reproducibility manifest](p2_1_reproducibility_manifest.json) records all source/config/preregistration/checkpoint/atlas/output hashes, environment versions, seeds, repairs, and validity status. Core files: [extractor](../../src/p2_1_extract.py), [validity audit](../../src/p2_1_validity.py), [analysis](../../src/p2_1_analyze.py), [normalization sidecar](../../src/p2_1_norms.py), and [summarizer](../../src/p2_1_summarize.py). Large `.npz` atlas arrays are intentionally local and Git-ignored; lightweight manifests, tables, scripts, and this report are versionable. Reproduction of the atlas requires the original locally retained D5 discovery checkpoints and archived prediction tables.
