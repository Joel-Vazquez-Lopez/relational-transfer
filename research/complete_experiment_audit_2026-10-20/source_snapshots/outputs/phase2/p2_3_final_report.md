# P2.3 final report — Interaction-Capable Matched-Budget Residual Test

## 1. Executive summary

P2.3 is VALID as a bounded discovery comparison. The frozen mechanical category is **P2.3-D — high-complexity-only advantage**. At 256 incremental fitted coefficients, the participant-state arm has lower mixed-response RMSE than the matched semantic arm on both primary structural holdouts, in 10/12 model pairs per holdout. This pattern is absent at nominal 32, 64, and 128; it does not establish a compact rule.

## 2. Validity status

All 15 frozen validity gates passed (see `analysis/final_validity_audit.json`). Twelve P2.1 discovery atlases, 73,728 score cells, and every recorded P2.1 source/checkpoint/normalization hash were verified. No protected seed was accessed and no task network was trained. P2.2 remains INVALID; D6/D9 supply no confirmatory evidence. This is still a previously observed discovery set, not prospective confirmation.

## 3. P2.2 failure being corrected

P2.2 compared a purely additive internal linear arm, whose query×candidate mixed finite difference was identically zero, with an interaction-capable semantic arm. Its primary comparison is permanently INVALID; its outcomes were not used to select P2.3 features, budgets, or controls.

## 4. Frozen hypotheses

H0: bounded semantic/contextual interaction suffices at comparable capacity. H1: participant-state interaction provides a compact predictive advantage across structural holdouts. Neither hypothesis claims a causal mechanism or operator law.

## 5. Prefit expressivity proof and tests

The audit passed before fitting. `MIX(g)=(zq1−zq0)ᵀW(zc1−zc0)`, because additive terms cancel. Synthetic tests exercised the actual projection/feature code and exceeded the frozen 1e-8 tolerance for both arms at every budget.

| Nominal | Dimensions | Semantic synthetic MIX | Internal synthetic MIX | Fitted parameters |
|---:|---:|---:|---:|---:|
| 32 | 4×5 | -0.158957 | 0.0369774 | 30 |
| 64 | 7×7 | -0.205671 | 0.0983487 | 64 |
| 128 | 10×11 | -0.184944 | -0.0517822 | 132 |
| 256 | 15×15 | -0.220138 | -0.0610284 | 256 |

## 6. Atlas/provenance audit

P2.1 reproducibility manifest SHA-256: `24a0068d688f94eacd0703ad059bb78984dcf545af904922d66b51ec0ce6874f`. P2.3 preregistration SHA-256: `5e255553fbc009282d84d2d2f0e13138d3adfabe993866641d1bf1e743ab8794`; human preregistration SHA-256: `e69795fe481b6114b49af2813af42943133d97499a367b973fc9e1ce651699a6`; frozen fitting implementation SHA-256: `2d731da1ae590338ed1227fd310dd841d22d8290ccb558aa5530fb3c9763864e`. All source hash checks passed both prefit and postfit.

## 7. Residual construction and OOF discipline

P2.1 M1 has 50 features and 51 fitted coefficients. The primary source pool used 14990 core score cells from A–D seeds 1701/1702. Five OOF folds each excluded its target rows (zero intersections); each M1 fold chose alpha only on source-core training rows. The full source-core M1 predicts evaluation cells; a single shared correction fits source OOF residuals. No target-seed fitting occurs.

## 8. Internal participant-state representation

Only P2.1 post-block-0 query CLS/source/target and candidate CLS/source/target selected states enter, 144 dimensions per side before fixed projection. The semantic labels required to select endpoint-bearing tokens are privileged side information. No all-token, layer, head, or neuron search was performed.

## 9. Semantic interaction representation

Each side has 42 declared one-hot semantic/context descriptors: source 6, target 6, event 2, support 2, family 2, evaluation kind 4, sorted context triple 20. No activation, answer position, candidate physical position, or seed ID enters this arm.

## 10. Exact algebraic interaction capacity

Both arms use an intercept, additive query and candidate terms, and all projected query×candidate products in a shared convex ridge fit. Both generate nonzero mixed finite differences. The interaction ablation zeros fitted cross-product coefficients without refitting. Projections are fixed Gaussian maps with zero fitted coefficients, but their realized materialized size is recorded.

## 11. Complexity matching

The S and I arms match fitted counts exactly at each budget; 30 is 6.25% below nominal 32 and 132 is 3.125% above nominal 128. No primary increment exceeds 256.

| Nominal | Exact incremental | Additive | Interaction | M1+incremental | S/I fixed projection elements |
|---:|---:|---:|---:|---:|---:|
| 32 | 30 | 10 | 20 | 81 | 378/1296 |
| 64 | 64 | 15 | 49 | 115 | 588/2016 |
| 128 | 132 | 22 | 110 | 183 | 882/3024 |
| 256 | 256 | 31 | 225 | 307 | 1260/4320 |

## 12–14. Primary holdouts and mixed responses

Means are unweighted over 12 models. Lower is better. The participant-state 256 advantage is partly a semantic-arm degradation at that capacity, so its I−S gap is not all absolute improvement beyond M1.

| Budget | Holdout | M1 mixed | S mixed | I mixed | I−S | I-better models | I score RMSE | S score RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | context | 7.2516 | 7.1739 | 7.1967 | +0.0228 | 6/12 | 3.4274 | 3.4675 |
| 32 | candidate_configuration | 7.3725 | 7.3317 | 7.3866 | +0.0549 | 3/12 | 3.5184 | 3.5157 |
| 64 | context | 7.2516 | 7.2251 | 7.2342 | +0.0091 | 6/12 | 3.4451 | 3.4713 |
| 64 | candidate_configuration | 7.3725 | 7.3750 | 7.3558 | -0.0192 | 6/12 | 3.5447 | 3.5763 |
| 128 | context | 7.2516 | 7.0242 | 7.3032 | +0.2790 | 3/12 | 3.8395 | 3.5520 |
| 128 | candidate_configuration | 7.3725 | 7.2411 | 7.3717 | +0.1305 | 6/12 | 3.8768 | 3.4800 |
| 256 | context | 7.2516 | 7.4512 | 7.1332 | -0.3179 | 10/12 | 3.5494 | 3.9525 |
| 256 | candidate_configuration | 7.3725 | 7.6367 | 7.2655 | -0.3712 | 10/12 | 3.6217 | 4.1229 |

![Mixed-response budget comparison](../../phase2/p2_3/figures/mixed_response_budget.svg)

## 15. Finite responses

Meaningful finite sign uses frozen `abs(true finite)>0.1`. Endpoint redirection is Family-S only.

| Budget | Holdout | S finite RMSE | I finite RMSE | S sign acc. | I sign acc. | S redirection MAE | I redirection MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | context | 4.4674 | 4.4931 | 0.7523 | 0.7478 | 2.8196 | 2.7253 |
| 32 | candidate_configuration | 4.5858 | 4.6400 | 0.7550 | 0.7478 | 3.5091 | 3.4952 |
| 64 | context | 4.4888 | 4.5614 | 0.7424 | 0.7402 | 2.7462 | 2.7353 |
| 64 | candidate_configuration | 4.6323 | 4.6594 | 0.7522 | 0.7459 | 3.4780 | 3.4655 |
| 128 | context | 4.4582 | 4.8899 | 0.7554 | 0.7246 | 2.6590 | 2.7450 |
| 128 | candidate_configuration | 4.5440 | 4.9150 | 0.7562 | 0.7272 | 3.3219 | 3.5008 |
| 256 | context | 4.9744 | 4.6901 | 0.6974 | 0.7347 | 2.2787 | 2.8395 |
| 256 | candidate_configuration | 5.1461 | 4.7599 | 0.6712 | 0.7339 | 3.3449 | 3.3562 |

## 16. Per-model consistency

At 256, I beats S on mixed RMSE in 10/12 models on context and 10/12 on candidate-configuration. Full condition/seed/holdout pairs and both score/mixed deltas are in `tables/paired_internal_vs_semantic.csv`; individual model metrics are in `tables/per_model_metrics.csv`. Score cells are not treated as independent scientific replications.

## 17–18. Interaction ablation and controls

At 256 the explicit interaction term improves *mean* mixed RMSE versus zeroing it; a second projection retains the I-versus-S mean direction. But the control differences are not consistently signed across models: wrong-state 8/12 on each primary holdout, permutation 7/12 and 8/12, and ablation 7/12 and 8/12. Thus robust participant specificity is **not** established. The paired permutation moves whole query/candidate state pairs within exact frozen split/factor strata, destroying example correspondence but preserving pair internals; singleton strata cannot be permuted. The wrong-state CLS may still contain endpoint information, further limiting specificity. Controls use the same cross-product family and fitted budget; ablation is the intentional additive-only exception.

| Holdout | I mixed | Ablation mixed | Wrong-state mixed | Alternate-projection mixed | Paired-permutation mixed |
|---|---:|---:|---:|---:|---:|
| context | 7.1332 | 7.2516 | 7.2882 | 7.1344 | 7.2680 |
| candidate_configuration | 7.2655 | 7.3725 | 7.4094 | 7.2922 | 7.3736 |

| Holdout | I better than S | I better than wrong | I better than permutation | I better than alternate | I better than ablation |
|---|---:|---:|---:|---:|---:|
| context | 10/12 | 8/12 | 7/12 | 5/12 | 7/12 |
| candidate_configuration | 10/12 | 8/12 | 8/12 | 6/12 | 8/12 |

## 19. Cross-seed status

A/B seeds 1701/1702 were fit; A/B 1703 was evaluated without calibration or alignment. At 256 the I advantage does not transfer in mixed RMSE. Raw coordinate mismatch makes this a descriptive stress test, not a refutation of within-model state usefulness.

| Budget | Context I−S mixed | Configuration I−S mixed |
|---:|---:|---:|
| 32 | +0.0760 | +0.0690 |
| 64 | -0.1780 | +0.0309 |
| 128 | +0.7527 | +0.7332 |
| 256 | +0.1915 | +0.3294 |

## 20. Previously observed event-disagreement stress test

P2.1 substitution-family event disagreement is secondary and previously observed; its score-only result is not independent confirmation.

| Budget | I score RMSE | S score RMSE | I−S |
|---:|---:|---:|---:|
| 32 | 4.1651 | 4.1551 | +0.0100 |
| 64 | 4.1922 | 4.2699 | -0.0776 |
| 128 | 4.6531 | 4.1655 | +0.4875 |
| 256 | 4.3315 | 5.3248 | -0.9933 |

## 21. Nuisance/failure analysis

A postfit deterministic reconstruction of the exact frozen fits reproduced all published score RMSEs to 1e-10 and stratified their errors by physical position, object slots, candidate permutation, kind, family, support, baseline-score magnitude, and nondeployable true-answer position. Condition and seed are explicit grouping columns. This is descriptive: differing ranges do not identify a cause, and no bounded nuisance audit proves absence of leakage. The largest-budget semantic fit worsened relative to lower budgets, a key failure mode in interpreting I−S.

| Stratum | Cells across 12 models | RMSE range across model×level rows |
|---|---:|---:|
| candidate_physical_position | 10128 | 2.736–5.113 |
| object_source_slot | 10128 | 2.659–5.204 |
| object_target_slot | 10128 | 2.623–5.451 |
| candidate_permutation | 10128 | 2.224–6.457 |
| evaluation_kind | 10128 | 2.790–5.306 |
| candidate_family | 10128 | 2.637–5.333 |
| query_support | 10128 | 2.898–5.830 |
| candidate_support | 10128 | 2.750–6.020 |
| baseline_score_magnitude_bin | 10128 | 1.710–5.930 |
| correct_answer_position_NONDEPLOYABLE | 10128 | 2.519–5.290 |

## 22. Complexity/Pareto analysis

Nominal 32/64/128 show no consistent mixed-response I advantage across both primary holdouts. The only I-versus-S directional result appears at exact 256 incremental coefficients (307 including M1), the preregistered primary ceiling. Control separation at 256 is present in means but weak at the individual-model level. This is a high-complexity-only result, not compact 32–128 compression. Absolute I gain over M1 is modest relative to the I−S gap because S degrades at 256.

## 23. Frozen interpretation category

**P2.3-D — HIGH-COMPLEXITY-ONLY ADVANTAGE.** The mechanical decision is stored in `analysis/decision.json`. It is not a significance claim and no outcome-dependent percentage threshold was added.

## 24. What P2.3 establishes

Within these 12 D5 discovery models and the frozen low-dimensional Gaussian projection family, participant-state cross-products at the largest allowed budget have lower *mean* held-out mixed residual error than the declared semantic cross-products and wrong-state control on both primary holdouts. The explicit cross-side term improves the mean relative to its ablation, and a second projection shows a similar I-versus-S direction. Individual-model control separation is inconsistent, so this is not robust participant specificity.

## 25. What P2.3 does not establish

It does not establish a compact 32–128 rule, cross-seed reusable coordinates, a causal mechanism, a transformer grammar/operator law, a predictive-state quotient, support causality, or a language/universal principle. The semantic competitor is one bounded family; the wrong-state control is not a clean semantic null.

## 26. Updated surviving explanation set

Compact semantic interaction, richer but untested semantic interpolation, model-specific internal interactions, and a high-capacity participant-state interaction remain possible. P2.1's broader explanation set is not collapsed by this discovery analysis. P2.2 stays INVALID; D6/D9 remain nonconfirmatory.

## 27. Most informative unresolved disagreement

Whether a 256-coefficient participant-state advantage reflects meaningful reusable interaction structure or a capacity/projection-specific failure of the bounded semantic competitor remains unresolved. Raw unaligned A/B seed transfer does not show the 256 advantage.
