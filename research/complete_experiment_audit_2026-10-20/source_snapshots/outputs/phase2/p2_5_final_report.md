# P2.5 final report — Functional-Response Compression Test

## 1. Executive summary

**VALID; P2.5-D — semantic/static description suffices under the frozen classification.** The four natural response families passed a strict disjoint-score algebra audit. Three observed response coordinates predicted some withheld consequences better than a matched four-coefficient static hidden-state description, but did not beat the matched semantic comparator consistently across families and both structural holdouts. At matched cost F wins all four query-family×holdout comparisons and loses all four candidate-family×holdout comparisons against S4; the fuller 51-coefficient semantic reference beats F in 5/8. F requires six archived network score observations per test block, so its four fitted coefficients are not a complete cost measure. The tested functional description does not establish a more economical/reusable computational organization.

## 2. Validity status

All 19 required gates passed. Source manifest/checkpoint/atlas hashes, 12-model scope, score replay, response formulas, train-only preprocessing, four-family holdouts, and output artifacts were audited. 1728 saved-model metric values replayed with maximum absolute difference 0. No protected seeds, task-network training, activation intervention, or P2.6 work.

## 3. Motivation from prior studies

D5 showed support-topology effects; D8 showed local endpoint availability need not imply effective use. P2.1 found partial semantic compression and informative static participant state. P2.2 remains INVALID. P2.3 was VALID-D (only high-capacity internal advantage), and P2.4 was VALID-D (no robust participant specificity). These findings motivate testing consequence-based descriptions without assuming an operator, class, or state quotient.

## 4. Frozen scientific question

Can a few observed, natural downstream response differences predict algebraically independent held-out response families on structural holdouts at lower/equal statistical complexity than frozen P2.1 semantic and P2.3 static-state descriptions? The cross-seed test asks whether semantically aligned functional coordinates reuse more naturally than raw hidden coordinates.

## 5. What P2.5 does not assume

No predictive-response state, discrete response class, quotient, operator algebra, transformation law, routing grammar, causal mechanism, or language/universal principle is assumed or fitted. No response-vector clustering is performed.

## 6. Provenance and integrity

P2.1/P2.3/P2.4 manifest SHA-256 values: `24a0068d688f94eacd0703ad059bb78984dcf545af904922d66b51ec0ce6874f`, `96d1c41b221d4ca26f2b2147adbe95004522bedd41ba715967c28d6b4d8161fa`, `a89afa0b914f478e49e5908aeb6f871e23f7933c2ddd8f9389f89785e2d8c6e2`. P2.5 preregistration SHA-256 `eb5c0a0d8201bbfd63757d0b9bddaa3f1ff462470740b79474a1aa0d726093a9`; frozen fitting source SHA-256 `a28891f2ea2441ed9b7888ed865c8ddeec00ad433c775cd1d783d687fd6e5d54`. All parent artifact hashes passed. A bounded 36-check archived score/response replay had maximum absolute discrepancy zero. Two prefit *technical* read-guard repairs are logged in the library artifact: parent manifest field-name compatibility and float64 subtraction for replay; neither changed the scientific design or fit.

## 7. Unit of analysis

One natural P2.1 4×4 query×candidate block, with semantic member 0 on each side as anchor. There are 384 blocks per model and 4608 discovery blocks overall. Eight source models supplied 410 legal training blocks after frozen structural/combination exclusions. The semantic candidate permutation maps physical positions to member indices; no new forward pass was required.

## 8. Frozen response library

The four coordinates cover query/candidate × endpoint-or-role/event changes. They were chosen from task semantics, not response magnitudes. S endpoint changes are source (query) and target (candidate); R endpoint changes are role reversals.

| Family | S candidate set formula | R candidate set formula | Calls |
|---|---|---|---:|
| Q_ENDPOINT | `+1*s(q1,c0) + -1*s(q0,c0)` | `+1*s(q1,c0) + -1*s(q0,c0)` | 2 |
| Q_EVENT | `+1*s(q3,c1) + -1*s(q0,c1)` | `+1*s(q2,c1) + -1*s(q0,c1)` | 2 |
| C_ENDPOINT | `+1*s(q2,c2) + -1*s(q2,c0)` | `+1*s(q3,c1) + -1*s(q3,c0)` | 2 |
| C_EVENT | `+1*s(q3,c3) + -1*s(q3,c0)` | `+1*s(q2,c2) + -1*s(q2,c0)` | 2 |

## 9. Response formulas

For each coordinate, the exact formula is the positive archived score cell minus a negative archived score cell shown above. Event substitutions preserve the corresponding ordered endpoints; endpoint/role substitutions preserve event. All candidate cells are indexed by semantic member, never physical position. Scores come from frozen network atlas consequences, not surrogate outputs.

## 10. Algebraic dependency audit

All 8 S/R target-family combinations were classified SAFE before fitting. Each candidate family uses eight distinct score cells for its four coordinates. In each family holdout, the target uses two cells absent from all three observed input coordinates; the 4×16 coefficient matrix has rank 4. No target is an input, reverse duplicate, or exact linear combination of inputs. This excludes exact score-algebra reconstruction but not statistical correlations or nuisance geometry. Full signed term expansions are in `analysis/response_algebra_audit.json`.

## 11. Functional description F

F receives exactly the three *other* response-family scalars for each target-family fold, plus an intercept in a ridge readout: four fitted coefficients. At test time these three observed responses require six distinct archived network score cells. The held-out target's two score cells are used only after prediction for evaluation.

## 12. Semantic comparator S

S uses the existing P2.1 M1 50 shared semantic/context features at anchor q0,c0. A fixed Gaussian projection provides the matched three-coordinate S4 input; projection seeds and widths are outcome-independent. S16/S32/S64/S128 and unprojected S_FULL (51 fitted coefficients) are frozen cost references, not outcome-selected primary replacements. S acquires descriptors from metadata, not extra score calls.

## 13. Static hidden comparator H

H uses only the previously frozen P2.3 STATE-1 query and candidate CLS/source/target state at anchor q0,c0, 288 raw dimensions. Fixed Gaussian projection yields three coordinates for matched H4; H16/H32/H64/H128 are frozen cost references. Endpoint token selection remains privileged semantic side information. No alternative layer, token, head, or neuron was searched.

## 14. Complexity and observation cost

Primary statistical complexity is matched at four fitted coefficients. F materializes three float64 values (24 bytes) per block but requires six archived score observations; the full response library/evaluation uses eight cells. S4 materializes three projected values plus a fixed 50×3 matrix; H4 uses a fixed 288×3 matrix and previously captured hidden states. S/H additional task-forward acquisition costs are not equated to F's score calls. Cost should be interpreted as a two-axis Pareto comparison, not one scalar.

| Description | Primary fitted coefficients | Raw input | Fixed projection elements | Functional response coordinates | Archived score calls to construct input |
|---|---:|---:|---:|---:|---:|
| F | 4 | 3 | 0 | 3 | 6 |
| S4 | 4 | 50 | 150 | 0 | 0 |
| H4 | 4 | 288 | 864 | 0 | 0 |
| S_FULL | 51 | 50 | 0 | 0 | 0 |

All arm/budget/fit coefficient counts and matrix sizes are in `tables/complexity_accounting.csv`.

## 15. Fitting discipline

One pooled shared ridge rule per target family and arm used A/B/C/D seeds 1701/1702, 410 legal source blocks. Alpha candidates [0.01, 1, 100] were selected by a source-only hashed validation split and refit on all legal source blocks. Feature means/scales were computed only on fit rows; fixed Gaussian projections and permutation maps were outcome-independent. All F/S/H fits saw exactly the same target labels and legal rows. Mean baseline used source-only targets; no MLP or kernel search.

## 16. Response-family holdouts

Four whole-family folds each withhold one named target from F inputs while observing the other three. The held-out target is a supervised label only on legal *source training* blocks; no target-family value or target score cell enters test input, projection, scaling, feature selection, or test-time observation. The frozen F1/F2/F3 subset ladder takes the first one/two/three remaining families in declared library order.

## 17. Structural holdouts

Whole-context and candidate-configuration are co-primary P2.1 masks. Source training excludes both and any block whose eight library cells touch P2.1 combination holdout. Event-disagreement scores are necessarily used for the declared event responses and are treated as already-observed discovery stressors, not independent confirmation. Context holds about ten blocks/model and thus individual-model estimates can be noisy; exact block counts appear in per-model CSV.

## 18. Primary F versus S results

F has lower mean RMSE than S4 in 4/8 target-family×holdout cells—all query-target comparisons—and S4 wins in the other 4/8—all candidate-target comparisons. S_FULL wins in 5/8 despite its larger 51-coefficient cost and zero functional score calls. The family-axis split, not an averaged grand score, determines the conclusion. Lower is better.

| Target | Holdout | F (4) | S4 | H4 | S_FULL (51) | Mean | Permuted F | Wrong-family F |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q_ENDPOINT | context | 2.903 | 2.962 | 4.526 | 2.561 | 4.366 | 2.983 | 3.319 |
| Q_ENDPOINT | candidate_configuration | 2.890 | 3.796 | 4.249 | 3.192 | 4.160 | 3.074 | 3.113 |
| Q_EVENT | context | 2.314 | 2.531 | 2.861 | 2.423 | 2.818 | 2.373 | 2.778 |
| Q_EVENT | candidate_configuration | 2.662 | 3.058 | 3.239 | 3.002 | 3.215 | 2.775 | 3.190 |
| C_ENDPOINT | context | 3.522 | 3.351 | 3.333 | 2.381 | 3.350 | 3.394 | 3.455 |
| C_ENDPOINT | candidate_configuration | 3.808 | 3.635 | 3.953 | 3.020 | 3.960 | 3.851 | 4.001 |
| C_EVENT | context | 2.834 | 2.717 | 3.549 | 2.436 | 3.558 | 2.724 | 3.308 |
| C_EVENT | candidate_configuration | 2.889 | 2.673 | 3.660 | 2.454 | 3.627 | 2.959 | 3.620 |

![P2.5 response-family RMSE](../../phase2/p2_5/figures/family_holdout_rmse.svg)

## 19. Primary F versus H results

F has lower mean RMSE than H4 in 7/8 target-family×holdout cells, failing only C_ENDPOINT on whole-context. This is a bounded positive indication that a few response observations can outperform compressed static STATE-1 coordinates at equal fitted readout count. It is insufficient for the main functional-compression claim because the semantic comparator often performs better and F pays six score calls.

## 20. Per-model consistency

Score cells are not treated as independent replications; the table counts models with lower F RMSE. Full model/seed paired values, mean-absolute errors, meaningful sign accuracy, and source block counts are in `tables/per_model_metrics.csv` and `tables/paired_F_vs_comparators.csv`.

| Target | Holdout | F better than S4 (models) | F better than H4 | F better than permuted | F better than wrong-family |
|---|---|---:|---:|---:|---:|
| Q_ENDPOINT | context | 9/12 | 11/12 | 10/12 | 11/12 |
| Q_ENDPOINT | candidate_configuration | 12/12 | 12/12 | 10/12 | 12/12 |
| Q_EVENT | context | 11/12 | 11/12 | 9/12 | 10/12 |
| Q_EVENT | candidate_configuration | 12/12 | 12/12 | 10/12 | 12/12 |
| C_ENDPOINT | context | 3/12 | 3/12 | 2/12 | 3/12 |
| C_ENDPOINT | candidate_configuration | 3/12 | 10/12 | 7/12 | 12/12 |
| C_EVENT | context | 7/12 | 10/12 | 5/12 | 8/12 |
| C_EVENT | candidate_configuration | 5/12 | 12/12 | 7/12 | 12/12 |

## 21. Response-family breakdown

At matched four-coefficient cost, F beats S4 for both query-target families on both structural holdouts and loses to S4 for both candidate-target families on both holdouts. F beats H4 in 7/8 cells. Candidate endpoint/role is especially difficult: on context F is worse than both matched S and H and even worse than a permuted signature. Thus the response direction—not a single grand average—organizes the disagreement; no family was dropped after observing outcomes.

## 22. Response-subset compression

The nested subset order was fixed by response-family order, not performance. Adding coordinates helps some targets and harms others; no greedy selection or response clustering was performed.

| Target | Holdout | F1 (2 coeff.) | F2 (3) | F3 (4) |
|---|---|---:|---:|---:|
| Q_ENDPOINT | context | 3.629 | 3.676 | 2.903 |
| Q_ENDPOINT | candidate_configuration | 3.448 | 3.366 | 2.890 |
| Q_EVENT | context | 2.342 | 2.325 | 2.314 |
| Q_EVENT | candidate_configuration | 2.668 | 2.671 | 2.662 |
| C_ENDPOINT | context | 3.363 | 3.349 | 3.522 |
| C_ENDPOINT | candidate_configuration | 3.848 | 3.849 | 3.808 |
| C_EVENT | context | 2.722 | 2.714 | 2.834 |
| C_EVENT | candidate_configuration | 2.915 | 2.906 | 2.889 |

## 23. Permutation and wrong-family controls

The paired signature permutation moves the three observed response coordinates together across matched structural/family/kind/support strata without using target values. Wrong-family rotates their order per block by a frozen metadata hash, preserving numeric values while breaking declared coordinate identities. F beats both controls in some folds, confirming nontrivial configuration-correspondence information there, but fails to beat permuted F on whole-context C_ENDPOINT and C_EVENT. Controls therefore do not support a uniform functional organization claim. Source control maps and singleton counts are in `analysis/response_library_prereg.json`.

## 24. Cross-seed reuse

A/B 1701/1702 fits were applied without target fitting, calibration, or alignment to A/B 1703. Functional coordinates are semantically named across models, whereas H uses unaligned hidden coordinates. The frozen B criterion nevertheless fails; see all target-family results below. Raw H transfer failure alone is not a mechanism claim.

| Target | Holdout | F | S4 | H4 | F−S4 | F−H4 |
|---|---|---:|---:|---:|---:|---:|
| Q_ENDPOINT | context | 3.543 | 3.058 | 5.034 | +0.485 | -1.491 |
| Q_ENDPOINT | candidate_configuration | 2.839 | 2.744 | 4.571 | +0.095 | -1.732 |
| Q_EVENT | context | 2.606 | 2.475 | 2.724 | +0.132 | -0.117 |
| Q_EVENT | candidate_configuration | 2.389 | 2.382 | 2.830 | +0.007 | -0.441 |
| C_ENDPOINT | context | 4.000 | 3.780 | 3.872 | +0.220 | +0.128 |
| C_ENDPOINT | candidate_configuration | 3.773 | 3.668 | 3.891 | +0.105 | -0.119 |
| C_EVENT | context | 3.276 | 2.655 | 3.700 | +0.620 | -0.424 |
| C_EVENT | candidate_configuration | 2.820 | 3.003 | 3.978 | -0.183 | -1.158 |

## 25. Cross-graph descriptive reuse

A/B source fits were also applied without adaptation to C/D models and their frozen structural holdouts. This is descriptive, not protected confirmation.

| Target | Holdout | F | S4 | H4 |
|---|---|---:|---:|---:|
| Q_ENDPOINT | context | 3.441 | 3.538 | 5.242 |
| Q_ENDPOINT | candidate_configuration | 3.903 | 4.525 | 4.835 |
| Q_EVENT | context | 3.023 | 3.230 | 3.434 |
| Q_EVENT | candidate_configuration | 3.703 | 3.890 | 3.841 |
| C_ENDPOINT | context | 3.950 | 3.596 | 3.811 |
| C_ENDPOINT | candidate_configuration | 4.981 | 4.633 | 4.952 |
| C_EVENT | context | 3.456 | 3.456 | 4.159 |
| C_EVENT | candidate_configuration | 4.097 | 3.338 | 4.640 |

## 26. Failure and nuisance analysis

F's performance varies by family, condition/seed, candidate family, evaluation kind, and support status. Full counts and error values for F/S4/H4 are in `tables/nuisance_metrics.csv`. Context holdouts have small per-model block counts; one response family can reverse the aggregate pattern. This descriptive audit cannot rule out nuisance geometry. Most importantly, S frequently outperforms F despite F's six score observations, and the candidate endpoint response resists F compression.

| Stratum | F response-RMSE range across model×value rows |
|---|---:|
| candidate_family | 1.053–7.476 |
| evaluation_kind | 0.594–6.367 |
| query_support | 0.594–6.367 |
| candidate_support | 0.594–6.367 |

## 27. Frozen classification

**P2.5-D — SEMANTIC/STATIC DESCRIPTION SUFFICES.** The mechanical rule selected D because S4 or H4 matched/beat F in at least two of four family folds on *each* primary structural holdout, and the stringent A criterion failed. It does not mean F carries no signal; it means the tested functional signature did not establish superior economical organization against both competitors.

## 28. What P2.5 establishes

Four algebraically independent natural response families are available in the existing atlas and can be used for leakage-safe cross-family prediction. At four fitted coefficients, three observed consequences often beat a correspondingly compressed static hidden-coordinate description, but not the semantic description consistently. The response observations encode some genuine configuration correspondence, yet the evidence is not uniformly reusable or Pareto-dominant once score-call cost is counted.

## 29. What P2.5 does not establish

No predictive state, response equivalence class, causal mechanism, transformation operator, grammar, language mechanism, or universal transformer principle is established. The experiment does not compare every possible functional library or static/semantic model. It uses cached discovery scores; no protected prospective test or new task-model training occurred.

## 30. Updated surviving explanation set

Bounded semantic interpolation remains a strong description. Functional responses offer partial information beyond low-dimensional static hidden coordinates, but the tested four-family signature is not generally superior. Static-state misspecification, semantic/family heterogeneity, nuisance geometry, and score-call economics remain viable explanations. P2.1 remains VALID; P2.2 INVALID; P2.3 VALID-D; P2.4 VALID-D; D6/D9 nonconfirmatory.

## 31. Most informative unresolved disagreement

Why three natural response observations help prediction of some withheld consequences (especially versus H) yet underperform semantic descriptors and sometimes a permuted signature is unresolved. This could reflect heterogeneous response families, sampling/noise, or the particular bounded library; P2.5 does not choose among them after seeing outcomes.

## 32. Reproducibility summary

Frozen preregistration `eb5c0a0d8201bbfd63757d0b9bddaa3f1ff462470740b79474a1aa0d726093a9` and fitting implementation `a28891f2ea2441ed9b7888ed865c8ddeec00ad433c775cd1d783d687fd6e5d54`; eight SAFE S/R family-fold algebra checks; 12 discovery models; 216 frozen ridge/mean fits across primary/cross-seed/cross-graph modes; all coefficients, train-only normalizers, alphas, raw tables, control maps, figure, validity audit, and hashes are in the reproducibility manifest. No scientific protocol repair or outcome-dependent family change occurred. Two prefit technical read/replay corrections are documented. P2.5 stops here.
