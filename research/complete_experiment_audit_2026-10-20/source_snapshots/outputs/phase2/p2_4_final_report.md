# P2.4 final report — Participant-Specific Interaction Test

## 1. Executive summary

**VALID; P2.4-D — nonspecific internal-state signal.** The exact P2.3 256 participant model replayed with zero discrepancy. P's mean mixed-response RMSE is lower than MATCHED_OTHER and the candidate-only DISTRACTOR on both primary holdouts, but only 7–8 of 12 model pairs favor P and seed 1703 reverses those mean directions. ROLE_SWAP and GLOBAL are comparable or better in aggregate on at least one holdout. The frozen specificity criterion therefore fails. This is a negative specificity result, not evidence that relational computation is absent.

## 2. Validity status

All 21 frozen validity gates passed. The P2.3 manifest and all 23 hashed P2.3 artifacts, P2.1 atlas/checkpoint hashes, 12-model inventory, frozen donor indices, identical 256-coefficient capacity, and unchanged holdouts were verified. Independent manual mixed-response reconstruction agreed on 120 arm×model×holdout results within 0. No protected seed was touched and no task network was trained.

## 3. Scientific motivation from P2.3

P2.3 was VALID, category D: an I-versus-semantic mixed-response advantage appeared only at exact 256 incremental coefficients, without robust wrong-state separation or raw cross-seed transfer. P2.4 tests whether true participant assignment is specifically necessary. P2.2 remains permanently INVALID and contributes no scientific evidence.

## 4. Frozen hypotheses

H_SPEC: correct endpoint states and example correspondence improve mixed-response prediction over hard internal controls. H_GENERIC: generic/contextual hidden-state or projection/capacity information suffices. H_ROLE: correct source/target organization matters beyond endpoint presence. These are predictive, not causal-mechanism, hypotheses.

## 5. P2.3 replay

The stored P2.3 I_256 weight (alpha 100.0) replayed 168 published score, finite, mixed, and redirection metric comparisons across all 12 models and frozen holdouts, with maximum absolute error 0 against a 1e-10 limit. P was not refitted in the primary comparison.

## 6. Provenance and integrity audit

P2.3 manifest SHA-256 `96d1c41b221d4ca26f2b2147adbe95004522bedd41ba715967c28d6b4d8161fa`; P2.4 preregistration SHA-256 `3636880102c5ecf7f095653fa4e84950be291f5b8f81784b27f79180b2e84487`; frozen P2.4 fitting implementation SHA-256 `66b8b25e8b29925624e252940b986713ef4bfec8d6aeeabb4735f785a0814a09`. The P2.3 source/manifest, all 12 atlas identities, and P2.1 source/checkpoint integrity were checked before fitting and at finalization. P2.2/P2.3 were not modified.

## 7. Exact internal model family

Each arm uses raw query/candidate 144-dimensional post-block-0 state inputs, the same fixed P2.3 Gaussian projection matrices to 15 dimensions per side, and a convex ridge correction with intercept, 15 query-additive, 15 candidate-additive, and 225 explicit cross-product coefficients. Incremental fitted count is exactly 256 (307 with common M1). No rank, projection, budget, layer, or hyperparameter search occurred.

## 8. True-participant condition P

P uses current-example query CLS/source/target and candidate CLS/source/target states selected by the P2.1 semantic metadata. This endpoint token selection is privileged side information. P's frozen P2.3 coefficients are reused exactly in the primary pool.

## 9. ROLE_SWAP

Within the same example, query and candidate source/target state slots were each exchanged before applying the identical P2.3 projection matrices. CLS, both endpoint identities, scores, context, and labels remained unchanged. This changes the state ordering fed to the surrogate, not the underlying task example. Since the control is refit and preprojection swapping is a reversible relabeling of the full raw vector, near-equal performance would not by itself prove role information is absent; fixed dimensionality reduction can also affect the comparison.

## 10. MATCHED_OTHER

Current-example CLS remains, while ordered source/target endpoint vectors come from other rows in the same model and identical four-flag split signature, with a different meaning and block. Query-side and candidate-side donors are selected separately, so cross-side donor coherence is not preserved; this is an explicit control limitation. The frozen hierarchy relaxes triple, kind, family, support, then event as needed; no score/prediction/residual/outcome enters matching. Across 24 model×side maps, assignment counts were L0_assigned=129273, L1_assigned=13656, L2_assigned=3006, L3_assigned=1293, L4_assigned=216, L5_assigned=12. Exact donor indices are frozen in `analysis/matched_other_donors.npz` (SHA-256 `d654121a7aac9a67cba1807fe3e22e5ca4a863a0a73c5b25726e64ef51da0fce`); model-level fallback and singleton rates are in `tables/control_matching_statistics.csv`. The small deepest-fallback fraction weakens exact event/support matching for those rows and is reported, not selected away.

## 11. DISTRACTOR definition and limitation

The query side stays P because no genuinely comparable nonparticipant query object token exists. The candidate side keeps current CLS and duplicates the unique current-example nonendpoint candidate object state into both endpoint slots. This is a semantically valid candidate-side test only; it is not a symmetric endpoint control, and duplication changes within-side covariance.

## 12. GLOBAL definition

Each side repeats its current-example CLS vector in all three 48-dimensional slots. It has no explicit endpoint-object/token slots and uses the same 256-coefficient interaction family/projection. CLS may encode endpoint information, so success is not proof that endpoint representations are irrelevant.

## 13. Prefit control audit

All control constructions and donor indices were fixed before control fitting; every actual transformed-state feature path produced nonzero synthetic mixed finite differences above the frozen 1e-8 tolerance. P2.3 replay and provenance passed.

| Arm | Synthetic MIX | Raw dims | Projected dims | Incremental / interaction coefficients |
|---|---:|---:|---:|---:|
| DISTRACTOR | 7.5757 | 144+144 | 15+15 | 256/225 |
| GLOBAL | -7.4201 | 144+144 | 15+15 | 256/225 |
| MATCHED_OTHER | -13.1132 | 144+144 | 15+15 | 256/225 |
| P | -11.8177 | 144+144 | 15+15 | 256/225 |
| ROLE_SWAP | 14.1145 | 144+144 | 15+15 | 256/225 |

## 14. Exact complexity parity

P, ROLE_SWAP, MATCHED_OTHER, candidate-only DISTRACTOR, and GLOBAL all have 31 additive and 225 interaction fitted coefficients, 256 incremental in total. Projection matrices are the same P2.3 I-arm matrices with zero fitted projection coefficients. Same source pool, targets, optimizer family, ridge alpha candidates, and evaluation masks apply.

## 15. Fitting and OOF discipline

Primary source pool: A/B/C/D seeds 1701/1702, 14990 core score cells. Exactly P2.3 five-fold out-of-fit M1 targets were reused via the frozen function; all fold train/target intersections are zero. Control models were one shared pooled fit per arm, using only legal source-core rows and frozen alpha grid [0.01, 1, 100]. No target-seed calibration or outcome-dependent preprocessing. Primary selected alphas: DISTRACTOR=100, GLOBAL=1, MATCHED_OTHER=100, P=100, ROLE_SWAP=100.

## 16. Primary holdouts

Whole-context and candidate-configuration are co-primary, with the unchanged P2.1 semantic-member reordering and complete 4×4 block mixed contrasts. Combination and substitution family are secondary score-only stress tests. Previously observed event disagreement is not independent confirmation.

## 17. Mixed-response results

Unweighted mean model-level RMSE; lower is better.

| Holdout | P | ROLE_SWAP | MATCHED_OTHER | DISTRACTOR | GLOBAL |
|---|---:|---:|---:|---:|---:|
| context | 7.1332 | 7.1107 | 7.3091 | 7.2832 | 7.0960 |
| candidate_configuration | 7.2655 | 7.2509 | 7.4358 | 7.4200 | 7.2873 |

![P2.4 primary mixed-response controls](../../phase2/p2_4/figures/primary_mixed_controls.svg)

## 18. Raw-score and finite-response results

These are secondary and do not override the mixed-response classification. Meaningful finite sign uses the frozen true-effect magnitude >0.1; endpoint redirection is Family-S only.

| Holdout | Arm | Score RMSE | Finite RMSE | Meaningful sign | Endpoint redirection MAE | Worst kind score RMSE |
|---|---|---:|---:|---:|---:|---:|
| context | P | 3.5494 | 4.6901 | 0.7347 | 2.8395 | 3.7575 |
| context | ROLE_SWAP | 3.5688 | 4.5982 | 0.7490 | 2.8140 | 3.8772 |
| context | MATCHED_OTHER | 3.5755 | 4.6000 | 0.7422 | 2.8325 | 3.7885 |
| context | DISTRACTOR | 3.5019 | 4.6005 | 0.7380 | 2.9019 | 3.7859 |
| context | GLOBAL | 4.0008 | 5.1094 | 0.7192 | 2.8685 | 4.5747 |
| candidate_configuration | P | 3.6217 | 4.7599 | 0.7339 | 3.3562 | 3.9214 |
| candidate_configuration | ROLE_SWAP | 3.6267 | 4.7063 | 0.7400 | 3.4089 | 4.0018 |
| candidate_configuration | MATCHED_OTHER | 3.6365 | 4.7270 | 0.7379 | 3.5137 | 3.9193 |
| candidate_configuration | DISTRACTOR | 3.5809 | 4.7007 | 0.7414 | 3.5087 | 3.8695 |
| candidate_configuration | GLOBAL | 4.0878 | 5.0917 | 0.7141 | 3.4781 | 4.6533 |

## 19. P vs ROLE_SWAP

Role swapping is not reliably worse than P. No role-organization claim is licensed.

| Holdout | Mean P−control mixed RMSE | P better models | Negative condition means | Negative seed means |
|---|---:|---:|---:|---:|
| context | +0.0226 | 4/12 | 1/4 | 1/3 |
| candidate_configuration | +0.0146 | 6/12 | 1/4 | 1/3 |

## 20. P vs MATCHED_OTHER

P has lower mean error, but only 7/12 and 8/12 model signs favor P, below the frozen ≥9/12 consistency criterion.

| Holdout | Mean P−control mixed RMSE | P better models | Negative condition means | Negative seed means |
|---|---:|---:|---:|---:|
| context | -0.1758 | 7/12 | 3/4 | 2/3 |
| candidate_configuration | -0.1703 | 8/12 | 4/4 | 2/3 |

## 21. P vs DISTRACTOR

P has lower mean error, but only 8/12 signs on each holdout favor P; this control tests candidate endpoint states only.

| Holdout | Mean P−control mixed RMSE | P better models | Negative condition means | Negative seed means |
|---|---:|---:|---:|---:|
| context | -0.1499 | 8/12 | 4/4 | 2/3 |
| candidate_configuration | -0.1545 | 8/12 | 4/4 | 2/3 |

## 22. P vs GLOBAL

GLOBAL is better on mean whole-context mixed error and nearly matches P on candidate-configuration; CLS may legitimately encode endpoints.

| Holdout | Mean P−control mixed RMSE | P better models | Negative condition means | Negative seed means |
|---|---:|---:|---:|---:|
| context | +0.0372 | 6/12 | 2/4 | 2/3 |
| candidate_configuration | -0.0218 | 5/12 | 3/4 | 2/3 |

## 23. Optional one-endpoint ablations

Not included. They were explicitly omitted in the frozen preregistration to avoid additional design freedom.

## 24. Per-model consistency

Negative P−control values favor true participants. This complete paired table shows why aggregate mean superiority alone is insufficient. Full precision and secondary deltas are in `tables/paired_participant_vs_controls.csv`.

| Condition | Seed | Context P−role | P−other | P−distractor | P−global | Config P−role | P−other | P−distractor | P−global |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1701 | -0.240 | -0.118 | -0.141 | -0.042 | -0.082 | -0.271 | -0.453 | -0.211 |
| A | 1702 | +0.170 | -0.182 | -0.086 | +0.025 | +0.059 | -0.441 | -0.339 | -0.314 |
| A | 1703 | +0.310 | +0.032 | +0.072 | -0.151 | +0.177 | +0.133 | +0.120 | +0.007 |
| B | 1701 | +0.124 | -0.544 | -0.759 | -0.045 | +0.235 | -0.314 | -0.279 | +0.148 |
| B | 1702 | +0.029 | -0.711 | -0.756 | -0.765 | -0.147 | -0.245 | -0.401 | -0.286 |
| B | 1703 | +0.601 | +0.022 | +0.571 | +0.182 | +0.294 | +0.082 | +0.339 | +0.127 |
| C | 1701 | -0.222 | -1.347 | -1.047 | -0.685 | -0.207 | -0.745 | -0.598 | -0.467 |
| C | 1702 | +0.210 | -0.625 | -0.184 | +0.328 | -0.063 | -0.433 | -0.358 | +0.105 |
| C | 1703 | +0.285 | +1.005 | +1.001 | +1.204 | +0.299 | +0.525 | +0.628 | +0.200 |
| D | 1701 | -0.985 | -0.225 | -0.507 | -0.162 | -0.543 | -0.227 | -0.620 | -0.170 |
| D | 1702 | +0.314 | +0.330 | -0.274 | +0.268 | +0.254 | +0.219 | -0.159 | +0.273 |
| D | 1703 | -0.326 | +0.253 | +0.310 | +0.290 | -0.101 | -0.326 | +0.266 | +0.325 |

## 25. Condition and seed consistency

The P advantage over MATCHED_OTHER and DISTRACTOR is concentrated in source seeds 1701/1702; held-out seed 1703 reverses it on both primary holdouts. Condition means are recorded in `analysis/decision.json`; no favorable subset is selected.

| Control | Holdout | Seed 1701 mean P−control | Seed 1702 | Seed 1703 |
|---|---|---:|---:|---:|
| ROLE_SWAP | context | -0.3304 | +0.1806 | +0.2176 |
| ROLE_SWAP | candidate_configuration | -0.1494 | +0.0258 | +0.1673 |
| MATCHED_OTHER | context | -0.5585 | -0.2970 | +0.3279 |
| MATCHED_OTHER | candidate_configuration | -0.3893 | -0.2253 | +0.1037 |
| DISTRACTOR | context | -0.6132 | -0.3250 | +0.4885 |
| DISTRACTOR | candidate_configuration | -0.4876 | -0.3143 | +0.3384 |
| GLOBAL | context | -0.2335 | -0.0361 | +0.3811 |
| GLOBAL | candidate_configuration | -0.1750 | -0.0554 | +0.1651 |

## 26. Cross-seed descriptive stress test

A/B 1701/1702 to A/B 1703 was refit only on the legal source and evaluated raw/unaligned on the target. It is secondary; coordinate mismatch may explain failures, and no alignment or target labels were used.

| Arm | Context mixed RMSE | Candidate-configuration mixed RMSE |
|---|---:|---:|
| P | 8.5534 | 8.4114 |
| ROLE_SWAP | 7.2038 | 7.6231 |
| MATCHED_OTHER | 7.7973 | 7.6229 |
| DISTRACTOR | 7.4577 | 7.5525 |
| GLOBAL | 7.1458 | 7.5262 |

## 27. Nuisance and failure analysis

The prespecified descriptive error stratification covers condition, seed, query/candidate event, support, evaluation kind, candidate family/permutation, physical and object slots, and baseline-score magnitude. `tables/nuisance_failure_analysis.csv` contains every model×arm×holdout×value count and residual RMSE. The seed-1703 reversal and weak GLOBAL separation are central failures. Descriptive strata cannot establish that no residual leakage or nuisance exists.

| Stratum | P score-error RMSE range across model×value rows |
|---|---:|
| query_event | 2.840–5.552 |
| candidate_event | 2.612–5.206 |
| query_support | 2.898–5.830 |
| candidate_support | 2.750–6.020 |
| evaluation_kind | 2.790–5.306 |
| candidate_family | 2.637–5.333 |
| candidate_permutation | 2.224–6.457 |
| candidate_physical_position | 2.736–5.113 |
| source_object_slot | 2.659–5.204 |
| target_object_slot | 2.623–5.451 |
| baseline_score_magnitude_bin | 1.710–5.930 |

## 28. Frozen classification

**P2.4-D — NONSPECIFIC INTERNAL-STATE SIGNAL.** Neither MATCHED_OTHER nor DISTRACTOR met the preregistered model-level separation rule on both primary holdouts, and GLOBAL did not separate. ROLE_SWAP was not reliably worse than P. The category was assigned mechanically from primary mixed-response metrics only; no score/finite subgroup overrode it.

## 29. What P2.4 establishes

The high-capacity P2.3 residual signal can be approximated by several matched hidden-state descriptions. In this discovery set, the correct participant assignment did not show the preregistered robust specificity pattern across models, especially on held-out seed 1703. This is informative evidence against attributing P2.3's 256 result specifically to correctly corresponding endpoint-bearing state under these controls.

## 30. What P2.4 does not establish

It does not prove endpoints are unused by the task network, rule out all participant-specific descriptions, identify a causal mechanism, locate a neuron/head/layer, establish an operator or grammar, or generalize beyond these discovery models. GLOBAL may contain endpoint information; DISTRACTOR is candidate-side-only; MATCHED_OTHER has some fallback strata. No protected confirmation was performed.

## 31. Updated surviving explanation set

Generic/contextual hidden-state information, redundant endpoint information in CLS or distractor-associated context, model-specific internal coordinates, and richer participant-specific descriptions remain possible. The tested P2.3 selected endpoint-slot description is not sufficiently specific for a stronger claim. P2.1 stays VALID, P2.2 INVALID, P2.3 VALID-D; D6/D9 are nonconfirmatory.

## 32. Most informative unresolved disagreement

Whether the seed-1703 reversal reflects coordinate mismatch/generalization failure of the shared surrogate or genuine nonspecific distribution of relational information across hidden states remains unresolved within P2.4. No larger regression or new experiment was run to resolve it.
