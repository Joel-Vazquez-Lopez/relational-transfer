# P2.6 — Query–Candidate Computational Asymmetry Test

## 1. Executive summary

**VALID; P2.6-D.** P2.5 replayed exactly. Under the frozen matched response library, the preregistered positive query-over-candidate asymmetry criterion did not hold. The mean paired difference A=G_Q−G_C was context: -0.1680, candidate_configuration: -0.1640. Negative A means greater candidate-side functional gain over the identical-information semantic comparator. This does not establish a candidate-specific mechanism.

## 2. Validity status

All 20 final validity gates: PASS. The protected seeds were not read, hashed, or loaded. No task network was trained or modified. The P2.1 combination split was not a co-primary P2.6 holdout; the 14-score-cell exclusion would have emptied some model/family strata. Both prescribed structural holdouts and identical legal Q/C rows were retained.

## 3–6. Motivation, replay, hypotheses, and non-assumptions

D5 is the acquisition-side causal anchor. P2.5 (VALID-D) showed F>S4 for 4/4 query and 0/4 candidate family×holdout cells, without establishing a general predictive state. P2.6 tests whether that side split survives exact semantic counterpart, scale, coefficient, score-call, and holdout matching. H_SYM, H_ASYM, and secondary H_COMP were frozen. No module, semantic register, relational-demand query, scene-representation candidate, or interface operator is assumed. The P2.5 replay matched 1,008 archived metric values with maximum absolute delta 0.

## 7–17. Provenance and frozen measurement

P2.5 manifest SHA-256: `009430d0dda17e62172e74c37a29fc09ec767ca605be4aca683ca2f2ce04c99b`. P2.6 matched-library SHA-256: `1cf53adf9cd1017eef8d03e13747f304cc3c810b44722f9513b10a050e98824b`. Algebra-audit SHA-256: `23576f5b646ba995f0b7903fc3897aeeaec902e29a3f37a0a998a1e83f30371e`. Fit implementation SHA-256: `62dc7f247a5fd140b86114343bb0cbd93189e9313395e556595ecd8be9155b9f`. Twelve archived discovery atlases were used: A/B/C/D × seeds 1701–1703. Eight source models (1701/1702) supplied fits; all 12 were evaluated on whole-context and candidate-configuration holdouts.

Each S-family pair represents source, target, or event substitution. Each R-family pair represents role, event, or role+event reversal. Q and C use the same semantic member index. Exact formulas, target scales, variances, cell counts, and observation costs are in `matched_response_library.csv` and the prefit JSON. Every target has two archived score terms disjoint from its four observed score terms; all 12 primary folds and the joint target are SAFE. F observes two response coordinates/four score calls and fits three coefficients. S uses the same 50-coordinate anchor semantic access on both sides with the same fixed 2D projection. H uses the frozen block-1 side-specific hidden state with matched fixed 2D projection. Train-only ridge selection and train-only mean-baseline normalization are identical. Six-pair difficulty matching uses source-only baseline RMSE ratio [0.75, 1.3333]. The permutation map preserves model, family, kind, supports, side, and structural split. Algebraic disjointness requires Q and C finite differences to be evaluated against different fixed counterpart members; semantic intervention and budget are paired, but the local score contexts are not literally identical. This bounds the interpretation of any side difference.

## 18–25. Primary matched results

| Holdout | Mean G_Q | Mean G_C | Mean A | Models A>0 / 12 | Difficulty-matched mean A | Matched pairs |
|---|---:|---:|---:|---:|---:|---:|
| context | +0.0035 | +0.1716 | -0.1680 | 1/12 | -0.1731 | 3/6 |
| candidate_configuration | +0.0296 | +0.1936 | -0.1640 | 0/12 | -0.0989 | 3/6 |

Paired family means, each across 12 models (all individual counts and errors are in the CSV):

| Holdout | Family/type | G_Q | G_C | A | Models A>0 |
|---|---|---:|---:|---:|---:|
| context | S1: source substitution | -0.0692 | +0.0531 | -0.1222 | 2/12 |
| context | S2: target substitution | -0.0234 | +0.0490 | -0.0724 | 5/12 |
| context | S3: event substitution | +0.0149 | -0.0573 | +0.0721 | 10/12 |
| context | R1: role reversal | +0.0083 | +0.3025 | -0.2942 | 2/12 |
| context | R2: event substitution | +0.0285 | +0.3230 | -0.2945 | 1/12 |
| context | R3: role plus event reversal | +0.0620 | +0.3591 | -0.2971 | 2/12 |
| candidate_configuration | S1: source substitution | +0.0341 | +0.2448 | -0.2107 | 1/12 |
| candidate_configuration | S2: target substitution | -0.0367 | +0.2644 | -0.3011 | 0/12 |
| candidate_configuration | S3: event substitution | +0.0634 | -0.0740 | +0.1374 | 11/12 |
| candidate_configuration | R1: role reversal | +0.0144 | +0.1899 | -0.1754 | 1/12 |
| candidate_configuration | R2: event substitution | +0.0548 | +0.2634 | -0.2086 | 1/12 |
| candidate_configuration | R3: role plus event reversal | +0.0476 | +0.2730 | -0.2253 | 3/12 |

Full per-model, per-family G_Q/G_C/A, raw RMSE, normalized RMSE, paired legal cell counts, source scale, condition, and seed are in `paired_asymmetry_metrics.csv`. The main contrast is negative in most families, except S3 event substitution. Across models, positive pooled A occurs in context: 1/12, candidate_configuration: 0/12. Thus neither an aggregate mean nor score-cell pseudo-replication supplies evidence for the preregistered positive query-side asymmetry.

## 26. Permutation control

| Holdout | Side | F normalized RMSE | Identity-permuted F | Mean baseline |
|---|---|---:|---:|---:|
| context | Q | 0.8395 | 0.8402 | 0.8825 |
| context | C | 0.6676 | 0.6827 | 0.8762 |
| candidate_configuration | Q | 0.8721 | 0.8856 | 0.9332 |
| candidate_configuration | C | 0.7169 | 0.7672 | 0.9614 |

## 27. Cross-side prediction

Cross-side ridge uses the opposite side’s two semantically corresponding responses and four disjoint archived score cells. Normalized RMSE by target side: context Q 0.7880, C 0.7787, candidate_configuration Q 0.8411, C 0.8809. These are predictive associations, not causal direction.

## 28–29. Joint complementarity and cross-fitted unique information

The held-out joint target is `s(q1,c3) − s(q0,c0)`, using two score terms absent from all six Q/C response coordinates. All four joint comparators fit seven coefficients. Q_ONLY/C_ONLY include all three own-side linear terms and three pairwise products; Q_PLUS_C uses six linear terms; SEMANTIC_MATCHED uses a fixed six-dimensional semantic projection.

| Holdout | Q_ONLY | C_ONLY | Q_PLUS_C | SEMANTIC_MATCHED |
|---|---:|---:|---:|---:|
| context | 0.7413 | 0.6256 | 0.6485 | 0.7240 |
| candidate_configuration | 0.8094 | 0.6836 | 0.6851 | 0.8302 |

Q_PLUS_C did not beat all same-budget controls on both holdouts. The OOF residual test used five source-only folds; full per-family/model residual gains appear in `unique_information_metrics.csv`. context: C→Q residual +0.0232, Q→C residual -0.0528, candidate_configuration: C→Q residual +0.0924, Q→C residual +0.0212. No complementary interface claim is licensed.

## 30. Cross-seed and cross-graph descriptive reuse

A/B 1701/1702 fits were evaluated without adaptation on A/B 1703, then C/D discovery models. Exact side, arm, family, holdout, condition, and seed results are in `crossseed_metrics.csv`. Transfer is secondary and cannot override the primary matched result.

## 31–36. Category, claims, surviving explanations, reproducibility

The frozen decision rule assigns **P2.6-D**. Matched query-side functional gain was not systematically greater; in this library the candidate-side relative gain was usually larger. This is consistent with the P2.5 side split being sensitive to unmatched measurement construction, but does not prove equal pathway computations. The surviving explanations include family-specific response geometry, side-dependent difficulty, and genuine computational differences in untested responses. The most informative unresolved disagreement is why S3 event substitution favors query-side gain while the other five paired types favor candidate-side gain. No mechanistic register, module, interface operator, causal direction, or general predictive state is established.

All fitted models, raw tables, frozen prefit documents, replay, validity audit, decision JSON, and reproducibility hashes are under `phase2/p2_6/` and `outputs/phase2/`. No further experiment was run.
