# P2.7 — Residual Factorization Test

## Executive summary and validity

**VALID; P2.7-D.** All 23 final validity gates passed. P2.1 M1 replayed exactly: 48 archived model×holdout rows, 672 numeric checks, maximum absolute delta 0. No protected seed was opened or hashed, no task network was trained, and no new network forward pass was made. A serialization-only CSV header repair occurred before any predictive outcome was inspected; the scientific design was unchanged.

## Scientific motivation and entering evidence

D5 remains the causal acquisition-side anchor. D8 and P2.1–P2.6 did not establish a privileged endpoint vector, a query/candidate interface, or a generic compact functional state. P2.7 asks whether the residual of the frozen 51-coefficient M1 shared-feature model separates into task-defined demand components. P2.2 is invalid and supplies no evidence.

## M1 replay, residual construction, and provenance

M1 uses its original 50 semantic/equality/support inputs plus intercept, P2.1 core partition, validation hash, alpha grid, and train-only normalization. Core-row residuals use five grouped 4×4-block OOF M1 fits; every noncore row uses the original M1 fit trained solely on core. The residual atlas contains 73,728 rows with exact score IDs, semantic meanings, support, split flags, M1 prediction, target, residual, and provenance. No in-sample M1 residual enters residual fitting.

## Predeclared demand families, overlap, and unseen combination

Endpoint demand means ordered query endpoints differ from candidate endpoints. Event demand means query and candidate event IDs differ. Fitting uses endpoint-only or event-only rows; the joint endpoint-plus-event demand is never fitted. These are symbolic task-native labels, not residual clusters or neural mechanisms. Endpoint-only and event-only fitting families are disjoint; their joint occurrence is the natural unseen combination. Per-condition/seed counts, target scales, and exclusions are in `residual_family_prereg.json`.

## Algebra/leakage audit

SAFE for unseen-combination and family-holdout tests. Predictors use no archived score or functional-response observation. Training and target score cells are disjoint by demand mask; there is no exact score reconstruction or finite-difference identity. The audit is in `residual_algebra_audit.json`.

## R0–R3 and exact complexity

R0 predicts zero residual (M1 alone). R1 fits one common eight-coefficient correction from a constant, demand count, and six projected semantic features. R2 fits four event-gated plus four endpoint-gated terms, including gated constants. R3 fits an undifferentiated eight-coefficient control from separate demand indicators and the same six semantic projections. All use the same source-only ridge alpha grid and source-only feature scaling. No hidden or functional score predictor is used. Parameter, projection, and family-indicator costs are in `complexity_accounting.csv`.

## Structural holdouts and primary unseen-combination results

Whole-context and candidate-configuration holdouts were frozen. Residual RMSE is normalized to R0 on the same target split (R0=1); lower is better. Each mean below is across 12 model-level values, not 73,728 independent replications.

| Split | R0 | R1 | R2 | R3 | Models R2<R1 | Models R2<R3 |
|---|---:|---:|---:|---:|---:|---:|
| unseen_combination | 1.0000 | 0.9131 | 0.9609 | 0.6680 | 4/12 | 0/12 |
| context_combined | 1.0000 | 0.9720 | 0.8841 | 0.5948 | 9/12 | 0/12 |
| candidate_configuration_combined | 1.0000 | 0.9168 | 0.9192 | 0.6442 | 6/12 | 0/12 |

The eight-coefficient R3 control outperformed R2 on every co-primary unseen-combination split. This is decisive against the preregistered factorization criterion, even where R2 improves over R0 or R1. Raw residual RMSE/MAE and exact per-model counts are in `unseen_combination_metrics.csv` and `paired_model_deltas.csv`.

## Family holdout and per-family results

Two directional single-family holdouts were run without fitting the absent component; exact endpoint-only and event-only results across fit-eligible/context/configuration splits are in `family_holdout_metrics.csv`. These tests cannot rescue failure on the unseen combination. Family support and overlap are recorded in the prefit document.

## Per-model consistency

All 12 condition×seed primary comparisons appear in `paired_model_deltas.csv`; condition and seed remain visible, and the decision uses model-level direction rather than score-cell pseudo-N.

## Unique-component OOF analysis

The remaining component was fitted on grouped out-of-fold residuals from a model omitting the tested component. Incremental RMSE, MAE, and residual sign accuracy by model/split/component are in `unique_component_metrics.csv`. Component-specific improvement alone is not factorization evidence when R2 loses to R3.

## Component-swap and label-permutation controls

| Split | R2 | Swapped components | Permuted demand labels |
|---|---:|---:|---:|
| unseen_combination | 0.9609 | 0.9910 | 0.7804 |
| context_combined | 0.8841 | 0.9050 | 0.7210 |
| candidate_configuration_combined | 0.9192 | 0.9439 | 0.7704 |

The label-permutation control is better than correctly assigned R2 on the tested splits. This further weakens a meaningful endpoint/event component interpretation. Permutations preserve declared strata and use no residual values.

## Cross-seed and cross-graph reuse

A/B 1701/1702 fits were evaluated without adaptation on A/B 1703; the same frozen features were evaluated on C/D. Full R0/R1/R2/R3 rows are in `crossseed_metrics.csv` and `crossgraph_metrics.csv`. Transfer cannot override the failed primary R2-versus-R3 comparison.

## Frozen classification and claim boundary

**P2.7-D** under the preregistered rules: the explicit two-component residual correction does not provide a consistent advantage over the undifferentiated same-budget control. This does not prove there is a single neural mechanism, nor that M1 has no residual structure. It says the tested task-defined two-way factorization is not supported at eight fitted coefficients. No neural modules, symbolic variables, causal states, operator algebra, transformer grammar, or language conclusion follows.

## Relationship to prior studies and surviving explanations

D5 topology effects remain intact. P2.1 established a strong compact interpolation baseline but left residual. P2.3/P2.4 did not isolate endpoint vectors; P2.5/P2.6 did not establish a general functional state or stable query/candidate asymmetry. Here, R3 indicates that some compact semantic residual correction exists, but explicit endpoint/event decomposition is not required by the tested data. Surviving possibilities include one shared correction, interactions among task-native variables, model-specific residual variation, and untested demand definitions; none licenses post-hoc rescue. The informative disagreement is why gated components improve some broad structural splits while the undifferentiated control is substantially better on the held-out joint demand.

## Reproducibility summary

Frozen M1 replay SHA-256: `5d837fcc9c21a20c08fc93c792541b33182e6bcc9949f6162c66e95eca57db9f`. Residual-family preregistration SHA-256: `d25d82633a2ff5d50f126e57e473a0894197e479577c21c322f2a214cf0e6c90`. Residual atlas SHA-256: `af9ca6900a502368ef81b5aac304805753062b07f8234841051e1b89b960a583`. Repaired fit implementation SHA-256: `072687bcb117d12abdfa4f2a7f2153cab8e7df33fce858d9b9026c35e4870c77`. The operational repair and old hash are recorded in `operational_repair.md`. All raw tables, fitted ridge coefficients, controls, audit, decision, and figure are under `phase2/p2_7/`. No further experiment was run.
