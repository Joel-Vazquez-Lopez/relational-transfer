# P2.9 — Compositional Boundary Mapping

## Executive summary and validity

**VALID; P2.9-E.** All 23 validity gates passed. The exact P2.8 replay matched 73,728 M1 residual rows, 414 archived model metrics, and 216 control rows with zero numerical difference. No protected seed or new task-network evaluation was used. A required conjunction-label control was documented in a pre-interpretation addendum after deterministic fits; this protocol-timing deviation is disclosed and does not change the frozen models or target partitions.

## Scientific question, entering evidence, and R3 freeze

P2.8 showed reliable single-axis and identity×event benefit but catastrophic identity×event×candidate-configuration failure. P2.9 fills the missing pairwise lattice cells to distinguish a pairwise failure, a higher-order boundary, and ordinary extrapolation. R3 remains the exact eight-coefficient P2.7/P2.8 semantic residual ridge predictor with frozen features, source-only RMS scaling, alpha grid, validation rule, and OOF/held-out M1 residual target. No new ontology or model search was introduced.

## Frozen lattice, directions, support, and algebra

The inherited D0/I/E01/E10/C/IE/IEC cells were replayed, not redefined. New IC fits identity-group0 rows and tests group1 on held-out candidate configurations; new EC fits event0 and tests event1 on held-out configurations. IE_nearest restores configuration overlap from the triple source/target definition. All primary pairwise and triple cells require observed query and observed candidate support. Event directions remain separate. IC_reverse is descriptive only on B/C/D because condition A has 14 target rows, below the frozen 20-row gate; EC_reverse is also descriptive. Every within-model source/target score row is disjoint, and no score-derived feature or finite-difference reconstruction enters R3. Exact formulas, counts, support classes, semantic overlap, and 144 SAFE algebra rows are in the prefit files.

## Primary pairwise and triple results

NRMSE is residual RMSE divided by M1-alone RMSE on the same target cells (lower is better). Values below are unweighted means of model-level values, never score-row pseudo-replication.

| Cell | Order | Models | Target cells | R1 mean NRMSE | R3 mean NRMSE | R3<1 models | R3<R1 models |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0 | 12 | 4912 | 0.7138 | 0.5938 | 12/12 | 12/12 |
| I | 1 | 12 | 7740 | 0.7147 | 0.5679 | 12/12 | 10/12 |
| E01 | 1 | 12 | 12786 | 1.7068 | 0.7024 | 12/12 | 12/12 |
| E10 | 1 | 12 | 12102 | 1.8857 | 0.6808 | 12/12 | 12/12 |
| C | 1 | 12 | 6912 | 0.7202 | 0.5848 | 12/12 | 12/12 |
| IE | 2 | 12 | 6534 | 0.6284 | 0.5111 | 12/12 | 12/12 |
| IC | 2 | 12 | 2262 | 0.7154 | 0.5569 | 12/12 | 11/12 |
| EC | 2 | 12 | 1743 | 2.3549 | 0.4899 | 12/12 | 12/12 |
| IE_nearest | 2 | 12 | 3813 | 14.0223 | 15.2682 | 0/12 | 0/12 |
| IEC | 3 | 12 | 1098 | 13.5871 | 14.7947 | 0/12 | 0/12 |

The nearest I×E relaxation is the decisive counterpoint to a simple order boundary: although the broad inherited IE cell benefits in 12/12 models, IE_nearest fails in 12/12 models (mean NRMSE 15.2682), almost as badly as IEC. Thus “all pairwise cells work” holds only for the broad cells, not all distance-near constructions.

Full raw RMSE, MAE, model-level ranges, condition/seed breakdowns, and directional variants are in `per_model_metrics.csv`, `model_level_summary.csv`, `pairwise_metrics.csv`, and `triple_metrics.csv`.

## Leave-one-dimension-out and pairwise-versus-triple comparison

IC restores event overlap, EC restores identity overlap, and IE_nearest restores configuration overlap relative to the frozen triple. The twelve model-level pair and triple NRMSEs for each relaxation are in `leave_one_dimension_out.csv`. Full-cell results alone do not identify conjunction order because source size and semantic distance differ.

## Semantic distance, matched distance, target subsets, and source size

The source-overlap distance is the frozen maximum match count among ten task-native categorical factors; target matching fixes condition/seed, observed support, source-SD-based residual scale bin, and distance within one factor. No R3 error was used to choose a match. The strict prefit matcher retained only 15 IC and 13 EC rows in its sparsest model; the strongest all-model distance-matched higher-order criterion therefore cannot pass.

| Pair | Matched rows/model, minimum | Matched pair NRMSE | Matched triple NRMSE | Positive gap models |
|---|---:|---:|---:|---:|
| IC | 15 | 0.6643 | 15.8880 | 12/12 |
| EC | 13 | 0.6499 | 20.7632 | 12/12 |
| IE_nearest | 30 | 14.9586 | 14.7434 | 6/12 |

IC and EC were also evaluated on the exact frozen IEC target score rows, eliminating target-size and target-scale differences for that comparison: mean NRMSE is 0.5473 for IC and 0.4967 for EC, each beneficial in 12/12 models, versus 14.7947 for IEC. See `shared_triple_target_metrics.csv`. Pairwise source-size controls use the prefit hash-selected rows: each training model contributes exactly its triple source count, for 1,530 pooled rows over the eight training models. These do not match the triple source residual distribution or semantic composition.

| Pair | Source-size-matched R3 mean NRMSE | Beneficial models |
|---|---:|---:|
| IE | 0.5138 | 12/12 |
| IC | 0.5752 | 12/12 |
| EC | 0.5065 | 12/12 |

The three frozen target-residual scale strata and their exact counts/results are in `target_scale_controls.csv`. On unweighted model means, pairwise IC/EC benefit in scale bins 1–2 but not bin 0; IEC fails in all three. Thus target-scale control supports a difference in two of three bins, not a scale-invariant pairwise benefit.

## Distance-only baseline, order adjustment, and non-additivity

A four-coefficient linear distance baseline was fitted only on archived P2.8 D0/D1 model-level results, using source overlap, log source rows, and source residual SD. It predicts P2.9 cell difficulty without fitting to those cells. A separate leave-one-model-out descriptive comparison adds one order-3 indicator to the same descriptors.

Distance-only OOF RMSE: 2.8155; distance-plus-order OOF RMSE: 0.6839. Detailed predictions are in `distance_only_baseline.csv` and `distance_order_oof.csv`. The frozen additive null is IE_nearest+IC+EC−I−E01−C+D0 per model; triple excess >0.2 occurs in 1/12 models. Exact values are in `nonadditivity_metrics.csv`. These are descriptive model-level comparisons, not a neural composition law.

## Catastrophic failure diagnostic

The inherited IEC model was inspected without repair. Across the twelve model-level ALL rows, the mean median prediction is −60.846 while the mean median true residual is −2.369, and the mean median absolute error is 58.174. This is a broad extrapolated offset rather than merely a few extreme rows. Prediction, true-residual, and absolute-error quantiles; maximum error; sign agreement; and S/R family/condition breakdowns appear in `catastrophic_failure_diagnostics.csv`. Nothing was clipped or recalibrated.

## Negative controls and specificity

For new pairwise cells, the fixed cyclic endpoint-identity feature permutation and strict source-residual permutation reuse P2.8 procedures. Inherited IE/IEC controls remain unchanged. The conjunction-label control uses exact 2^12 model-level sign flips on the prefit matched pairs; its definition was frozen in the disclosed pre-interpretation addendum before looking at P2.9 performance.

| Pair | Observed matched gap NRMSE | Exact sign-flip p |
|---|---:|---:|
| IC | +15.2237 | 0.0002 |
| EC | +20.1132 | 0.0002 |
| IE_nearest | -0.2152 | 0.7493 |

Mean control NRMSEs (broken identity / permuted source residual) are IE 0.4949 / 0.5296 versus R3 0.5111; IC 0.5559 / 0.5436 versus R3 0.5569; EC 0.4912 / 0.5259 versus R3 0.4899. Both controls lie within the frozen 0.02 similarity margin for IE and IC, satisfying the P2.9-E rule. Exact model-level controls are in `negative_controls.csv`; numerical pairwise benefit is not evidence of correspondence-specific reusable computation.

## Cross-seed and cross-graph descriptive stress tests

A/B seeds 1701–1702 supplied frozen source fits evaluated without adaptation on A/B seed1703 for IE/IC/EC/IEC. Mean cross-seed R3 NRMSE over the two held-out models is IE 0.3878, IC 0.3975, EC 0.4365, and IEC 10.2852. IE and IEC cross-seed rows were inherited from P2.8. Condition-level A/B/C/D profiles are in `crossgraph_metrics.csv`; these do not isolate a D5 causal mechanism.

## Frozen category, licensed conclusion, and surviving explanations

The mechanical category is **P2.9-E**. The exact gating evidence is in `decision.json`: pairwise benefit, triple failure, source-size and scale controls, strict matched-distance adequacy, non-additivity, distance-only account, and specificity controls are all reported separately. The strongest licensed conclusion is a bounded predictive generalization profile of the unchanged R3 residual description. P2.9 does not establish grammar, symbolic composition, neural modules, transformation operators, causal states, predictive states, or a universal transformer law. D5 remains the acquisition-side causal anchor; any cross-condition boundary difference here is descriptive. Surviving alternatives include specific pair interactions, candidate-configuration shift, and ordinary semantic-distance extrapolation wherever the matched controls cannot eliminate them. The most informative unresolved disagreement is whether the catastrophic triple behavior survives a genuinely adequately populated distance-matched pairwise comparison; the frozen strict matcher is sparse in IC/EC and was not loosened after results.

## Reproducibility and stop

P2.8 manifest SHA-256: `feed7a303286ca70b477c41ccb69404d9346e41ff1bf2ebea394be554abb82ef`. P2.9 lattice preregistration SHA-256: `b79d52c22b6aaf5c0153695859d7bfeb673881fc56b7990f52c6be9211ba3f19`. P2.9 fit implementation SHA-256: `bc8dabd7fe6c9bcd9701784f0793d3cd0eb968b1a845187a2ad41fafd8128f52`. All tables, fitted residual-model coefficients, figures, audits, and code are hash-listed in the reproducibility manifest. No P2.10 was designed or run.
