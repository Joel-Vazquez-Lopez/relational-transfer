# P3.1 resumed — final scientific report

**P3.1-C; validity: VALID.** This is a new scientific result after the unchanged P3.1-F feasibility stop and P3.0-R READY acquisition. It does not establish a privileged, causal, or human-interpretable latent ontology.

## Handoff and frozen method

The P3.0-R artifact hashes, twelve discovery-checkpoint identities, 15,360 response-index joins, 1,536 ten-branch families, state/output schemas, intervention flags, and 17/17 readiness gates replayed. Protected seeds 1704–1706 were not opened. P3.0-R and original P3.1-F were not modified. The new preregistration and analysis hash were frozen before source selection; the selected K/alpha were frozen before target evaluation.

The primary paired response is the raw score change for the fixed BASE-correct candidate under nine input/scene interventions. Secondary outcomes are a fixed post-block-1 state-change norm and exact paired crossed-score inclusion–exclusion. S0/S1/S2/S3 contain 96/288/1008/3024 raw pre-intervention dimensions. Source-standardized PCA uses K∈{2,4,8,16,32}; G0 is additive, G1 pairwise, G2 adds up to eight source-selected triple gates. Model selection uses only T0 source-fit/validation. Masks group all branches by base scene.

B0 is source-fitted action-bit additive, a stronger baseline than a constant source mean; B2 is semantic; B3 direct linear and B4 rich-state RBF are per-action; B6 random projection, B7 coordinate permutation, B8 gate permutation, B9 response permutation, and B10 wrong-state are fixed controls. B1 M1 is not algebraically applicable to the C/X response library without unarchived coefficients, so was not refitted. The unseen-action T6 uses a generic 256-feature RBF Fourier compositional comparator in place of per-action B4. B4 is more expressive than G2 and is not called matched. Synthetic prefit tests showed G0 cannot express a latent pair, G1 can, G1 cannot express a triple gate, and G2 can. Direct linear kernel cannot express generic quadratic response, while RBF can. The implemented B2 omitted the target-identity one-hot named in the preregistration; this is a documented comparator implementation discrepancy, not silently corrected after target evaluation. Its superiority to G2 is observed even with this omission, but B2 is not the fully specified semantic comparator.

## Source selection and split sizes

Selected behavioral G1 source-validation settings: S0: K=16, α=100.0, S1: K=32, α=100.0, S2: K=16, α=1.0, S3: K=16, α=100.0. B4 alpha=1.0. Full source-selection trace is archived.

| Fold | Fit | Source validation | Target |
|---|---:|---:|---:|
| T0 | 699 | 100 | 225 |
| T1 | 992 | 0 | 32 |
| T2 | 768 | 0 | 256 |
| T3 | 850 | 0 | 174 |
| T4 | 504 | 0 | 520 |
| T5 | 942 | 0 | 82 |
| T6 | 799 | 0 | 225 |
| T7 | 1024 | 0 | 512 |

T0 uses 799 source bases for target fits (699 initial fit plus 100 source validation); T6 uses the same 799 source bases and withholds IEC action; T7 targets all 512 seed-1703 bases. T1 has only 32 target bases, often 2–4/model, so its per-model estimates are fragile.

## Primary held-out behavioral prediction

Mean per-model RMSE normalized by its own fold/model B0 RMSE; lower is better. Full per-checkpoint metrics are in `per_model_metrics.csv`.

| Fold | B0 | G0 | G1 | G2 | B2_SEMANTIC | B3_DIRECT_LINEAR | B4_GENERIC_RBF | B4_GENERIC_RFF_COMPOSITION | B6_RANDOM_PROJECTION | B7_COORD_PERM | B8_GATE_PERM | B9_RESPONSE_PERM | B10_WRONG_STATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | 1.000 | 0.981 | 0.806 | 0.798 | 0.730 | 0.938 | 0.774 | — | 0.940 | 0.834 | 0.823 | 0.810 | 0.819 |
| T1 | 1.000 | 0.957 | 0.837 | 0.837 | — | — | — | — | — | — | — | — | — |
| T2 | 1.000 | 0.975 | 0.862 | 0.846 | — | — | — | — | — | — | — | — | — |
| T3 | 1.000 | 1.013 | 0.878 | 0.875 | — | — | — | — | — | — | — | — | — |
| T4 | 1.000 | 0.999 | 0.969 | 0.962 | — | — | — | — | — | — | — | — | — |
| T5 | 1.000 | 0.999 | 0.840 | 0.821 | 0.758 | 0.934 | 0.816 | — | 0.980 | 0.831 | 0.877 | 0.829 | 0.817 |
| T6 | 1.000 | 0.930 | 1.164 | 1.164 | — | — | — | 1.105 | 1.281 | 1.211 | 1.164 | 1.168 | 1.195 |
| T7 | 1.000 | 1.009 | 0.990 | 0.988 | 0.754 | 0.954 | 0.984 | — | — | — | — | — | — |

G1 gives a large T0 gain over G0, but G2 adds only 0.008 NRMSE beyond G1. On the predeclared T6 unseen IEC branch, G2 is worse than B0, G0, and the generic Fourier comparator. Seed-1703 cross-seed G2 is near B0. B2 human semantics outperforms G2 on T0, T5, and T7; it uses privileged task labels, so this is an explanatory alternative rather than an equal-information hidden-state comparison.

## Capacity, controls, crossed responses, and stability

G2 S3 has 240 fitted coefficients; B4 independently fits nine RBF heads. Its total smoother effective degrees of freedom are T0 2059.6, T5 2377.0, T7 2555.4. B4 is richer and more costly; neither its small numerical wins nor G2 efficiency alone proves a privileged decomposition. The B7/B9/B10 controls are within 0.05 NRMSE of G2 in important folds, so specificity fails. Control mapping counts are archived.

The paired IE/IC/EC/IEC mixed responses are nonzero and finite in every model; `per_model_direct_nonadditivity.csv` gives their exact means and RMS by fold/model. IE is particularly large in this fixed-position score, while IC/EC and the triple residual tend to be much smaller. Nonadditivity of the downstream score is not evidence of a latent-factor interaction. Secondary internal and mixed prediction tables are in `aggregate_metrics.csv` and `per_model_metrics.csv`; no secondary outcome selects the category.

| Fold | Internal G0 | Internal G1 | Internal G2 | Mixed G0 | Mixed G1 | Mixed G2 |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 0.957 | 0.757 | 0.736 | 0.997 | 0.939 | 0.860 |
| T5 | 1.004 | 0.770 | 0.719 | 1.015 | 1.000 | 0.956 |
| T6 | 0.893 | 1.774 | 1.772 | 1.005 | 1.078 | 1.019 |
| T7 | 0.998 | 0.925 | 0.924 | 1.002 | 1.012 | 1.010 |


Source bootstrap subspace overlaps were 0.838, 0.833; source-validation profile correlations were 0.948, 0.935. This supports a stable predictive subspace under two resamples, not coordinate identity.

The source-nominated two-coordinate predictive ablations, matched random-pair ablations, and 1/42 gate audit are archived. On T0 the first source-nominated gate contains one below-median marginal coordinate and an above-75th-percentile source gate effect, but the T6 test of that gate has exactly zero active IEC cells; it does not validate a general prospective 1/42 effect on the withheld composition. The nominated gates predominantly use R/X intervention bits; IEC has neither, so the specific G2 gates cannot provide a gated IEC contribution. These are surrogate-model ablations, not causal perturbations of the trained task network.

Predictive ablation NRMSE degradation relative to intact G2 (positive means worse):

| Fold | Nominated first | Nominated second | Nominated pair | Random pair |
|---|---:|---:|---:|---:|
| T0 | 0.050 | 0.050 | 0.062 | 0.067 |
| T5 | 0.087 | 0.016 | 0.084 | 0.143 |
| T6 | 0.050 | -0.014 | 0.029 | 0.017 |

The nominated pair has no clear superadditive joint-loss pattern relative to its individual losses; the matched random pair can degrade prediction similarly or more.

Post-fit human-factor projections are archived with no semantic labels used to define PCA. Descriptive D5 acquisition-condition comparison for T0 G2:

| Condition | G2 NRMSE (mean of 1701/1702) | Direct crossed RMS (mean) |
|---|---:|---:|
| A_original_clique_hole | 0.756 | 3.312 |
| B_relocated_clique_hole | 0.713 | 3.370 |
| C_degree_balanced_matching_holes | 0.884 | 3.409 |
| D_path_connectivity_swap | 0.841 | 3.580 |

These differences do not identify a training-caused latent factor.

## Frozen decision and boundary

The mechanical A gates are A_compression=False, A_conditional=False, A_specificity=False, A_generic_predictor=False, A_stability=True, A_crossed_measurement=True. E, D, and B do not fire; C fires because T0 conditional-capable G2 improves same-phi G0 but specificity fails. **P3.1-C** is therefore the frozen category. The data show some low-dimensional predictive interaction value on familiar held-outs, but not a robust privileged conditional organization across controls, the unseen triple action, and cross-seed generalization. Generic nonlinear prediction and human-semantic structure remain live alternatives.

No inference about causal latent factors, true internal ontology, discrete machine-native levels, a transformer grammar, universality, natural language, or inference/training controllability is licensed. No P3.2, protected-seed inspection, new task-network run, or new hidden-state causal intervention occurred.

## Operational and implementation record

A response-indexing error in the generic Fourier comparator was corrected **before** the prefit analysis hash and before any scientific fit. No target-dependent fit or protocol adjustment followed. After held-out evaluation, the semantic B2 input omission described above was discovered; the frozen implementation, predictions, and decision were preserved. The B0 action-bit baseline was the preregistered choice, but is not the literal constant/source-mean comparator suggested in the task. These comparator limitations constrain the interpretation and are not repaired with post-hoc model selection. Baseline model parameters were deterministically reserialized from the unchanged source data for reproducibility; no target response was used in that archival step.

