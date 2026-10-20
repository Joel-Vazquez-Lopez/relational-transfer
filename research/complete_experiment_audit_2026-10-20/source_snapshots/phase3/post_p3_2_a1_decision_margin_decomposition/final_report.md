# A1 — P3.0-R decision-margin decomposition (post hoc archival analysis)

**Status:** authorized read-only exploratory analysis. P3.2-E remains the experimental endpoint and its frozen category is unchanged. No model forward pass, initialization, fitting, retraining, new intervention, protected-seed access, or prior-artifact modification occurred.

## ARCHIVED FACT

The valid P3.0-R archive covers 12 permitted condition×seed checkpoints and 1,536 base states. The read-only join used 15,360 archived branch rows; all required source artifact hashes matched. Saved post-final-normalization query and candidate vectors reconstructed every archived score with maximum absolute error 0. The three-term identity held for 25,344 valid non-BASE correct-versus-foil margins with maximum error 2.93e-06 (tolerance 1e-05). No matching candidate exists for 5,376 non-BASE branches; the report includes all other three foils without outcome-based selection.

The score vectors used in the identity are individually normalized CLS outputs. The stage arrays are pre-final-normalization traces and were not substituted for the final scorer inputs. Candidate slot correspondence follows the unchanged outer candidate order; C only permutes objects inside each candidate and X only changes distractor identity.

## NEW POST-HOC ARCHIVAL OBSERVATION

See `action_validity_counts.csv`, `model_action_support_summary.csv`, `condition_action_support_summary.csv`, `paired_AB_seed_summary.csv`, and the complete `margin_decomposition.csv`. Model/checkpoint summaries, not margin rows, are the reproducibility units; A/B seed pairing is at the summary level only, never scene-paired. C/D comparisons are descriptive. `support` and `kind` are the existing archive strata; no outcome-defined grouping was used.

The primary signed components are Q (query-side), C (candidate-side), and X (cross-term). Their complete within-model distributions, 5th/50th/95th percentiles, and equal-seed condition summaries are preserved in the tables. Cancellation is retained as `1−|Δm|/(|Q|+|C|+|X|)` and as signed shares; a high-cancellation flag uses the explicitly descriptive cutoff 0.9. Encoder-only zero terms are algebraic sanity checks, not discoveries. No margin-level p-values are reported.

## INTERPRETATION

The decomposition is an exact, baseline-dependent expansion of the archived dot-product margin change. Q, C, and X are accounting terms, not independently manipulated causes or identified internal mechanisms. A large term can cancel another; the absolute/signed summaries preserve that structure. Query-only and candidate-only intervention zeros follow mechanically from unchanged saved vectors. The number of margins must not be treated as an independent sample size; the archive has only three permitted seeds per condition.

This analysis sharpens the compatibility-function framing by making the decision margin and its query/candidate/cross changes inspectable on archived branches. It does not establish a compact law across interventions or support structures, and it does not test unsupported P3.2 Y/V branches because their post-action hidden vectors were not archived. D4/D5/D7-R/D8A.2/P3.1/P3.2 validity classifications are unchanged.

## SPECULATION

Any proposal that stable Q/C/X patterns constitute a general execution-level law, that a particular term mediates support effects, or that this decomposition predicts unsupported P3.2 behavior remains speculative. The archived algebra alone cannot distinguish learned causal organization from ordinary distributed representations evaluated by the fixed normalized dot-product scorer.

## Summary answers

1. **Establishes:** exact arithmetic decomposition of all eligible archived P3.0-R margins, with verified score reconstruction and full record-level provenance.
2. **Rules out or weakens:** any claim that the analyzed score changes cannot be represented by the specified three terms, or that all valid branches use the same query-only/candidate-only pathway. It does not rule out mechanisms.
3. **Equally compatible:** stable conditional organization, intervention-specific computation, ordinary distributed nonlinear encoding plus the fixed dot-product scorer, magnitude/gain explanations, and heterogeneous context dependence.
4. **Framework:** it modestly strengthens the utility of `D → Cθ → Fθ → B` as an organizational map for this archived margin question; it does not establish the arrows as a causal mediation chain or show that Q/C/X are privileged components of Cθ.
5. **Sharper question:** do model-level Q/C/X distributions predict held-out candidate margins across pre-existing contexts and interventions after accounting for support, baseline margin, and response scale—or do their patterns remain intervention- and context-specific? No next experiment is proposed here.

## Integrity and provenance

P3.0-R manifest SHA-256: `5d437b68be8dab0e344f6414aeb6c3c0d2bd0f7fb04d21c2201d6910a3aa440a`. A1 analysis-code SHA-256: `8b940aa295e0767f5747cd18fcccfc9e53b32a7d59838fd6b802c908bb2f5e80`. See `integrity_audit.json` and `analysis_manifest.json` for source and output hashes. P3.2 preregistration, implementation, predictions, acquisition, report, and decision files were not changed.
