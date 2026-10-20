# P2.2 — Matched-Budget Residual Rule Test

**Completed 20 September 2026 · PRIMARY H0/H1 COMPARISON INVALID.** All preregistered fits and controls ran once and are preserved. A final expressivity audit found a critical design asymmetry: the internal Q+C residual arm was mathematically incapable of predicting the mixed query–candidate interaction that the semantic arm was permitted to predict. No post-outcome repair, feature addition, threshold change, refit, or rescue run was made. The mechanically generated `P2.2-A` category remains in the frozen [decision artifact](../../phase2/p2_2/analysis/decision.json) for provenance, but **it is not a licensed scientific interpretation**. The authoritative validity finding is [expressivity_audit.json](../../phase2/p2_2/analysis/expressivity_audit.json).

## 1. Executive summary

P2.2 asked whether a small participant-state model explains the errors left by P2.1's compact 51-parameter M1 semantic interpolation baseline better than an equally budgeted semantic extension. The nominally matched arms had exactly 32, 64, 128, or 256 *incremental fitted coefficients* and used one shared pooled fit, not per-seed residual fits. Descriptively, the semantic extension improved as budget increased: on context holdout its equal-model-weight residual score RMSE went from a pooled-M1 baseline **3.490** to **2.406** at budget 256; on candidate-configuration holdout **3.532** to **2.642**. The joint internal-state arm scored **3.781** and **3.836** respectively at budget 256. These numbers cannot establish that semantic interpolation suffices, because the arms were not matched in the ability to represent the target interaction.

The exact structural problem is that the internal arm formed a fixed **linear** projection of concatenated query-only and candidate-only states and used a **linear** ridge readout. Its correction is therefore `a(query) + b(candidate) + constant`. For any legal query and candidate substitutions, the four-cell mixed finite difference of that correction is **zero**. The semantic bank explicitly included query×candidate conjunctions. Across 96 primary I-QC model×budget×complete-holdout metric rows, the largest change in mixed-response RMSE was only **2.44×10⁻⁶** (floating-point scale). The semantic arm changed mixed RMSE materially. This mismatch invalidates the declared primary test, not merely a secondary metric.

## 2. Validity status and frozen boundary

The P2.2 machine-readable [preregistration](../../configs/p2_2_residual_rule_prereg.json) SHA-256 is `298c89943e821657928115f5c208ab2a4c283a24f0b2fd8e284104cf3fd98529`; the [human preregistration](../../phase2/preregistrations/p2_2_residual_rule.md) SHA-256 is `a45bdf22057f13df35c7b9173ce8b272bfc1a9f23d3855dbf4aea44543c64423`. A prefit clarification of the shared pooling rule and OOF grouping is documented with both old and new hashes in [prefit_amendment.md](../../phase2/p2_2/analysis/prefit_amendment.md). It occurred before any P2.2 comparative fit or result.

The [prefit audit](../../phase2/p2_2/analysis/prefit_audit.json) passed: all 12 P2.1 atlases, checkpoint identities, normalization sidecars, replay status, source hashes, structural split flags, and candidate semantic-member/physical-position mappings matched their preserved records. The P2.1 M1 per-model context RMSE reproduced with **maximum difference 0**. No activations were re-extracted. P2.2 loaded no model for a forward pass, trained no task network, and did not access protected seeds 1704–1706. D6 and D9 remain invalid and supplied no confirmatory evidence.

The [mechanical decision](../../phase2/p2_2/analysis/decision.json) marked its listed operational gates true and assigned P2.2-A from the frozen *score-only* category condition. The post-run algebraic audit exposed an **additional critical scientific validity failure** implicit in the user-specified matched-capability requirement and central mixed-response target. A coefficient-count match is insufficient when one arm cannot express the central response. The H0/H1 claim, P2.2-A interpretation, and any P2.2-B rejection are therefore **invalid**. This audit did not change the preregistration or existing outcomes.

## 3. Exact frozen hypothesis comparison

H0 was conditional/shared-feature interpolation: richer bounded semantic/context conjunctions should explain M1's residual at least as efficiently as internal participant state. H1 was a compact reusable state-dependent residual signal beyond that declared semantic extension. Even a valid H1 result would not have established a causal mechanism, operator law, acquisition principle, predictive state, grammar, or language transfer. P2.2 used existing D5 A/B/C/D discovery models at seeds 1701–1703, already observed in P2.1; this was a discovery analysis, not protected prospective confirmation.

## 4. Residual construction and out-of-fit discipline

The exact P2.1 50-feature M1 ridge, alpha grid `[0.01, 1, 100]`, train-only centering/scaling, core validation, and full-core refit were reused. Source fitting pooled A/B/C/D seed-1701/1702 core rows once per arm and budget. Five OOF folds omitted seed from the semantic cell key so matching source-seed cells shared a fold. Each source-core residual target was the true score minus an M1 prediction from a fit **excluding its fold**; the [OOF audit](../../phase2/p2_2/analysis/baseline_oof_audit.json) records counts and selected alphas. Holdout predictions used the full source-core M1 fit, then added the pooled residual prediction. Target-seed labels, scales, and calibrations were never fitted. No random row-level test split replaced the frozen P2.1 context, candidate-configuration, combination, or event-disagreement holdouts.

Finite-response and mixed-response residuals were derived from corrected score matrices on complete context and candidate-configuration blocks. Partial-cell combination and substitution-family holdouts were evaluated for score residual only. Sign accuracy used P2.1's meaningful-effect cutoff |true finite effect|>0.1.

## 5. Complexity matching—and why it was insufficient

The S bank used P2.1 semantic features plus bounded identity-role, identity-event, endpoint-pair, and context-triple conjunctions. The I bank used P2.1 STATE-1 query CLS/source/target tokens and candidate CLS/source/target object states, selected using the declared semantic metadata. Neither arm got a fitted PCA basis: fixed Gaussian projections had **zero fitted coefficients**, with seeds 2102 (primary) and 2103 (projection control). A ridge intercept plus projected coordinates gave exact incremental fitted counts **32/64/128/256**, for equal total M1-plus-extension counts **83/115/179/307**. Q-only and C-only were assessed through budget 128 because their raw pathway dimensionality is 144. Projection matrix sizes, input dimensions, alphas, coefficient counts, and weight sizes appear on every row of the [per-model table](../../phase2/p2_2/tables/per_model_metrics.csv).

This matches scalar coefficient budget but **not representational capacity**. For S, the six 6×6 cross-side identity conjunction groups can change a mixed interaction. For I-QC, `P[q;c]` followed by `wᵀ` is separable into query and candidate terms. The privileged semantic metadata used to pick endpoint-bearing states was disclosed, but it does not create a cross-side interaction in the linear readout. The combined S+I arm can change mixed interactions only through its S half; it is not evidence that I adds a mixed rule. The permuted-state control independently permuted rows, breaking natural query/candidate factorization and making it an unsuitable *interaction-specificity* control. These limitations were discovered after the outcomes and were **not repaired**.

## 6. Structural holdouts

P2.1's context holdout withheld whole relational triples, candidate-configuration holdout withheld complete four-candidate blocks, combination holdout withheld query/candidate meaning combinations, and substitution-family holdout withheld event-disagreement score cells. The same masks and [exact split counts](../../phase2/p2_1/analysis/split_counts.csv) were reused. The primary pooled source fit used eight models; all twelve models contributed per-model structural evaluation. Separately, a frozen A/B-only source pool (seeds 1701–1702) evaluated A/B seed 1703 without fitting or aligning the target.

## 7–8. Semantic and internal residual results—descriptive only

Equal-model-weight residual score RMSE on the two complete-block holdouts:

| Incremental budget | Context S | Context I-QC | Configuration S | Configuration I-QC |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 3.581 | 3.542 | 3.617 | 3.590 |
| 64 | 3.193 | 3.501 | 3.287 | 3.571 |
| 128 | 3.050 | 3.710 | 3.245 | 3.784 |
| 256 | **2.406** | 3.781 | **2.642** | 3.836 |

The pooled M1 baseline was **3.490** context and **3.532** configuration. At budget 256, S finite/mixed residual RMSE was **3.260/5.100** on context and **3.524/5.669** on configuration; I-QC was **4.857/7.252** and **4.881/7.373**. The I mixed figures are effectively the unchanged M1 baseline by mathematical necessity. The event-disagreement stress-test score residual RMSE at budget 256 was **2.520** S versus **4.511** I-QC; this stratum was already motivating evidence in P2.1 and cannot independently confirm either theory. All budgets, arms, holdouts, finite/mixed effects, signs, margins, and per-model values are retained in the [aggregate](../../phase2/p2_2/tables/aggregate_metrics.csv) and [per-model](../../phase2/p2_2/tables/per_model_metrics.csv) tables.

## 9. Query versus candidate versus joint participant state

At budget 32 on context, candidate-only I-C scored **3.459** and query-only I-Q **3.548**, compared with joint I-QC **3.542**; candidate-only also scored **3.489** on configuration. This apparent candidate advantage is neither stable across budgets nor specific: the wrong-token control scored **3.453** context and **3.515** configuration at the same budget. At budgets 64 and 128, both I-Q and I-C were generally worse than S and showed no frozen-consistent pathway direction. Budget 256 was preregistered only for I-QC. No P2.2-D pathway-specific signal is licensed, especially given the primary validity failure.

## 10. Negative controls and nuisance audit

The wrong-token, row-permuted-state, and alternate-random-projection controls had the same incremental fitted coefficient counts. At budget 32, wrong-token score RMSE (**3.453** context) was marginally *better* than I-QC (**3.542**); the independent random projection (**3.546**) was virtually identical. At budget 64, I-QC (**3.501**) did not beat wrong-token (**3.481**) or alternate projection (**3.509**) by the frozen 5% threshold. At budget 128, the permuted-state control (**3.562**) outperformed I-QC (**3.710**) on context. I-QC did not satisfy the negative-control criterion at any budget. The candidate pathway offers only one nonendpoint object token, so the wrong-token control used distractor plus zero padding; it cannot by itself establish specificity even if it had lost.

A separate non-deployable nuisance-only diagnostic included physical candidate position, object slots and permutation, evaluation kind, condition/seed identifiers, support, baseline score-magnitude bins, and **true correct-answer position**. The latter is privileged label-side information used solely to probe leakage, not a legitimate predictor. Nuisance score residual RMSE remained **3.688/3.727** at budget 32 and **3.571/3.610** at budget 256 on context/configuration. This bounded audit found no simple nuisance-only improvement over S; it cannot prove absence of all candidate/evaluation leakage, particularly because the central comparison was invalid.

## 11. Cross-seed and generalization

The separate A/B seed-1701/1702 fit evaluated two unseen A/B seed-1703 models without target fitting, calibration, or label-based alignment. Context S versus raw unaligned I-QC residual RMSE was **3.848/3.290** at budget 32, **3.327/3.591** at 64, **4.333/3.388** at 128, and **4.143/8.738** at 256. The ordering is unstable. Hidden-state coordinates are independently initialized and no alignment was preregistered; raw transfer failure alone would not refute a reusable computation. It certainly does not rescue the invalid mixed-interaction comparison. All target-level values are in the per-model table under `A_B_cross_seed_unaligned`.

## 12. Previously observed event-disagreement stress test

The substitution-family holdout is explicitly **previously observed motivating stress test**, not new confirmatory evidence. Its pooled-M1 baseline score residual RMSE was **4.200** across models. S improved to **3.744, 3.700, 2.520** at budgets 64/128/256; I-QC scored **4.204, 4.577, 4.511**. This shows descriptive predictive capacity of the declared semantic bank on those cells, but the arm mismatch prevents a theory-selection claim.

## 13. Complexity/Pareto comparison

The [budget figure](../../phase2/p2_2/figures/budget_residual_rmse.svg) plots residual score error against identical fitted coefficient budgets. It is **descriptive**, not a valid H0-versus-H1 Pareto verdict because the I arm lacks cross-side products. Fixed projection recipes require a seed, dimension, and Gaussian rule rather than fitted basis coefficients, but the dense matrices would occupy substantially more bytes if materialized; both fitted and materialized costs are preserved in the raw table. The M1 baseline remains 51 coefficients and is common to all arms. The combined S+I model at budget 256 scored **3.223** context and **3.366** configuration, worse than S alone at the same total incremental budget; its mixed correction is attributable to S features under the frozen linear construction.

## 14. Per-model consistency

The [paired table](../../phase2/p2_2/tables/paired_internal_vs_semantic.csv) gives all **192** I-minus-S model×budget×holdout differences. I-QC beat S on context in **7/12** models at budget 32, **0/12** at 64, **0/12** at 128, and **0/12** at 256; on configuration the corresponding counts were **6/12, 1/12, 1/12, 0/12**. These counts show the observed pattern without pooling rows as independent replications. They do **not** cure the representational mismatch.

## 15. Failure analysis and integrity

The [expressivity audit](../../phase2/p2_2/analysis/expressivity_audit.json) verifies the near-zero mixed-response change for I-QC over 96 primary metric rows and records the algebraic proof. The [mechanical run manifest](../../phase2/p2_2/analysis/run_manifest.json), [prefit audit](../../phase2/p2_2/analysis/prefit_audit.json), and [reproducibility manifest](p2_2_reproducibility_manifest.json) preserve all completed runs, projection seeds, checkpoint/atlas identities, exact budget counts, code/output hashes, and environment. No run was selected, restarted, extended, or tuned after observing results. The only specification edit was the logged **prefit** pooling clarification. After the expressivity failure was recognized, there was no scientific-semantics repair or new fitting.

## 16. Interpretation category

The frozen score-only classifier emitted `P2.2-A — SEMANTIC EXTENSION SUFFICES`; that output is preserved, not overwritten. The **validity hierarchy takes precedence**: P2.2 is **INVALID for its primary H0/H1 matched-capability comparison** because the declared I arm cannot express the central mixed target while S can. Therefore P2.2-A is **not** accepted as a scientific outcome. The appropriate record is `P2.2-INVALID (primary comparison)`, with descriptive score tables retained. Nor is P2.2-F an appropriate rescue label: this is a known design invalidity, not merely a noisy but fair test.

## 17. What P2.2 establishes

It establishes an operational negative lesson: matching fitted coefficient counts alone does not make residual theories comparable when their feature classes have different algebraic interaction capacity. It also records, descriptively, that the declared bounded semantic features predict more of the particular P2.1 residual scores as budget rises. That predictive fact is not evidence that a distinct compact internal rule is absent.

## 18. What P2.2 does not establish

It does not establish H0, refute H1, discover a reusable internal rule, show query/candidate causal asymmetry, identify a mechanism, explain acquisition, prove a predictive state, or support a grammar/language claim. The observed P2.1 event-stratum and selected-token findings remain prior discovery observations, not independent P2.2 confirmation. D5 remains the bounded causal acquisition anchor; D6/D9 remain invalid; protected D8 seeds remain sealed.

## 19. Updated surviving explanation set

Because the primary comparison is invalid, **the P2.1 explanation set is unchanged**: simple shared-feature interpolation remains a serious partial null; a compact state-dependent residual account is still open; selected-token state can improve higher-capacity within-model score prediction; and no reusable operator, quotient, or grammar has been established. P2.2's descriptive tables may inform later review but do not promote or eliminate a theory.

## 20. Single unresolved scientific disagreement—and stop

Can a genuinely interaction-capable, complexity-matched participant-state description predict residual mixed responses across contexts and models better than a fair semantic interpolation extension? P2.2 does **not** answer that question. This states the unresolved disagreement only; it is **not** a P2.3 design or authorization. P2.2 stops here.
