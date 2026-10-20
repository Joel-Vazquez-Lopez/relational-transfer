# B-D30 frozen discovery protocol: additive query factorization

Status: **discovery only**. Chosen after B-D29 rejected arbitrary identity correspondence for eight-pair orthogonal alignment.

Test whether the event-query difference bank across 30 ordered identities is captured by `intercept + source effect + target effect`, with six cyclic ordered pairs held out as in B-D27. Fit multivariate least squares on the remaining 24 pairs, with no candidate vectors, scores, or held query use. Fit separately within each NO_CUE, CUE, P95 and P99 seed3801/3802 model at step3000. Compare to intercept only, source only, target only, and independent wrong pair-label permutation (32 frozen draws, RNG202630). Use the same scene grid and held pairs for functional score evaluation with the untouched model candidate branch.

Report held query cosine, norm-relative query error, score RMSE to actual query, concordant/conflict accuracies, and relation/cue contrasts. Also report train and held variance explained for the additive model. A good additive reconstruction would explain the small fitting budget in B-D28 as a query-code factorization, not a reusable state operator. Poor reconstruction would push toward identity-pair interactions or a nonlinear compositional code. No model retraining, no protected/confirmation data, no canonical mutation.
