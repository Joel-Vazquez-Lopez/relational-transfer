# B-D5 frozen discovery protocol — block-0 object-token substitution transport

**Status:** Discovery-only prospective-within-discovery test, selected after B-D4 token-patch result. The B-D4 fit pairs are previously observed; every B-D5 test pair is a newly generated raw scene. No B-D5 test alternative is forwarded before the donor deltas and score predictions are sealed. Canonical P3.2-E is unchanged.

## Scientific question and null

B-D4 found that a post-block-0 changed-object-token patch recovers nearly all of K's identity-substitution effect, and block-1 attention is necessary for the large final contrast. Does a **single reusable token substitution delta** carry this effect to new candidate scenes, or is the B-D4 patch merely a same-scene transplant of an identity embedding difference?

The strong nulls are: (i) exact raw identity embedding difference, (ii) leaving the token unchanged, (iii) nearest donor delta memorization, (iv) arbitrary same-norm direction, and (v) role mismatch. A portable delta that loses to raw embedding is not a learned native operation. Strong score prediction alone is insufficient if final candidate vectors or the held relational-pair stratum fail.

## Models and scenes

Use A and K matched seed-1701/1702/1703 checkpoints already audited in B-D4. Fit donors are all 192 B-D4 scene pairs, with source/target substitution 1→2, thirds 3/4/5, both events and 16 geometries/cell. The B-D5 test set has new deterministic geometry, 12 scenes for each source/target × third{0,3,4,5} × event, 192 pairs/model. Third=0 is structurally held out from the donor set. One identity one-hot changes in the source or target object row; all other raw input, object slots, xy trajectories and distractor are identical within each pair. The fixed oracle alt query reads the candidate score. No seed 1704–1706 or confirmatory material.

## Frozen predictors and budget

At candidate post-block-0, compute donor `Δ_i = h_alt[changed object token] - h_base[changed object token]`, 48 dimensions. Fit **global mean Δ** (48 scalars) across all 192 donors and **role-conditional means** (2×48 scalars) as a predefined bounded alternative. Neither sees third identity, event, geometry, score, or test-alt state. Apply to the changed token of each test BASE state and resume the untouched candidate block 1 and final LayerNorm. Predict final candidate vector and alt-query score before test-alt forwarding.

Baselines: identity patch; exact input-layer one-hot embedding difference `candidate.inp([id2]-[id1])` added to the block-0 token; Euclidean nearest donor BASE changed-token delta (192 stored donors); wrong-role mean; and fixed random direction matched to global-delta norm. The embedding comparator uses only model weights, no test outcomes. Record 48/96 scalar budgets versus 192-donor memory. No hyperparameter selection from B-D5 test outcomes.

## Outcomes and gates

After predictions and hashes are frozen, forward the B-D5 test alternative once. Verify direct versus resumed output within 1e-5 and checkpoint state unchanged. Report pooled and per-model final-vector Euclidean RMSE and score RMSE for every predictor, split by role and by seen thirds 3/4/5 versus held third 0. Also report the actual score-shift magnitude and the predicted shift, so a tiny response cannot masquerade as skill.

The **global token-operation candidate** survives this discovery test only if, on all three K seeds: (1) its pooled score and final-vector errors are ≤0.50×identity, (2) both errors are ≤0.90×raw-embedding and ≤0.90×nearest-donor errors, (3) both roles individually and held-third=0 have score error ≤0.75×identity, and (4) the actual target-state replay is exact. If the global law fails but the frozen role-conditional alternative passes the same criteria, report a *role-conditional candidate* requiring one context bit, not a global operation. Neither result by itself is independent confirmation or proof of relational processing; B-D4 donor selection was adaptive, and the K training contrast is discovery-only. If both fail, stop this token-transport candidate without adding third/event/geometry-specific exceptions.
