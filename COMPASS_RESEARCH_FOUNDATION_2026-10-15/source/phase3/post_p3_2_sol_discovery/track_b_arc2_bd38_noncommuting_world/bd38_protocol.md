# Arc II B-D38 frozen redesign: held translation→rotation

Status: **discovery only**. Selected after the post-audit B-D37 additive shortcut was identified. B-D37 remains a logged failed design; no B-D37 model was trained.

Retain the exact B-D37 continuous 2D action generator and token encoding. Training families are single R, single T, R→R, T→T, and **R→T**, 20,000 examples each. Hold **all T→R** from training and model selection; independent tests contain 4,000 examples per family. New data RNG 20263801 training and 20263802 test. The held oracle is `R_s(x+v)=R_s x+R_s v`. Its difference from the additive single-action construction `R_s x+v` is `(R_s-I)v`, which is nonzero with probability one under the continuous v distribution. The minimum successful external function must use a sign-dependent interaction with the translation; this still does not require an explicit internal sequential operator.

Before model training, audit exact type disjointness, input duplication, oracle regeneration, rotation sign/parameter balance, and train/test marginal coverage. Evaluate held T→R predictors: zero/x-only, action-only `R_s v`, first T only `x+v`, last R only `R_s x`, additive/commutative `R_s x+v`, and the average of additive and exact order outcomes. A candidate using bilinear `R_s v` together with `R_s x` is the exact oracle and is reported as such, not treated as a shortcut.

Proceed only if no T→R train input, no exact duplicate, float32 oracle error <1e-6, exact regeneration of stored outputs, and no non-oracle restricted baseline attains held R²≥0.9. If the gate fails, redesign before training. All artifacts remain discovery-only; no protected/confirmation material or canonical mutation.
