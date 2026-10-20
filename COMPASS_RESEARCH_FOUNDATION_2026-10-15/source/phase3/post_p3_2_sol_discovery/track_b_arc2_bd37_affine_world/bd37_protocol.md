# Arc II B-D37 frozen discovery protocol: affine composition microscope and shortcut audit

Status: **discovery only**. Selected after B-D36 closed the cue-classification microscope. No previous model/world artifact is changed.

## Task and held composition

State `x∈[-1,1]^2`. An action is either `R_s`, a ±90° rotation around the origin with sign `s∈{-1,+1}`, or `T_v`, a translation by continuous `v∈[-0.8,0.8]^2`. Apply actions left-to-right. The output is the final 2D coordinate. `R_s` and `T_v` do not commute: `T_v(R_s x)=R_s x+v`, while `R_s(T_v x)=R_s x+R_s v`. Translation parameters and initial states vary continuously. The action token provides kind one-hot, rotation `(cosθ,sinθ)` or translation vector; the input token provides `x`. PAD marks a single-action sequence.

Training families: single R, single T, R→R, T→T and T→R. **All R→T examples are withheld** from training and model selection. The primary held test is fresh R→T with independent continuous x/v/sign. In-support independent tests cover all five training families. Use exactly 20,000 examples per family (100,000 total), and 4,000 independent examples per test family (24,000 total). Data RNG 20263701 for training and 20263702 for test. All family labels, x, v and rotation sign are balanced/randomized independently; no object identity, candidate order, renderer, or image nuisance exists.

## Audit before model training

Freeze generator and audit code before generating data. Check exact oracle output, type-family disjointness, duplicate input overlap, distribution/marginal balance, held sign and parameter coverage, and counterfactual noncommutation. Evaluate analytic restricted predictors on held R→T: x-only (zero conditional mean for balanced signs/v), action-only (`v`), first-action-only (`R_s x`), last-action-only (`x+v`), order-insensitive bag (`R_s(x+v)`), and a commutative average of the two order outcomes. Report MSE and variance-normalized R². These baselines are not exhaustive; in particular an unrestricted function that directly computes `R_s x+v` solves the held family and is functionally compositional even if internally not sequential.

Proceed to model training only if: no R→T training input, no exact train/test duplicate, oracle error <1e-6 after float32 storage, exact regeneration of stored outputs, and no restricted baseline achieves held R²≥0.9. Otherwise redesign this world first. This is a diagnostic microscope, not a proof that any successful network has explicit action operators.

Protected seeds1704–1706 and confirmation material remain sealed. P3.2-E unchanged. All outputs under this new directory.
