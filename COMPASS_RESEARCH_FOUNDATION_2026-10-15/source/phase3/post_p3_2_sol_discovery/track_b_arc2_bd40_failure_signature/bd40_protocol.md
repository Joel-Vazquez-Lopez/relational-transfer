# Arc II B-D40 frozen discovery protocol: failed-composition functional signature

Status: **discovery only**. Selected after B-D39 showed near-perfect in-support learning but failed held T→R: Transformer R²≈0.21, MLP R²<−1.5. This is adaptive characterization, not confirmation.

For each seed3901/3902 and architecture at step3000, use B-D38's 4,000 held T→R scenes and untouched checkpoint. Compare predictions in absolute coordinates to predeclared analytic extensions: exact oracle `R(x+v)`, additive wrong order `R x+v`, first T only `x+v`, last R only `R x`, action-only `R v`, commutative average, and zero. Report MSE to each extension and centered prediction–extension correlation. Also fit a descriptive three-term linear expansion of predictions on `[R x, v, R v]` by ordinary least squares using only the held scenes, reporting coefficients and residual R²; this fit is explicitly post-outcome and cannot be used as a predictor or mechanism. No checkpoint selection, new training, or parameter interventions.

If Transformer predictions are close to `R x+v`, its generalization may be a commutative superposition of single-action effects. If MLP predictions are not close to any simple extension, characterize that ambiguity instead of naming an operator. All discovery-only, no protected/confirmation data.
