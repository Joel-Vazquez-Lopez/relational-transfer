# Arc II B-D46 frozen hostile null: explicit low-parameter bilinear regression

Status: **discovery only**. Selected after B-D45's matched RANK2 bridge produced high held y/full-plane R² in Transformer and MLP models. The null is that this follows from identifying a small affine response surface, with no reusable state transition.

Fit an explicit predictor separately for rotation signs ±1 using **only** B-D44 training rows. First fit `A_s,b_s` in `y≈A_s x+b_s` on single-R rows. Then, holding those estimates fixed, fit `B_s` in bridge residual `y−(A_s x+b_s)≈B_s v` on T→R bridge rows by minimum-norm least squares. The final predictor is `A_s x+b_s+B_s v`. No held y/full-plane output, Transformer/MLP checkpoint or test metric enters fitting or model selection. No known rotation matrix or oracle formula is inserted. The same fixed regression procedure is applied to RANK1 and RANK2 training arrays.

Evaluate independent B-D41 x/y-axis T→R tests and B-D38 full-plane T→R. Report MSE/R², fitted A/B matrices, design ranks and coefficient norms. Compare qualitatively with B-D45 neural results. If this tiny model reproduces rank1 failure/rank2 success, the B-D45 behavioral result is consistent with ordinary identification of an affine bilinear function; it is not evidence of an internal reusable operation. The models' different rank1 extensions remain architecture-dependent, but rank2 success alone cannot identify mechanism.

P3.2-E unchanged; all discovery-only; no protected/confirmation data.
