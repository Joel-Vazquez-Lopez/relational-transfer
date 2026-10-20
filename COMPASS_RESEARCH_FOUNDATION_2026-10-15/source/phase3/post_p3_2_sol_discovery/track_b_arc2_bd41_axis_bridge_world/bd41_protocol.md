# Arc II B-D41 frozen bridge-world protocol

Status: **discovery only**. Selected after B-D39 failure and B-D40's exact wrong-order additive Transformer extension. This is a new training condition, not a rescue of B-D39's frozen test.

Retain B-D38's 100,000 training examples and all its independent tests, but replace 5,000 deterministically chosen R→T examples with 5,000 new **T→R bridge** examples where translation `v=(u,0)`, `u∼Uniform[-0.8,0.8]`, x uniform in the original square and rotation sign balanced. Re-shuffle the 100,000 rows. Training thus has 15,000 R→T and 5,000 T→R-x examples, keeping all other families at 20,000. Use RNG 20264101 for bridge and 20264103 for replacement/shuffle. Generate two independent 4,000-scene T→R tests with translation on x-axis and y-axis using RNG 20264102. The original B-D38 full-plane T→R test remains a secondary evaluation.

The **primary held composition** is T→R with `v=(0,u)`. No y-axis T→R appears in training. It requires the previously unobserved interaction of rotation sign with the y component of translation. Audit exact support, duplicates, sign/marginal balance, oracle and noncommutation. Evaluate the same restricted baselines as B-D38, especially wrong-order additive `R x+v` and commutative average. Train models only if no y-axis T→R train row, no duplicate, oracle error <1e-6, exact regeneration, and no non-oracle baseline R²≥0.9 on the primary y-axis test.

If the bridge induces y-axis transfer, inspect whether it is a general bilinear rule, an architecture-specific interpolation, or an actual reusable computational transformation. If it does not, quantify what the model learned on the x-axis bridge before further redesign. All discovery-only; no protected/confirmation data or canonical mutation.
