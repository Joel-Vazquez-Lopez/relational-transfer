# Arc II B-D43 frozen positive control: learned shared transition applied twice

Status: **discovery only**. Selected after B-D39 zero-bridge Transformer/MLP failures and B-D42 restricted-bridge partial transfer. This model is deliberately given a sequential parameter-sharing bias; it is a positive control for task learnability under that bias, not evidence that the unconstrained models should discover explicit operators.

A single learned MLP transition `f(state_2D, action_token_7D)→state_2D` with four hidden layers width128 and GELU is applied to action1 and then action2. PAD deterministically skips the second update. No rotation/translation formula is hard-coded; only the *reuse and order* of the transition is. Parameter count is approximately matched to the B-D39 MLP. Train new models at seeds3901/3902 on the **original no-bridge B-D38** 100,000 examples, same batch-index schedule, AdamW lr0.001, batch256, 3000 updates and MSE loss. Save steps500/1500/3000. Held T→R outcomes are not used for model or checkpoint selection.

After training both seeds, evaluate all saved checkpoints on the exact B-D38 independent family tests, the B-D41 fresh y-axis T→R test, and paired T→R versus R→T order difference. Primary success gate matches B-D39: step3000 full-plane held T→R R²≥0.90, held MSE <0.5× B-D38 commutative-average MSE, and mean in-support two-action R²≥0.95. The y-axis test is supplemental. If it succeeds, reuse-by-construction enables systematic held composition with the same experience that failed in flat models; this isolates a useful inductive bias, not an acquired Transformer-native law. If it fails, examine whether the single-step map was learned or failed out-of-range state transport.

All discovery-only. P3.2-E unchanged; no protected/confirmation data.
