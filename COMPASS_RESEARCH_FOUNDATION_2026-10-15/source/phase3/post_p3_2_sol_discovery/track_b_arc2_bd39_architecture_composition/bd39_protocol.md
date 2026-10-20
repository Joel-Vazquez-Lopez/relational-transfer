# Arc II B-D39 frozen discovery protocol: learned held affine composition

Status: **discovery only**. Selected after B-D38 shortcut audit passed. The primary held family T→R is absent from training and requires a sign-dependent interaction between translation and rotation.

## Models and training

Two whole-system regressors receive identical three-token input and 2D target. **Transformer:** shared 7→64 token embedding, learned CLS and positions, two standard four-head self-attention/MLP residual blocks (the existing `pilot.Block` implementation), final LayerNorm and linear 64→2 output. **MLP:** flatten 3×7, four hidden layers width128 with GELU, linear 128→2. Report parameter counts; matching is approximate. Train both architectures at new discovery seeds3901/3902 on the exact B-D38 100,000 examples, same batch-index schedule per seed (RNG20263900+seed), AdamW lr0.001, batch256, 3000 updates, MSE loss. Save steps500/1500/3000. No held T→R result may select architecture, checkpoint, optimizer or capacity. If a model fails in-support, report it without rescue.

## Frozen evaluation

After all four runs finish, evaluate each saved checkpoint on 4,000 independent examples per family. Report family MSE and variance-normalized R². Primary held success criterion: at step3000 held T→R R²≥0.90 *and* held MSE <0.5× B-D38 commutative-average MSE. Require mean in-support two-action R²≥0.95 to interpret success. Also form a paired R→T input for each held T→R scene by swapping the two action tokens; compare predicted order difference to the analytic commutator `R_s v-v` by MSE, cosine and magnitude ratio. The paired R→T is an in-support action order with fresh continuous parameters.

Accuracy does not establish a reusable internal operator. If both architectures pass, seek a broader functional/developmental explanation before mechanistic attribution. If only one passes, test whether capacity/optimization or architecture explains it. If both fail, inspect which operation families and interactions were learned and redesign rather than post-hoc increase capacity.

P3.2-E unchanged; no protected/confirmation data; all models and outputs discovery-only.
