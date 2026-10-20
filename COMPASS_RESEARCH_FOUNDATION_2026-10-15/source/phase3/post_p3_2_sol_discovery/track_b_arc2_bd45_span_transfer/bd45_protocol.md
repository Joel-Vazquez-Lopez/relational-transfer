# Arc II B-D45 frozen matched rank-support training and transfer test

Status: **discovery only**. Chosen after B-D44's exposure/latent/norm matching audit passed.

Train the exact B-D39 Transformer and MLP, seeds3901/3902, under both B-D44 RANK1 and RANK2 worlds. Same seed starts from exactly the same initial weights in both arms; data row order and batch-index schedule are matched; only the orientation of 5,000 composite bridge translations differs. AdamW lr0.001, batch256, 3000 updates, MSE; save steps500/1500/3000. No held y/full-plane result used to choose any training setting.

After all eight runs, evaluate independent B-D41 x-axis T→R and y-axis T→R tests, a fresh diagonal T→R test generated deterministically with RNG20264501 and translation `(u/√2,u/√2)`, B-D38 full-plane T→R, and B-D38 independent ordinary-family tests. Report MSE/R² and paired order commutator error on the y-axis and full plane. RANK1's diagonal test is off its bridge support; RANK2's diagonal is in its bridge support. The y-axis remains held in both.

Predeclared strong span-transfer criterion for an arm at step3000: y-axis R²≥0.90; y-axis MSE <0.5× B-D41 commutative-average MSE; x-axis bridge R²≥0.95; mean ordinary in-support two-action R²≥0.95; and, for RANK2, diagonal bridge R²≥0.95. If rank2 passes and rank1 fails for both seeds within an architecture, the rank/support account survives a matched exposure control in that architecture, but does not establish explicit operators. If rank2 does not pass, reject the simple support-span sufficiency claim rather than increasing capacity after the result. Treat architecture differences quantitatively and investigate a mundane interpolation explanation before promoting a principle.

All evidence discovery-only. P3.2-E unchanged; no protected/confirmation data.
