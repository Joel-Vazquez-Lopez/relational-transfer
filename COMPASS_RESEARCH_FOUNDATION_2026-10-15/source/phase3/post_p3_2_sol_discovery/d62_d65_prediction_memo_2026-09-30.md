# Discovery memo: a bounded prospective prediction, and a failed stability predictor

**Date:** 2026-09-30. **Status:** discovery only. This memo covers D62–D65 after the D52–D59 collision-partition study. The canonical endpoint remains P3.2-E. No protected seed 1704–1706, P4 material, or other protected confirmation artifact was used. Canonical code, schedules, and A results were read only. All new scripts, checkpoints, outputs, plans, and hash manifests are in this discovery directory.

## Question and decision rule

The previous evidence suggested that training rows where identifying two H identities makes the positive meaning collide with a foil can force separation during learning. A stronger proposed explanation was that early state plus training dynamics would predict whether a merged solution remains stable, especially in the variable noncollision L arm. D62 and D63 tested candidate recovery assays on the known seed-1701/1702 outcomes. Both had usefulness gates written before their own outcomes. After they failed, D64 froze a narrower K-versus-L trajectory and behavior prediction before running the corresponding arms on permitted discovery seed 1703. D65 checked a specific interpretation of the older L trajectories without training new models.

## D62: recovery after merging a trained checkpoint did not predict the known trajectories

At update 2,000 of the original untied K and L models, D62 averaged identity-1/2 message-symbol rows and candidate input columns, then released the tie for 256 updates on the original indexed training schedule with fresh AdamW. Unmerged reference branches used the same subsequent batches. The prespecified usefulness gate required the released L1701 source contrast to exceed L1702 by 0.10 and both K branches to exceed both L branches. It failed. At 256 updates, all-H source contrast was K1701 **0.857**, K1702 **0.165**, L1701 **0.147**, L1702 **0.098**. The L gap was only **0.049**, and K1702 was not a robust recovery case despite its strong original final result. This assay is not a predictor of the observed late outcomes. It also restarted AdamW, because the original checkpoint did not save optimizer state. The all-H readout includes pairs other than 1/2; D63 and D65 use an exact 1/2 readout.

## D63: an exactly tied L history fits but does not recreate the untied route

D63 trained L for 2,000 updates with identity-1/2 message-symbol rows and candidate-input columns projected equal after every step, saving model **and** AdamW state. The tied parents fit sampled training rows (cross entropy **0.0208** for seed1701; **0.0133** for seed1702). Two paired release branches continued for 1,000 updates with either retained optimizer moments or symmetrized identity-1/2 first and second moments. The predeclared predictor gate required seed1701's exact 1/2 source contrast to exceed seed1702's by at least 0.15 in a branch, conditional on tied-parent CE below 0.03. The fit condition passed; the contrast condition failed. At release update1,000, retained-moments source contrast was **0.0272 versus 0.0231**; symmetrized was **0.0280 versus 0.0156**. Moment manipulation had little effect on this short trajectory.

This is a negative result for **the assay**, not proof that the merged solution is dynamically stable in the original runs. The tied history changed the route the model took into the update-2,000 state. D65 read the original untied L checkpoints: exact 1/2 source contrast was **0.372/0.426/0.875** for seed1701 at updates 2,000/4,000/8,000 and **0.283/0.268/0.212** for seed1702. Both original models had substantial 1/2 contrast by update2,000; one later amplified it, the other attenuated it. D63 removed that early distinction and therefore cannot recreate the original fork by simply releasing the tie. The specific cause of the later divergence remains unidentified.

## D64: a frozen within-discovery prediction passed on seed 1703

The D64 plan and readout code were saved before examining K/L seed1703 outcomes. The input audit found **51,207** K modified rows, all positive–foil quotient collisions, and **102,395** L modified rows with zero collisions. Both arms used the same seed1703 initialization, indexed schedule, positive oracle meanings, optimizer, and 8,000-update length. Canonical A1703 was an archived comparison: withheld S/R joint accuracy **0.822/0.493**.

| Arm and update | Source H query contrast | Withheld R joint accuracy |
|---|---:|---:|
| K at 2,000 | 0.887 | 0.866 |
| L at 2,000 | 0.235 | 0.503 |
| K at 8,000 | 0.909 | 0.932 |
| L at 8,000 | 0.147 | 0.606 |

All **six** frozen inequalities passed: at update2,000, K source contrast ≥0.75, L ≤0.40, and K−L withheld-R accuracy ≥0.25; at update8,000, K source contrast ≥0.80, K withheld-R accuracy ≥0.85, and K−L withheld-R accuracy ≥0.10. Final withheld-S joint accuracy was **0.939** for K versus **0.847** for L. Familiar-S joint accuracy was **0.9988** versus **0.9984**, so the large withheld-R gap is not a broad familiar-task failure. The final withheld-R K−L difference was **0.327**.

This is a genuine *prospective prediction within the discovery programme* for this seed and these frozen thresholds, but it is still discovery-only: seed1703 belongs to the same synthetic task, architecture, intervention design, and seed family chosen after many adaptive diagnostics. It does not independently confirm a general acquisition law or a language-model mechanism. The K result now holds for three permitted discovery seeds; L's late behavior remains variable (seed1701 partial recovery, seeds1702/1703 weak recovery).

The K/L runs recorded identical initial model-state hashes. Both evaluation routines recorded unchanged parameter state before versus after evaluation. A final integrity check recomputed **94 recorded file hashes across six D62–D65 manifests**, with no missing files or mismatches.

## What changed in the explanation

The reliable part is more precise than the original broad “contextual distinguishability” story. The **labeled choice task** can reject the merged description of identities1/2 on a small number of examples. In this system, those quotient-breaking examples repeatedly select an acquisition path with preserved relational distinction and stronger unsupported H–H behavior. That is a bounded empirical route, not yet a mechanism: K also changes hard-negative gradients, candidate scenes, and the path taken by both message and candidate encoders.

The simple early-state stability proposal has **not** yielded a usable scalar predictor. D62 failed across seeds; D63 imposed a different history and failed to predict the known L divergence. D65 shows why a fully merged update-2,000 state is an especially poor proxy for the actual L runs. The unresolved scientific target is a path-dependent account that predicts growth versus suppression of an already present distinction under noncollision experience, without fitting the predictor to the final result. It should track task gradients and the joint message/candidate computation through training, not infer stability from static graph topology, a tie penalty, or one-step gradient alone.

## Boundary and next test

The K/L contrast still confounds quotient collision with the size and direction of hard-negative gradients. A sharper future design would hold the *experienced gradient field near the current model trajectory* closer while changing whether a labeled row explicitly breaks the identity quotient, then freeze predictions on a genuinely new task or architecture. If feasible, save complete optimizer states and developmental checkpoints from the start; D62/D63 showed that after-the-fact merger is not a faithful proxy for the acquired state. The next high-value target is whether a prospective trajectory-level descriptor can predict the **soft** L/E route, including late divergence, before final training. Until then, the quotient-collision finding is a strong bounded discovery result and the general “internal grammar” claim remains open.

## Artifact index

- D62: `d62_plan.json`, `d62_merge_recovery.py`, `d62_merge_recovery.csv`, `d62_manifest.json`.
- D63: `d63_plan.json`, `d63_tied_l_release.py`, `exploratory_D63_tied_L_release/`, `d63_manifest.json`.
- D64: `d64_plan.json`, `d64_seed1703_prediction.py`, `d64_input_audit.json`, `d64_prediction_readout.py`, `exploratory_D64_seed1703/` including frozen prediction readout, verdict, manifests, and model checkpoints.
- D65: `d65_plan.json`, `d65_original_l_exact12.py`, `d65_original_L_exact12.csv`, `d65_manifest.json`.
