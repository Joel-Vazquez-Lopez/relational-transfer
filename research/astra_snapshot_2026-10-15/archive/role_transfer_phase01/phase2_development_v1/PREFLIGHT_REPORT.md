# Phase 2 learning preflight

The source manifest, all 16 graph cells, 1,280-row pools, pair withholding, bidirectional role exposure, swap involution, candidate-label permutation, finite losses, and nonzero query/scene gradients passed the code audit. The transformation checks were run before full training.

One already examined sham graph (family 0), one new start (321001), 960 updates per attempt, and familiar four-choice accuracy were used to decide whether the training procedure learned the native task. These are **not** independent research outcomes. The focal score shown in saved checkpoint JSON was not used to select the loss weight.

| Arm or role-swap weight | Familiar accuracy at 960 updates | Preflight decision |
| --- | ---: | --- |
| Standard | 98.44% | pass |
| Participant-only | 98.63% | pass |
| Ordered-role | 98.83% | pass |
| Permutation compute control | 98.44% | pass |
| Role-swap, raw-score weight 0.5 | 50.20% | fail; retained in `preflight/.../role_swap` |
| Role-swap, raw-score weight 0.1 | 83.98% | fail; retained in `preflight/.../role_swap_weight_0p1` |
| Role-swap, raw-score weight 0.05 | 99.02% | pass; retained in `preflight/.../role_swap_weight_0p05` |

Before those attempts, a normalized score loss (score differences divided by 10) was checked on an existing ordered-role checkpoint and found nearly inert (mean squared discrepancy 0.0056). The raw-score loss on that same checkpoint was 0.558. The initial weight 0.5 then blocked familiar learning, so the predeclared preflight gate stopped full training. The smallest tested raw-score weight, 0.05, passed. It is now frozen for the complete development comparison; no further weight selection will use graph-condition or focal outcomes.

This audit also exposes a conceptual limit: jointly swapping names in message and scenes enforces identity-relabeling equivariance. It does not itself force correct role assignment. The arm may therefore fail even if a different role-reversal constraint could help.
