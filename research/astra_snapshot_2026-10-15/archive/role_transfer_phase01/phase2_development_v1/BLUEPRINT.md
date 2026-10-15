# Phase 2 development blueprint — 14 October 2026

## Status and scientific scope

This is **development on previously examined graphs**, not independent confirmation. It uses the eight 10-entity, degree-four graph families in `../rq2_rewire_repair_v1/GRAPH_MANIFEST.json`. Each family has the same withheld focal pair in a sham and a high-overlap two-switch rewiring condition. The package changes common partners **and** other local routes/co-occurrences; the estimand is sensitivity to this *rewiring package*, not a unique common-partner effect. The strict 12-entity factorial missed its prespecified success gate. That negative result remains part of the record.

## Five arms (four requested objectives plus a compute control)

All models have the same two-tower Transformer and two auxiliary heads; the heads are trained only in the two auxiliary arms. Every arm sees the same 1,280 native training configurations, balanced distractor schedule, scene generation, answer positions, and 6,000 AdamW updates at batch 128 and learning rate 0.0008. Paired arms use the same initialization and random stream for each family, condition, and seed. Two new paired seeds, 321001 and 321002, are used. The previously saved 96-model study is historical motivation, not part of this analysis.

Let `z(x)` be the four candidate scores and `CE(z,y)` the ordinary four-choice cross-entropy. Let `P=(1,0,3,2)` exchange the actor/patient candidates within each event. `T(x)` swaps the two message participant names, swaps their identity channels in **every** scene while leaving motion and event features fixed, then reorders the four candidate scenes by `P`. The correct answer index therefore becomes `P(y)`. For relative scores, `c(z)=z-mean(z)`.

| Arm | Exact loss | Purpose |
| --- | --- | --- |
| Standard | `CE(z(x), y)` | Native-task reference |
| Participant-only | `CE(z(x), y) + 0.5 L_part` | Parameter- and supervision-count matched identity control |
| Ordered-role | `CE(z(x), y) + 0.5 L_role` | Explicit actor and patient labels |
| Role-swap consistency | `CE(z(x), y) + 0.05 mean((c(z(Tx))-P[c(z(x))])^2)` | Enforces score-pattern equivariance without an extra gold role label |
| Permutation compute control | `CE(z(x), y) + 0.5 CE(z(Px), P(y))` | Extra candidate-order forward/backward pass and native supervision without a role swap |

`L_part` and `L_role` are the mean of four cross-entropies from message and correct-scene heads, exactly as in `role_aux_v1.py`. Participant heads predict sorted identities; ordered heads predict actor then patient. The role-swap loss compares centered logits, so arbitrary score offsets do not matter. The initial normalized variant was nearly inert. Raw-score weights 0.5 and 0.1 failed the familiar-learning preflight (50.2% and 84.0% familiar accuracy at 960 updates); weight **0.05** reached 99.0%. This choice used only the examined graph and familiar accuracy, before full development training; all preflight attempts are retained. `T` is **not** merely replaying reversed training examples: it constrains paired score vectors from the same batch. The compute control uses the same second model pass but only repeats native supervision under a candidate permutation. Compute is closely matched for the role-swap/control contrast; the explicit role heads add a smaller, different cost. Record wall time and report this remaining mismatch.

The simultaneous message-and-scene swap is an **entity-label equivariance** constraint. It does not by itself require the model to select the correct actor or patient. That is a substantive limitation of the proposal's rule, and a null result for this arm will not imply that every role-aware consistency objective fails. A message-only role reversal is a separate intervention and is not silently substituted here.

## Runs, outcomes, and analysis

Eight graph families × two conditions × two paired seeds × five arms = **160 models**. The preflight uses one examined graph, one seed, and 960 updates per arm; it tests transformations, gradients, finite losses, familiar learning, and identical batches. It is excluded from analysis. Full runs save final and intermediate checkpoints at 1,500, 3,000, 4,500, and 6,000 updates, source/manifest/protocol hashes, ledgers, loss traces, model hashes, and raw pair-level evaluation.

The **primary development contrast** is the ordered-role minus participant-only difference in the high-minus-sham focal binary-role accuracy penalty, computed within paired seed, then averaged within family, then equally over eight families. The secondary role-swap contrast uses the permutation compute control. The ordered-minus-role-swap difference addresses whether explicit labels outperform the proposal's consistency rule. Report all four requested arms and the control, family effects, a 95% family-level interval and exact two-sided family sign-flip p values as descriptive development statistics. No development p value is independent confirmation.

The primary operating point is the fixed 6,000-update checkpoint and fixed scene suite 260107; suite 260108 is a rendering check. Report four-choice, binary role, and event accuracy on the focal pair; familiar accuracy; and performance on all other unseen pairs. A model at or above 97% focal role accuracy in both graph conditions is flagged as ceiling-saturated. A flatter penalty from saturation alone is not interpreted as reduced structural dependence. Also report error rates and a log-odds sensitivity measure with a fixed 0.5-count continuity correction.

**Performance matching is secondary and outcome-blind to the focal contrast:** for each non-ordered arm, select one common checkpoint (1,500/3,000/4,500/6,000) closest to ordered-role at 6,000 on the average of eight preselected *nonfocal* unseen pairs per condition and family, subject to familiar accuracy ≥95%. The same checkpoint is used for every family/seed/cell of that arm. Declare matching successful only within 2 percentage points of ordered-role nonfocal accuracy; otherwise state that performance-matched inference is unavailable. Then examine the focal structural contrast at the selected checkpoints. This is a development sensitivity analysis, not a new confirmatory test.

## Compute estimate and gates

The historical two-worker 96-model run took about four hours on this machine; 160 models with four checkpoints and extra evaluation are estimated at **7–11 wall-clock hours** with two workers, plus preflight. Disk estimate: under 1 GB for checkpoints, metrics, and logs; verify free space before launch. The runner refuses completed outputs and resumes only pending exact configurations. Technical failures may be rerun with the same configuration; no exclusion or replacement may depend on scores.

**Stop before full training** if the transformation changes the gold candidate meaning, graph/data audit fails, gradients are zero/nonfinite, or the preflight cannot learn familiar examples by its diagnostic 960-update checkpoint. **Stop and report during full training** for a repeated technical error, corrupt checkpoint, or a generator/label validity failure. Do not stop for an unfavorable score. After analysis, advance to fresh independent graphs only if a substantive interaction is not entirely ceiling-driven, survives the participant/compute controls, and shows a consistent direction across graph families. Otherwise report the bounded result and revise the hypothesis.
