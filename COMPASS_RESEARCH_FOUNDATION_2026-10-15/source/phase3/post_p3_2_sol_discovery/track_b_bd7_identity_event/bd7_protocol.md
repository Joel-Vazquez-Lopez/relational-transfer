# B-D7 frozen discovery protocol — identity/event factorial

**Status:** Discovery-only prospective composition test chosen after B-D6. Its object is the *factorization of candidate computation*, not a rescue of the failed B-D1 final-CLS map or B-D5 global-token gate. P3.2-E remains unchanged.

B-D6 showed that the K identity-substitution effect reaches CLS almost entirely through changed-token **value content** under comparatively stable block-1 CLS attention. A strong mundane explanation is identity lookup independent of relational event. We will ask whether an identity change and an event change compose at the candidate-token stage and whether the resulting score consequence is predictable before the joint scene is forwarded.

## Fixed 2×2 input grid

For each A/K matched seed 1701–1703, generate 96 fresh base geometries: role(source/target) × third{0,3,4,5} × 12 repetitions. Start with a valid event-0 scene involving identity 1. Define `I` by changing only its one-hot to identity 2. Define `E` by reflecting the source object's velocity about the target velocity, `v_source' = 2 v_target − v_source`, keeping all start positions, target/distractor trajectories, object slots and identities fixed. This turns approach into recede without making an arbitrary OOD trajectory. Require both event geometries to satisfy the task's distance-change validity check. The four scenes are BASE, I, E, IE. In target-role rows, I and E alter different object tokens; this is the clean primary stratum. In source-role rows they alter the same object token and are a predefined stress stratum.

Forward BASE, I and E only. At post-block-0, patch the changed identity token delta `h(I)-h(BASE)` into `h(E)`, then resume untouched candidate block 1 and final LayerNorm. Seal the resulting final vector and score under the **IE oracle query** before forwarding IE. Controls: unmodified E state, exact raw one-hot embedding delta at the same token, wrong distractor-token delta, same-norm random direction, and full-tensor additive finite difference `h(I)+h(E)-h(BASE)` (higher information budget). All predictors are fixed functions of the three observed corners; no learned fitting or outcome-selected capacity.

After sealing, forward IE once. Verify actual resumed/direct equality within 1e-5 and checkpoint unchanged. Report per seed and role final-vector Euclidean RMSE and IE-query score RMSE, each relative to the E identity baseline; actual and predicted score shifts; the internal identity-by-event cross-difference at post-block-0 and final vector; and ranking of controls. A causal patch at this stage tests downstream sufficiency, but a local finite-difference success is not automatically a globally reusable law.

**Primary target-role gate:** in all three K seeds, the changed-token composition must achieve score and vector RMSE ≤0.25×the E identity baseline, beat the raw embedding and random/wrong-token controls, and be no worse than 1.25×the full-tensor additive comparator. If it fails, reject this local token-composition law. Source-role results map its boundary. Even if it passes, the remaining null is a generic first-order local response to two separately changed tokens; the result would require a later independent cross-scene/architecture test before confirmation.

No protected seeds 1704–1706, P4 or confirmatory material. Canonical files are read only; no training.
