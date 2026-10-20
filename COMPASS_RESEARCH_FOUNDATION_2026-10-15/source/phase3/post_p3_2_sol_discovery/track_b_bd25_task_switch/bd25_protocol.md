# B-D25 frozen discovery protocol: contradiction task switch

Status: **discovery only**. Selected after B-D24's two P99 seeds failed to converge by step8,000 despite exact trajectory reproduction.

## Question

Does the time at which contradiction-bearing examples appear causally affect relational acquisition, beyond the final mixture of cue reliability and total update count?

## Matched switches

Use B-D21 CUE (p=1) and B-D23 P95 (p≈.95) arrays built from the same latent scenes. For each seed 3801 and 3802, reproduce an existing CUE or P95 run from initialization through step1,000 with the same AdamW optimizer, initialization, and minibatch-index stream. Assert exact equality of every model parameter at step1,000 with the corresponding frozen checkpoint. Preserve optimizer state. Then switch the training array at update1,001:

1. **CUE→P95:** introduce cue–relation contradictions after shortcut acquisition.
2. **P95→CUE:** remove contradictions after early mixed-cue acquisition.

Continue both to step3,000 using the same batch indices as uninterrupted controls. Save at steps2,000 and3,000. This is 2 switch directions × 2 seeds, with no canonical model training.

## Frozen outcomes

At steps1,000/2,000/3,000, evaluate four-choice independent accuracy on both CUE and P95 tests and the frozen 600-scene 2×2 relative-event × cue grid. Report conflict relation accuracy, concordant accuracy, and relation/cue score contrasts. Compare to the existing uninterrupted CUE and P95 trajectories at the same seeds and steps.

Interpretation: CUE→P95 uptake comparable to P95-from-start would argue against strong early shortcut lock-in; failure despite P95-from-start uptake would support a history-dependent barrier. P95→CUE retention would show persistence after contradictions disappear; collapse toward CUE-from-start would show ongoing gradient dependence. These are discovery contrasts, not independent confirmation. The same finite dataset and fixed optimization schedule remain possible explanations for quantitative rates.

P3.2-E remains the canonical endpoint. Never access protected seeds 1704–1706 or other confirmation material. Keep all new outputs in this directory and leave canonical artifacts untouched.
