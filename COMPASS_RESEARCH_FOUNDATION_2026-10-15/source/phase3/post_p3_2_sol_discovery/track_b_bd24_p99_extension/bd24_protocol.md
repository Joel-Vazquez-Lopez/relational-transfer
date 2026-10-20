# B-D24 frozen discovery protocol: P99 trajectory extension

Status: **discovery only**. Selected after B-D23 found a strong P95 relation trend and seed-divergent P99 relation uptake at step 3,000. This protocol is frozen before the extended outcomes.

## Question

Does a 1% contradiction rate eventually drive relational computation, or do the P99 listeners remain cue-dominant despite continued exposure to the same finite dataset?

## Method

Reproduce the B-D23 P99 seed3801 and seed3802 runs from initialization with exactly the same architecture, data hash, optimizer, learning rate, batch-index stream, and deterministic settings. At update 3,000, **assert exact model-state equality** with each frozen B-D23 checkpoint. Abort if either differs. Continue both runs through 8,000 updates without changing the optimizer state or schedule. Save checkpoints at 4,000, 6,000, and 8,000; preserve the 3,000 baseline. No retraining of canonical models occurs.

At 3,000/4,000/6,000/8,000, evaluate independent four-choice test accuracy and the frozen 600-scene 2×2 relative-event × cue grid. Report conflict accuracy by true relation, concordant accuracy, and relation/cue score contrasts. Do not select checkpoints by their outcome. If test accuracy falls, report overfitting; if conflict relation accuracy rises while test stays high, report delayed acquisition. Divergence between seeds is itself an outcome, not a reason to add exceptions.

This continuation is adaptive discovery and cannot independently confirm a principle. Protected seeds 1704–1706 and other confirmation material remain sealed; P3.2-E and all canonical artifacts remain unchanged.
