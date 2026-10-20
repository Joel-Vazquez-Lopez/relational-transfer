# B-D23 frozen discovery protocol: noisy cue reliability

Status: **discovery only**. Selected after B-D21/B-D22 outcomes, before B-D23 data generation or model training.

## Question

When a source-only cue is mostly reliable but occasionally conflicts with the true relative event during training, does the listener learn the relation, retain the shortcut, or combine them? This distinguishes mere cue availability from the structure and frequency of error-bearing comparisons.

## Matched worlds

Reuse the exact B-D21 unshifted latent candidate scenes, oracle queries, candidate order, and labels. For each candidate, assign the global x-translation sign matching its event with probability **0.95** (P95) or **0.99** (P99), using one frozen uniform random draw per scene so the conflict sets are nested. Translate all three objects by ±0.65 exactly as in B-D21. This preserves relative geometry and keeps the translation marginal balanced. B-D21's NO_CUE (p≈0.5) and CUE (p=1) are the comparison endpoints.

Train new explicitly exploratory models with seeds 3801 and 3802. For each seed, use the same initialization procedure and same minibatch-index schedule as B-D21: fixed oracle messages, frozen speaker/emit, AdamW, learning rate 0.0008, batch128, 3,000 updates. Save at steps 1,000, 2,000, 3,000. No canonical training lineage is changed.

## Frozen outcomes

1. Audit empirical cue reliability by event and answer position in train/test; verify that conflict examples occur in training and test for both P95 and P99.
2. Independent four-choice test accuracy at each checkpoint. Models below 0.95 final accuracy are not assigned a mechanism.
3. On the frozen B-D21 600-scene 2×2 relative-event × cue grid, report accuracy on concordant cells and on conflicting cells **by true relation** and **by cue**. Every grid cell is supported by P95/P99 training, though conflicts are rare in P99.
4. Compute relation and cue score contrasts and inspect their developmental change across saved checkpoints.

## Interpretation and adversarial alternatives

If P95 or P99 moves toward relation on conflict scenes, contradictory training examples may create a useful gradient signal for relational computation. This would be an acquisition finding, not a unique circuit or operator. If they remain cue-dominant, task accuracy can conceal systematic relational failure despite in-support contradictions. If seeds diverge, optimization/history remains central. Loss weighting, finite sample counts, candidate-foil structure, and learned calibration are alternative explanations; follow-up diagnostics must separate them before proposing a law.

P3.2-E remains the canonical endpoint. All new work is discovery-only. Never access protected seeds 1704–1706, P4, or other confirmation material. Do not modify canonical artifacts.
