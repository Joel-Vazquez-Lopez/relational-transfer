# Arc II B-D42 frozen protocol: does a one-axis composition bridge transfer?

Status: **discovery only**. Selected after B-D41 audit passed. This is a separate training condition chosen after B-D39 failed; it is not post-hoc repair of B-D39's frozen result.

Train the exact B-D39 Transformer and MLP architectures at the same discovery seeds3901/3902, optimizer, batch256, rate0.001, 3000 updates and batch-index schedule, now on B-D41's 100,000-row training array containing 5,000 x-axis T→R bridge examples. Save steps500/1500/3000. No y-axis or full-plane held T→R outcomes may select architecture, capacity, checkpoint or regularization.

Evaluate each saved checkpoint after all runs finish on: independent x-axis T→R bridge test, primary held y-axis T→R test, original B-D38 full-plane T→R test, and five original training families. Report MSE and R² for each. Report paired order-difference error on x/y/full-plane tests. Compare endpoint metrics with the frozen B-D39 no-bridge counterparts at the same seed/architecture; this cross-condition comparison is discovery and not independent confirmation.

Primary transfer gate at step3000: y-axis R²≥0.90, y-axis MSE <0.5× B-D41 commutative-average MSE, x-axis bridge R²≥0.95, and mean original in-support two-action R²≥0.95. Report full-plane transfer separately. If x-axis bridge is not learned, y-axis failure cannot diagnose compositional generalization. If x-axis is learned but y-axis fails, a direction-specific interaction is favored over a reusable action rule. If both architectures transfer, seek a broader functional account; if only one transfers, investigate capacity/optimization and architecture.

No protected/confirmation material. P3.2-E unchanged. All artifacts discovery-only.
