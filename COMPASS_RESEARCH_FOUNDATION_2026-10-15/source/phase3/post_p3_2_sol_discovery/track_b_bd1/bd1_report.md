# B-D1 discovery result — candidate-state V/Y transport fails

**Decision:** The preregistered primary post-block-1 CLS operator **fails**. Stop this V/Y candidate at its frozen rank, stage and action set. The failure is discovery-only and does not alter canonical **P3.2-E**. No relation-changing operation or transformer-native grammar is established.

## Freeze and firewall

The [protocol](bd1_protocol.md), three source scripts and all raw scenes were frozen in `bd1_preflight.json` (SHA-256 `42bc0827e9142d9ea91230d7c59392bfcc263024dc3c046b8bfd2f025aa943d1`) before B-D1 forward passes. The raw scene plan SHA-256 is `566a251bb019a9a4137a6c5d3b979737d694e77c7a59f4fb4057652912c3cbb6`. BASE/V/Y single-action states, fitted maps and all test predictions were then frozen in `bd1_fit_freeze.json` (SHA-256 `2362c2b23589bdd2a3be300571de66fd98e9378d9f88e414d84be72cf09a5381`) while no B-D1 composed outcomes existed. The separate evaluator verified hashes before forwarding V², VY, V²Y and V⁴. Eight existing A/B/C/D × seeds 1701/1703 checkpoints were used, 576 fresh scenes/checkpoint: 288 fit, 96 single-action validation, 96 new-geometry test and 96 held relational-pair test. No model was trained. Protected seeds and confirmatory material were not accessed; canonical artifacts were not modified.

The old P3.2 VY aggregate had been seen in B00 and was explicitly declared in the preflight. No old composed row was used to fit or select B-D1 maps, capacities, metrics or thresholds. The new compositions are prospective **within discovery**, not independent confirmation.

## Frozen primary tests

The table gives pooled eight-checkpoint score RMSE for the fixed oracle query and candidate vector after patching the predicted post-block-1 CLS state through the frozen final LayerNorm. Lower is better. The actual target-state patch reproduced its direct score within 1e-5 in every checkpoint, so the patch and readout were technically valid; at this late stage that agreement is largely a deterministic readout check.

| Target | Primary map | Identity | Additive | Matched MLP | Full-token alternative | Raw-geometry/semantic | Nearest neighbor |
|---|---:|---:|---:|---:|---:|---:|---:|
| V² | **1.083** | 1.051 | 1.054 | 1.075 | 1.153 | 3.652 | 1.662 |
| VY, order Y∘V | **1.470** | 1.545 | 1.493 | 1.657 | 1.603 | 3.552 | 2.346 |
| V²Y, order Y∘V∘V | **1.560** | 1.593 | 1.548 | 1.731 | 1.741 | 3.590 | 2.669 |
| V⁴ closure | **0.495** | 0.000 | 0.123 | 0.446 | 0.675 | 3.539 | 2.148 |

The map's V² score error is **1.031× identity**, violating the ≤0.85 gate. Its V²Y score error is **0.979× identity**, violating the ≤0.85 gate, and **1.007× additive**, violating the ≤0.90 gate. V²Y CLS error is **0.950× identity**, violating the ≤0.85 gate. The full-token alternative is worse in score; it is not a rescue. The privileged raw-input predictor and nearest neighbor are much worse, which rules out those particular baselines as explanations for the modest map gains, but does not turn a weak map into an operation. A magnitude-matched random-direction patch even slightly beats the primary map on V² score (1.070 versus 1.083); the wrong-action patch is worse (1.718).

Across checkpoint-specific V²Y score ratios, the unweighted mean is 0.943; a descriptive 10,000-resample checkpoint bootstrap gives a 95% percentile interval of **[0.891, 0.997]**. This small average advantage does not meet the frozen magnitude, repetition, identity-split or cross-seed criteria. The pooled RMSE ratio above weights larger-error checkpoints more heavily.

## Where the algebra breaks

1. **Single step.** On the predeclared non-held validation scenes, the V map's CLS RMSE is **0.982× identity**; the Y map reaches **0.834× identity**. This post-outcome audit merely characterizes singles already acquired before the fit freeze (`bd1_posthoc_single_step.json`). V supplies little useful state transport beyond leaving its input alone, while Y has a modest one-step signal.
2. **Repeated action.** Iterating V twice worsens score prediction against identity. The actual V⁴ input and state return exactly to BASE, but four map applications drift to CLS RMSE about 0.32 per checkpoint and pooled score RMSE 0.495. Additive drift has score RMSE 0.123. The frozen map does not respect the simple C4 closure even approximately at the functional level.
3. **Mixed composition.** VY is modestly better than identity (score RMSE ratio 0.951), but V²Y is only 0.979 and loses to additive. The gain weakens as the V step repeats, rather than supporting a reusable composition law.
4. **Commutation.** The two predicted orders differ by 0.127× the actual VY CLS displacement and 0.178× the actual V²Y displacement, satisfying the commutator gate. This is insufficient: near-identity maps can commute while predicting the wrong transformed state and score. Commutation is a necessary algebra check, not an operation result by itself.
5. **Functional and causal readout.** Every predicted state was actually patched into the frozen final candidate readout; prediction errors above are downstream score errors, not merely geometric distances. The actual-state patch exactly reproduces the actual score. Because post-block-1 CLS is immediately before LayerNorm, these patches establish readout consequences rather than localizing a deeper relational circuit.
6. **Scene and identity structure.** On V²Y, the score error is 0.917× identity for new geometry but **1.016× identity** for held relational pairs. Only **1/8** checkpoints beats identity on both V² and V²Y; the frozen requirement was at least 6/8 with replication in each seed.
7. **Cross-seed transport.** BASE-only Procrustes alignment transferred the seed-1701 maps into seed-1703 coordinates. V²Y score-error ratios to identity were **1.011, 0.986, 1.260, 1.083** in A/B/C/D. None meets the ≤0.90 gate, and C also misses the commutator bound. Independent same-family fits do not establish literal seed transport.

The preregistered `primary_candidate_survives` gate is **false**. The first decisive break is weak V single-step transport, followed by repeated-action and exact-closure failure. The small commutator is a false comfort. This kills the specified global, low-rank, post-block-1 V/Y transport law. It does **not** establish that all compact operations are absent in this task or in transformers generally. Any alternative stage, capacity, action family or conditional law must be a separately named discovery candidate with fresh firewall; it cannot repair B-D1 retrospectively.

## Provenance and limits

The machine-readable decision and every metric are in `bd1_results.json` and `bd1_metrics.csv`. `bd1_acquisition_manifest.json` hashes all composed outcomes; `bd1_final_manifest.json` hashes the frozen fit and results. The post-outcome single-step script and output are explicitly labeled descriptive. Eight checkpoints rather than all twelve were predeclared to keep this first composition test small. The raw-input comparator had 2,736 coefficients and privileged scene/meaning access; its poor performance does not prove semantic information is irrelevant. All B-D1 claims are limited to these new synthetic scenes and permitted discovery checkpoints.
