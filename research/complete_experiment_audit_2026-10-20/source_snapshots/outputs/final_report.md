# Relation-endpoint communication: engineering/scientific pilot report

**Date:** 2026-09-08  
**Recommendation:** **MODIFY** before any locked confirmatory experiment  
**Highest gate reached:** Pilot 0 / Gate A (failed)

This was a viability pilot, not a test of a “transformer-native grammar.” The
separate report referenced in the brief was not included, so the supplied brief
was treated as the specification.

## 1. Environment validation

The implemented world has three objects drawn from six color/shape identities,
four continuous 2D observations, ordered source and target roles expressed only
through trajectories, approach versus move-away motion, and an independently
moving distractor. Object slots are shuffled. The speaker receives one source
scene. The listener receives only a four-symbol hard message (vocabulary eight)
and four independently rendered candidate scenes. Candidates are the correct
meaning, endpoint reversal, event reversal, and both reversals.

Six directed endpoint pairs (three undirected identity pairs in both directions)
are compositionally held out. This symmetric construction ensures reversal foils
have the same train/held-out status as the answer.

Final automated checks:

| Check | Result |
|---|---:|
| Duplicate episodes | 0 in every split |
| Candidate positions | exactly balanced globally and within meaning |
| Object slots | randomized; largest source/target slot-count spread 107/12,288 |
| Meaning balance | exact within every split |
| Distractor identity | balanced within meaning |
| Ambiguous trajectories | 0 |
| Source/target ordering cue | no fixed slot; reversal foils symmetric |
| Held-out semantic overlap | 0 (48 train, 12 held out) |
| Mean nuisance coordinate by answer | range below 0.025 in every split |

Three invalid engineering benchmarks were retained rather than overwritten. They
revealed, in order: asymmetric holdout membership leakage, an event/answer-position
correlation hidden by global balance, and a meaning/distractor correlation. Each
was repaired with a minimal balancing change and promoted into the automated gate.
The clean 1,000-step benchmark then produced no-message accuracy 24.6% and shuffled
accuracy 28.5%, consistent with chance (25%).

## 2. Communication results

Only seed 1701 was trained because it failed Gate A; the requested three additional
seeds were therefore not launched.

| Run | Steps | Train | Validation | IID test | Held-out composition |
|---|---:|---:|---:|---:|---:|
| Clean benchmark, seed 1701 | 1,000 | 38.0% | 36.0% | 37.0% | 24.6% |
| Pilot 0, seed 1701 | 3,000 | 53.8% | 47.4% | 49.6% | 25.7% |
| Bounded extension, seed 1701 | 8,000 | 83.2% | 48.5% | 48.7% | 44.8% |

Final communication controls:

| Condition | IID accuracy |
|---|---:|
| Normal message | 48.7% |
| No message | 25.6% |
| Shuffled message | 4.3% |
| Candidate-order permutation | 48.7% |
| Object-slot permutation | 48.3% |

Thus the listener does depend on the message, candidate ordering is not being
used, and object-slot permutation does not impair performance. The very low
shuffled score is an anti-correlation, not stronger evidence than a chance-level
control. Most importantly, normal IID accuracy is only 48.7%, so communication is
not useful enough for downstream mechanistic interpretation.

The model has 118,208 parameters. The 1,000-step benchmark took 14.2 seconds
(70.3 steps/s); the 8,000-step run took 113.8 seconds (70.3 steps/s), both on CPU
with PyTorch 2.8.0. MLX/Metal was unavailable to the headless process. A simple
update-count proxy is 945.7 million parameter updates for the 8,000-step run;
FLOPs were not measured and are not inferred from that proxy.

## 3. Protocol analysis

The final IID protocol used only symbols 1 and 3 despite an eight-symbol channel:
53.54% and 46.46% of token occurrences, respectively. There were eight unique
messages and 1.94 bits of empirical whole-message entropy. The leading messages
were `1111` (1,285/3,072), `3333` (953), `3133` (482), and `1131` (252).

Meaning-to-message tables are saved verbatim in the run metrics. They show a
coarse, low-cardinality partition rather than a reliable one-message-per-meaning
code. These are descriptive relationships only; they are not mechanistic evidence.

Gate B was not evaluated. With one unsuccessful seed there is no basis for a
cross-seed diversity claim, and optimal symbol relabeling was therefore not used
to manufacture one.

## 4. Mechanistic findings

Pilot 1 was not run because Gate A failed. Consequently there is no activation-
patching, head/MLP necessity, path-level, or probe result to interpret. The code
contains listener head and MLP ablation hooks, but using them on this overfit model
would answer a question about an unreliable policy rather than the intended task.

Evidence classification:

- **Decodability/correlation:** coarse message/meaning correlations exist.
- **Representational evidence:** none established.
- **Causal necessity:** the whole message is necessary at the behavioral level;
  no internal component necessity claim was tested.
- **Causal sufficiency:** none established.
- **Mechanistic implementation:** none established.

## 5. Hybrid intervention results

Selective source, target, event, source+target, and nuisance-preserving hybrid
interventions were not run. Their prerequisite—a successful, generalizing model—
was absent. There are therefore no observed hybrid outcomes to compare with the
pre-registered predictions (hybrid selection rather than donor-answer copying).

## 6. Competing explanations

- **Candidate-position shortcut:** falsified by conditional balancing and exact
  candidate-order permutation invariance.
- **Object-slot shortcut:** not supported; object permutation preserved accuracy.
- **No-message candidate prior:** falsified in the clean environment (25.6%).
- **Informative communication:** supported behaviorally, since normal beats no-
  message and shuffled controls.
- **Whole-message lookup, fixed-position decoding, pooling, endpoint-selective
  representation, or context-dependent selection:** underdetermined. Gate A
  failure makes discrimination among them scientifically premature.
- **Finite-geometry memorization/overfitting:** strongly supported by the 34.7-point
  train–IID gap after 8,000 steps.

## 7. Failures

1. The first generator inverted the approach/away velocity sign; validation caught
   all trajectories as ambiguous/inconsistent before training.
2. Three successive dataset audits exposed asymmetric holdout, conditional answer-
   position, and conditional distractor leakage. Failed checkpoints and metrics are
   preserved.
3. The clean 3,000-step run underfit.
4. The single bounded 8,000-step extension fit training scenes but did not improve
   IID validation/test performance enough. No further world redesign or sweep was
   attempted.
5. Protocol diversity and mechanistic pilots were correctly skipped.

## 8. Candidate causal program H

None is justified. The correct Decision Gate C status is **PILOT INVALID for
mechanistic interpretation**, specifically because optimization on a fixed finite
geometry set produced substantial overfitting. This is not evidence that the
underlying mechanistic hypothesis is false.

## 9. Recommendation

**MODIFY.** Do not train the six confirmation seeds and do not freeze a causal
program. The communication environment now passes the tested leakage checks, but
the learning procedure does not yield a sufficiently reliable IID policy.

The smallest next change is to sample fresh nuisance geometry online during
training while preserving the exact semantic balance, candidate construction,
architecture, channel, optimizer, gate thresholds, and held-out pairs. This targets
the diagnosed finite-geometry memorization without optimizing for interpretability.
Permit exactly one predeclared run; if IID remains below 75%, stop the project or
revisit the task specification openly.

## 10. Exact next experiment

There is no GO experiment to freeze. For the proposed MODIFY diagnostic:

1. Freeze code version, seed 1701, all model/channel hyperparameters, the symmetric
   held-out pairs, and the existing Gate A thresholds.
2. Replace the fixed 12,288-scene training tensor with deterministic online scene
   generation keyed by `(seed, step, batch_index)`; do not change validation/test.
3. Train once for 8,000 steps and evaluate the same controls and untouched splits.
4. Proceed to three diversity seeds only if IID >75%, no-message and shuffled <40%,
   and held-out composition >40%.
5. If it passes, pre-register the exact patch sites, hybrid candidate construction,
   and outcome statistics before examining Pilot 1 activations.

All source, configurations, checkpoints, raw metrics, failed runs, and the research
log are retained in the project tree. The plot accompanying this report summarizes
the final accuracy/control pattern.

