# Phase 1 scientific report — 14 October 2026

## Question and design

The earlier ten-entity rewire changed common partners and three-edge paths
together. This prospective 12-entity, eight-family 2×2 experiment varied
common partners (one or two) and simple three-edge paths (three or four)
separately. All graphs were degree four, had 24 training pairs, and matched
total and focal-vertex triangle counts within family. Four-cycle counts varied
by at most one. Three paired training seeds gave 96 models in total. The
320-rendering focal test and analysis rules were frozen before factorial
training. All 96 runs completed without technical failures.

## Prespecified outcome

The common-partner contrast (two minus one, averaged across path levels) was
**−9.94 percentage points** in focal binary role accuracy. The exact
family-level two-sided p-value was 0.03125; the prespecified Holm-adjusted
p-value for two primary contrasts was **0.0625**. The 95% family t interval
was **−20.20 to +0.32 points**. It therefore did **not** meet the frozen
practical-success rule (adjusted p≤0.05 and magnitude≥10 points).

The three-edge-path contrast (four minus three, averaged across partner
levels) was **+3.96 points**, adjusted p=0.1953, 95% interval **−2.27 to
+10.20 points**. The interaction was +7.57 points, exploratory p=0.1484.
Minimum familiar four-choice accuracy was 99.53%, so ordinary task-learning
failure does not explain the focal result.

The prespecified second scene-rendering suite gave similar descriptive
effects: common-partner −9.37 points (unadjusted exact p=0.0547), path
+3.57 points (p=0.2344). These are robustness diagnostics, not a second
independent set of graph families.

## What the result does and does not show

The direction of the common-partner effect agrees with the earlier
confounded rewire, and the effect is stable across two renderings. However,
two families show large losses while several families are near ceiling;
the predeclared confirmation criterion failed. The result supports a
*possible family-dependent structural sensitivity*, not a confirmed general
rule or a unique causal role for common neighbors.

Exploratory checks found that simple four-edge-path counts also changed
systematically with the common-partner contrast (mean change −3.69 paths
across families). This graph property was not matched in Phase 1. In the two
families with the largest losses, the low-scoring condition was near chance
across both actor directions, events, and answer positions, while familiar
accuracy stayed high. That rules out a single answer-position artifact as
the whole explanation but does not identify the model's mechanism.

## Decision

**Do not advance to Phase 2 on this evidence alone.** Preserve this
near-miss and the earlier negative results. A graph-only search found eight
fresh, nonisomorphic 12-entity anchor classes in which the same 2×2 cells can
also match four-edge paths exactly, using at most two degree-preserving
switches per condition. This makes a tighter independent Phase 1 follow-up
feasible. It must be prospectively frozen and tested on new graphs; the
existing families remain exploratory and must not be used to tune or select
the next result.
