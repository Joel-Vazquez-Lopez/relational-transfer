# Factorial relations between developmental responses

**Design note, 17 October 2026. No new model inference.** The earlier four role-swap examples are exposed development material. This document fixes the mathematical object and the main confounds before any factorial model outputs are examined. It does not preregister a held-out test: the incoming research request ended mid-sentence and may add constraints.

## Four cells and what they mean

For each name/verb family, let `R` change which named person is Agent and `V` change active to passive voice while preserving Agent and Patient. An example is:

| Cell | Sentence | Agent | Patient | First name in sentence |
|---|---|---|---|---|
| `00` | Alice praised Beth. | Alice | Beth | Alice |
| `10` | Beth praised Alice. | Beth | Alice | Beth |
| `01` | Beth was praised by Alice. | Alice | Beth | Beth |
| `11` | Alice was praised by Beth. | Beth | Alice | Alice |

Use the same question about the Agent in every cell, and repeat with a Patient question as a separate output condition. A model must answer all eight baselines correctly to enter a causal analysis. A failure is reported, never replaced after inspection.

**Important:** `R` and `V` *both* exchange the surface order of the two names. In `RV`, the original first-name position returns. Thus a parallelogram effect can be caused by name order, token position, or the passive construction. The four cells alone cannot identify a semantic interaction. Additional matched word-order/lexical controls and selective patches are necessary.

## Response geometry

At a predeclared layer `l` and aligned site `s`, write residual states `h00`, `h10`, `h01`, `h11` in one model's native coordinates. Define

```
R_active  = h10 - h00
R_passive = h11 - h01
V_A       = h01 - h00
V_B       = h11 - h10
I         = h11 - h10 - h01 + h00
          = R_passive - R_active = V_B - V_A
```

The equality of the two formulas for `I` is an **algebraic identity**, not two replications. Record `||I||`, the four main-effect norms, and labeled pairwise cosines, together with layerwise updates of these quantities. Preserve magnitudes and directions separately. Near-zero vectors have undefined direction and must be flagged rather than assigned cosine zero.

Use the **final question token** as the one directly comparable site in the initial geometry: the same question is at the end of all four prompts. For sentence sites, align by semantic function (Agent, verb, Patient) and record the actual token offsets and span sizes; absolute positions are not equivalent across voice. Do not silently average different subtoken counts into a claim of token-level equivalence.

The four-cell Gram matrix of centered contrasts, or a relation tensor of labeled layerwise updates, is invariant to a **single shared orthogonal transform** of a model's hidden states. Norm ratios also tolerate a common global scale. Neither is invariant to arbitrary nonlinear reparameterizations, layer-specific changes of basis, or unconstrained depth warping. Across models, compare in the shared label space, not by directly matching Qwen and Llama coordinates.

## What would count as evidence

`I = 0` describes affine additivity of *the measured representation*. It is neither a proof of independent algorithms nor a proof that R and V are semantically disentangled. `I != 0` can be induced by a perfectly separable binary state `(r,v)` passed through the invertible coordinate change `(r,v) -> (r,v,rv)`; the externally independent factors have not changed. Conversely, an additive hidden code `(r,v)` can feed a nonlinear downstream rule `y=rv`. The accompanying [toy calculation](factorial_identifiability.py) checks these counterexamples.

The meaningful next question is predictive: after frozen development-set choices, does the *labeled pattern* of R/V/RV responses predict held-out model-native **selective causal effects** across voice, role, answer question and layer better than generic depth schedule, surface-order controls, and output-only trajectories? An answer effect should be a full-string log-probability margin between the two names, corrected by self-sham and wrong-site interventions. A generic final-token answer transplant does not by itself show a role-specific computation.

## Required comparison and stop conditions

1. Split lexical families into development and untouched evaluation before inspecting factorial outputs. The four previous role-swap families cannot be evaluation data. Freeze names, verbs, templates, labels, question forms, sites, stage mapping and scorer before evaluation.
2. Run the eight-question competence screen for each family and model. Report denominator and failures by model. Do not patch failed families or silently substitute new ones.
3. On development material, measure R/V/RV geometry and output trajectories. Compare with name-order-only, word-order-only, unrelated-sentence, same-prompt self-sham and neighboring-token controls. The existing unrelated control already shows that generic depth structure is strong.
4. Freeze one prediction that distinguishes a semantic relation from surface-order/generic-answer accounts. The decisive evaluation must use new lexical families; ideally a second sentence frame where surface order and semantic-role changes do not have the same factorial relationship. If no frame gives divergent predictions, **stop**: the experiment is uninformative.
5. If competence or control selectivity fails, report a feasibility failure. If the labeled atlas predicts no better than the generic or surface-order baselines, reject the shared-computation claim. A positive result would support a bounded predictive relation among transformations, not a novel circuit or unrestricted equivalence of algorithms.

## Alternate surface frames

The canonical passive places Patient first, making `R` and `V` both swap name order. A grammatical agent-first passive (for example, “By Alice, Beth was praised”) keeps Agent first while changing voice, at the cost of marked topicalization. A fixed-name preamble (“People: Alice is person 1; Beth is person 2”) with event clauses referring to person 1/2 keeps the name tokens fixed, at the cost of a more artificial task. Treat both as *different frames*, never as invisible substitutions for the canonical sentence. Before a held-out run, choose and freeze at least one frame whose semantic-role and surface-order accounts make divergent predictions; record whether the models understand it at baseline. If neither works, stop at a descriptive feasibility report.
