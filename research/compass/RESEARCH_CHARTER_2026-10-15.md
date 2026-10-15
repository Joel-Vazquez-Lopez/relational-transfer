# Compass — independent structural reuse investigation

**Status:** hypothesis development; no new pretrained-model causal result established.
**Date:** 2026-10-15
**Scope:** Compass only. Do not modify frozen experiments, other workstreams, or sealed data.

## Central question
Can we distinguish a Transformer that naturally computes a relational binding before seeing a question and reuses that computed result from a Transformer that carries source information forward and constructs the requested binding only after receiving the question?

This is a distinction about **when and where a relational computation occurs**, not merely whether the model uses reusable parameters/circuit components.

## Competing accounts
- **H-R (pre-question relational computation):** context-prefix processing computes role-resolved event bindings; later questions selectively retrieve those bindings.
- **H-Q (question-conditioned reconstruction):** context-prefix processing retains information sufficient for answering, while role-resolved binding is computed independently after the question.
- **H-M (hybrid):** some precomputation occurs but later questions may reconstruct or correct bindings.

All three may exhibit perfect answers and respond similarly to coarse prefix activation interchange. Successful probing or whole-prefix patching alone does not discriminate.

## Next experiment: pre-question causal boundary

### Phase 0 — validity and negative control
Build two transparent computational controls with matched outputs: an explicit prefix binder and an explicit post-question reconstructor. Ensure the proposed mechanism test labels these distinctly; otherwise report non-identifiability, not success. Random or nonrelational pre-question representations should be additional controls.

### Phase 1 — audit before pretrained inference
Reuse historical raw predictions only as development diagnostics. Identify documented construction, name-position, grading, quantization, and prompt confounds. Do not treat repeatedly explored examples as independent confirmation.

### Phase 2 — controlled small Transformer
Train or use transparent controlled models with matched task competence, held-out participant combinations, counterbalanced linear position, and delayed questions. Compare pre-question role-resolved causal mediation to post-question reconstruction. Budget and training requests require coordinator/researcher authorization.

### Phase 3 — pretrained model test (approval for substantial compute)
Use a frozen model with causal attention and events placed before questions. Contrast a question-independent role-resolved intervention with nonrelational information-carrying controls, matched-size noise, and downstream question-specific interventions. Evaluate under held-out syntactic transformations, entity positions, multiple events, and both answer roles. Use localization, patching and path-based causal tests with multiple corruption choices and metrics; assess stability.

## Main decision rule
An outcome favors H-R only if (i) pre-question native processing has a role-resolved causal contribution, (ii) multiple held-out questions rely on that same earlier computation, (iii) targeted edits produce selective event-specific effects, and (iv) tests reject the transparent H-Q negative control. A successful intervention alone is insufficient.

If role resolution first appears question-dependently and the H-R criterion fails, report support for H-Q *at tested resolution*. If evidence is mixed, report H-M or non-identifiability.

## Threats to validity
- Decoder/probe flexibility can induce apparently structured variables.
- Activation patching may transport raw sentence information rather than an already-computed binding.
- Across-question circuit reuse differs from reuse of a computed intermediate result.
- Causal masking guarantees prefix representations cannot depend on later questions, **not** that those representations already contain computed bindings.
- Query answer format, lexical position heuristics, repeated name pairs, artificial intervention off-manifold effects, and model/back-end differences.
- Shared natural circuit cannot generally be uniquely identified from finite interventions.

## Initial status and deliverable
1. Repo access confirmed; only this branch is writable by Compass under collaboration convention.
2. Prior local research archive and previous Compass reuse-lab outputs are the evidence base, but neither new model inference nor a new causal proof is claimed here.
3. The next executable deliverable is a discriminative test suite with transparent positive/negative controls and a preregistered decision table.

## Reporting contract
Every result must state: model/checkpoint, prompt or data version, seeds, backend, exact count of independent event families, intervention site, raw outputs, uncertainty, competing explanations, negative results, and novelty status. No claim of mechanistic discovery until evidence is replicated on untouched data.
