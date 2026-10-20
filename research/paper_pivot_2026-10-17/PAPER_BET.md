# Paper pivot: pre-inference witness audits for activation-patching claims

17 October 2026. **REJECTED AS THE LEAD PAPER BET after primary-method review.** Retained as the reasoning record; see [PAPER_BET_V2.md](PAPER_BET_V2.md). No new model experiment was run.

**Why rejected:** Gur-Arieh, Geva and Geiger's [Mixing Mechanisms, §3.2](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf) already constructs counterfactual activation patches so positional, lexical and reflexive mechanisms predict distinct tokens. The proposed generic witness search would need a major independent algorithmic/validity contribution beyond that and beyond causal abstraction. Our present two-event certificate is ordinary finite enumeration. We should not make it the main paper plan.

## The bet

**Working title:** *Before You Patch: Finding Discriminating Witnesses for Mechanistic Claims about Language Models*.

**Problem:** An activation patch can clearly change a model's answer yet leave multiple computational explanations observationally equivalent. The local event study is an example: a semantic event-address theory and a competent syntax/content compound-key copier predict identical answers in all eight order × marker × role cells, even though the event-address theory differs from a simple ordinal-index rival in four cells. A positive patch would be causal evidence of answer-source steering but not of the named event-selector algorithm.

**Potential contribution:** An executable pre-inference audit that takes (i) competing *explicit* causal/predictive programs, (ii) permitted input and internal interventions, (iii) constraints on linguistic and annotation validity, and (iv) observable endpoints, then returns either a minimal separating witness or a finite-scope equivalence certificate. It must expose the witness and the exact rival disagreement, not merely advise researchers to use controls. The current [event certificate](../decisive_witness_gate_2026-10-17/REPORT.md) is a small motivating case, not sufficient validation of a general tool.

**Scientific claim we might eventually test:** Such audits detect materially non-discriminating patch designs *before model runs* and find valid discriminating cases where ordinary human-designed controls do not. The paper must demonstrate this on multiple independent case studies; otherwise this is familiar experimental-design advice packaged as code.

## Why this pivot, not more event patching

The current Qwen 7B result is narrow and closely related to causal binding/index steering in [Dai et al. 2024](https://aclanthology.org/2024.emnlp-main.967/) and [Dai et al. 2026](https://aclanthology.org/2026.acl-long.2194/). The latter also analyzes a real-text relation dataset, so “we use natural language” is not enough. Multi-event extraction, event confusion and trigger provision are studied in [TabEAE](https://aclanthology.org/2023.acl-long.701/), [EAESR](https://aclanthology.org/2024.findings-acl.328/) and [Shaar et al. 2025](https://aclanthology.org/2025.tacl-1.71/). Our one-reviewer local scoring panel also failed the two-role baseline.

A quick *heuristic*, source-only census of the already local MAVEN-Arg archive, excluding documents in the five prior local panels, yielded **66 TRAIN + 18 VALID candidate documents** under a stringent filter: two same-type Agent/Patient events, four distinct annotated entities and surface strings, each selected role mention within 180 characters of its trigger, triggers within 750 characters. This is an upper bound before human semantic review. Looser 600-character mention distance gave 95 + 22 candidate documents, still before review. The dominant families were Military_operation, Attack and Conquering; some previously reviewed Attack and scoring annotations were unsuitable. This does not establish enough clean holdout cases for a strong natural-event mechanism paper.

## What would make a solid paper

1. **Method:** Precise hypothesis language and intervention semantics, a finite witness search or constraint solver, validity constraints, and explicit `NO WITNESS IN THIS TEST FAMILY` output. No claim of universal nonidentifiability from a finite search.
2. **Independent cases:** At least three nontrivial mechanisms drawn from different published task families, with the rival programs and permitted patch operations reconstructed from primary methods rather than invented strawmen. Include a case where the tool *finds* a feasible witness and a case where it proves equivalence under a declared finite family.
3. **Empirical utility:** On predeclared witnesses, run a bounded model test only if the witness is valid and hypotheses predict different outcomes; show that the audit changes a scientific conclusion or prevents a misleading one. Record annotation validity and all failed witnesses. The old Qwen event result serves as one retrospective example, not the whole validation.
4. **Baseline comparison:** Human-designed standard controls, naive random counterfactual generation, and a simple enumeration baseline. The method must add value beyond hand enumeration or unit testing.
5. **Released artifact:** Reproducible code, rival specifications, witness matrices, validity decisions and raw outputs where rights permit.

## Closest prior work and novelty risk

- [Causal abstraction](https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf) already formalizes interchange interventions and when a high-level account maps onto a model.
- [Zhang and Nanda](https://arxiv.org/abs/2309.16042) already document how patch metrics and corruption methods change conclusions.
- [The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510) provides a different but substantial analysis of patching estimands and interactions.
- [Mixing Mechanisms](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf) already fits competing positional, lexical and reflexive retrieval mechanisms using conflicting predictions.
- Mechanistic interpretability benchmarks and automated hypothesis-generation tools exist. An implementation that just enumerates two toy policies and prints a disagreement table is **not a paper**. The publishable difference would have to be a practical, validity-aware witness search with clear improvements over these methods across independent real cases. This gap is provisional pending full-method review.

## Seven-day feasibility gate, before new model runs

1. Reproduce the local event equivalence certificate and encode one *independent* published example as rival programs with exact intervention access. Find a real, source-valid witness there, or identify why none is possible.
2. Complete a full-method prior-art matrix for causal abstraction, automated interpretability experiment design, Mixing Mechanisms, and activation-patching best practices. State the exact algorithmic operation the proposed tool adds.
3. Build a minimal witness-search prototype and compare it with plain enumeration on at least two nontrivial tasks. If the prototype adds no scientific or computational value, **kill this paper bet** without new LLM inference.
4. Only if those gates pass, draft a full preregistered empirical evaluation and seek a second reviewer for source validity. The previous single-experiment authorization is exhausted; any new model run needs a separate bounded plan and authorization.

## Decision discipline

**GO toward a paper** only if the method produces a non-obvious valid witness or a useful equivalence certificate in independent cases and survives the prior-art comparison. **PIVOT AGAIN** if it merely restates generic nonidentifiability, if strong rivals are underspecified, or if validity constraints make witnesses artificial. A strong 15-credit thesis can include this feasibility work even if the publication gate fails, but a paper requires the extra contribution above.
