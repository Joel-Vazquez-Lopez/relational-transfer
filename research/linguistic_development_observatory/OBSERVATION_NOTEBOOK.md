# Observation notebook — informal science, explicit uncertainty

This notebook intentionally accepts messy, partial and contradictory observations. It is a place to ask questions before inventing explanations. Do not convert an exploratory entry into confirmed evidence merely by adding a plot.

## Entry 001 — Why the full curves attracted our attention
**Origin:** Qwen7B/Llama3B four-example role-swap multi-site layer plots. **Evidence status:** visual exploratory observation.

- The subject and object contrast magnitudes both broadly decline across much of the network, with different late-layer rebounds.
- An unchanged verb's hidden representation responds to the name/role manipulation, prominently in earlier layers.
- The question-token representation differentiates later, with model-dependent pace and strength.
- The *joint order of those changes* looks related in the two models, although scale and timing differ; the representations themselves do not have a shared PCA basis.
- **Question:** Is this joint site-by-depth rhythm a cross-model property of role processing, of any lexical perturbation, or of the shared Transformer architecture?

**Rivals:** token substitution/name movement; generic depth schedule; token-site causal mask; output preparation; non-comparable token offsets.

**What would change our interpretation:** quantitative four-site shape comparison with word/name changes that retain roles, role changes that control position, question-only changes, and arbitrary matched-sentence changes. Use matched families and honest uncertainty.

## Entry 002 — The single-path metric was not specific
**Origin:** exposed saved-vector reanalysis, Issue #34.

Unrelated sentence changes produced greater step-Gram agreement than controlled role swaps; the last block was often the largest movement. **This is a genuine warning for Path-Cosine, not proof that every multivariate observation is generic.**

**Question:** Can we isolate conditional differences and cross-site relationships from a nontrivial common background without subtracting away signal?

## Entry 003 — Outputs can imitate internal-processing signatures
**Origin:** researcher-reported cross-question geometry pilot.

Name-axis removal retained impressive directional alignment, but an explicit copying model recreated it; Qwen task-conditioned directions tracked an output-based predictor in all 16 described comparisons. **Question:** what can we still predict when the answer, first-mentioned name, source tokens and voice are crossed?

## Entry template (copy and fill)

### Entry N — short title
- Date / investigator / model and exact revision:
- Question / reason for looking:
- Source artifact / hash / script / output file:
- Linguistic contrast and expected invariants:
- Positions/sites/layers and token alignments:
- What was directly seen (counts, units, uncertainty):
- What interpretation is tempting:
- Alternative explanations and what each predicts:
- The most surprising *negative* case:
- Next cheap check on existing data:
- What data are already exposed vs truly fresh:
- Status: impression / descriptive / controlled / causally tested / independently replicated

## Idea parking lot

- Are shape patterns similar because all affected tokens transition from lexical embedding to contextualization and then answer readout?
- Does a site **lead–lag graph** describe development better than four independent norm curves?
- Do voice-preserving paraphrases share cross-site lag relationships while role reversal changes one link?
- Is the generic layer schedule itself informative across models, before any semantics is considered?
- Would head/MLP contribution decomposition explain late rebounds or show cancellations hidden by the residual?
- How would invented names, rare names and multi-token names alter the entire path without changing roles?
- Can we build a known-mechanism toy model where nominally similar multi-site curves conceal incompatible algorithms?
