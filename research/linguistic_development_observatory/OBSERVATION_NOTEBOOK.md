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


## Entry 004 — Question-conditioned developmental response, after Codex's final audit (17 October 2026)
**Origin:** [Codex complete report, frozen commit `7cfb406`](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/7cfb40634baf9783f7a1c0a057cde8feb82dbed9/research/trajectory_relation_audit_2026-10-17/REPORT.md), [query summary JSON](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/7cfb40634baf9783f7a1c0a057cde8feb82dbed9/research/trajectory_relation_audit_2026-10-17/QUERY_SUMMARY.json), [Issue #33 outcome](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33#issuecomment-6240873041). **Evidence status:** one-model bounded descriptive result with retained multi-model failures; no identified mechanism.

**Contrast:** Within each model, compare active/passive role-reversal response vectors, consistently oriented by which person is Agent; compare the same construction families under “who did the action?” vs “who was mentioned first?” using layerwise cosine. This captures response *direction*, unlike earlier response-magnitude plots. Eight new lexical families × four sentence cells × two questions × four models produced 256 recorded responses; fixed late summary over final quarter of blocks.

- **Qwen7B:** late Agent cosine **+0.099**, first-mentioned cosine **−0.911**, difference **+1.010**; difference positive for **8/8 families**. Correct 24/32 Agent, 32/32 first-mentioned; fully competent in only **3/8 families**. Passed model-specific descriptive threshold.
- **Llama3B:** late **−0.039 / −0.907**, difference **+0.868** and 8/8 positive, **but FAILED the frozen technical invariance control**: event-prefix states varied across later questions by up to 0.003827 in relative norm versus a 0.001 tolerance. **Do not call this clean replication.**
- **Qwen1.5B:** difference **+0.485** but only **1/8 competent families**; failed competence gate.
- **Qwen0.5B diagnostic:** difference **−0.132**, **0/8 competent families**.
- **Whole three-primary-model rule failed.** No further adaptive inference was done.
- Direct output-axis removal does not abolish the observed Qwen7B difference, but a **simple answer-response baseline matched all 16 final alignment signs**; a programmed surface-keyed answer copier reproduced key geometry and axis-removal signatures. It does not reproduce every measured magnitude, but defeats any unique-semantic-computation interpretation from signs alone.

**What we find intriguing:** Same source transformations can have different *late developmental geometric relationships* depending on retrieval demand; early-to-late evolution matters. The earlier shape-based patterns were a real motivation for asking about the relation between *contrasts*, not sufficient evidence for common linguistic machinery.

**Do not conclude:** semantic-role computation was localized; Qwen and Llama replicate a universal law; late convergence proves reusable role identity; removing one axis eliminates distributed answer code; all plotted 8 families are independent corpus-validity confirmation.

**Ground-development ideas, not assignments to Codex:**
1. Separate **linguistic operation**, **question formulation**, **output name** and **surface word order** in a crossed conceptual model; compute which contrasts are identifiable before paying for more traces.
2. Investigate whether there are **site-by-depth developmental phases** under the question-conditioned contrast, not just one last-quarter cosine. Inspect full family traces and absolute magnitudes alongside signed direction; quantify how answer correctness relates to motif differences without conditioning away errors.
3. Ask which patterns a **surface-keyed answer copier** predicts *across full layers and multiple sites*, not merely terminal signs; explore toy systems first.
4. Treat Llama's pre-question numerical discrepancy as an unresolved technical confound. Never retroactively relax its threshold; debugging requires separate status.
5. Preserve our original linguistic north star. Codex can later receive a narrow independently specified test after our exploratory inspection, with no automatic experiment expansion.

**Decision:** Add to observational evidence ledger. No claim of cross-model mechanism; no new model runs triggered by the archival update.
