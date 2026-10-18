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


## Entry 005 — Long-range role-contrast specificity surviving numerical-shape correction (18 October 2026)
**Origin:** Codex [frozen benchmark findings at commit `f59a798`](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/f59a798d2a2c76f6cf9ba4e73541cbf5b94fdf1a/research/trajectory_specificity_benchmark_2026-10-18/FINDINGS.md), [complete report](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/f59a798d2a2c76f6cf9ba4e73541cbf5b94fdf1a/research/trajectory_specificity_benchmark_2026-10-18/REPORT.md), [draft technical note](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/f59a798d2a2c76f6cf9ba4e73541cbf5b94fdf1a/research/trajectory_specificity_benchmark_2026-10-18/TECHNICAL_NOTE_DRAFT.md). **Evidence status:** prospectively bounded measurement benchmark (8 author-constructed lexical families, Qwen2.5-7B-Instruct/Llama3.2-3B-Instruct, 4-bit MLX; 176 forward passes, no answers scored, no causal interventions).

**Question:** Are within-model role-reversal contrast trajectories more alike than broad unrelated contrasts once layer-adjacent Gram structure and runtime shape artifacts are controlled?

**Prespecified long-range (step-lag ≥8), period-token, fixed-shape summary:**

| Model | Role / role | Role / unrelated | Gap | Leave-one-family-out gap |
|---|---:|---:|---:|---:|
| Qwen 7B | 0.4477 | 0.3515 | +0.0961 | +0.0653 to +0.1301 |
| Llama 3B | 0.2695 | 0.1171 | +0.1524 | +0.1154 to +0.1809 |

**Interpretation:** A small descriptive *within-model* role-contrast specificity gap exists on this benchmark and its leave-one-family-out sensitivity checks. Leave-one-out ranges are **not** uncertainty intervals or independent replications; comparisons share source sentences, and model architectures share 28 blocks and the MLX runtime. Different prior benchmarks had different contrasts and comparisons; do not collapse them into a contradictory or confirmatory meta-result.

**Technical null:** First-person active/passive states had mathematically identical causal prefixes. Native shape crossings (63 vs 65 token prompt lengths around an execution threshold) produced nonzero first-person contrasts in 8/8 Qwen families that vanished under fixed 128-token-shape capture; Llama was invariant. This revealed one false model-family difference. **At the sentence-period site, shape correction changed role-response angles at most 0.005417 for Qwen and 0 for Llama**, under the 0.05 prespecified materiality threshold. Long-range numerical-null step-Gram coherence was approximately 0.0036 Qwen / 0.0538 Llama. The period-site specificity gap survived.

**Cross-model caution:** Direct long-range role-operation matching remained weak, ~0.167 active and 0.191 passive. Qwen-unrelated → Llama-role ~0.148, showing an operation-specific interpretation is not established. Both models having positive within-model gaps is *not* shared-algorithm evidence.

**Crucial competing explanation:** The broad unrelated comparator changes names and usually verbs, while role reversal keeps a lexical bag fixed and permutes order. It is not a matched control for semantic-role computation. The next high-information contrast should **reorder participants without changing their roles**, using counterbalanced surface constructions and tokenization/length checks. Active/passive alone also alters syntax, so use multiple independent constructions and matched comparisons to avoid trading lexical-order aliasing for a voice/syntax alias. Compare prediction sets for semantic-role, permutation/position and answer-copying accounts **before** fresh inference.

**Link to our original scope:** The result supports investigating developmental *structure* rather than a final hidden-state endpoint. Yet the reported Gram descriptor does not by itself tell us the cognitive stages, within-layer mechanism or multi-site sequence responsible. Our **Linguistic Development Atlas** remains the broader research program: linguistic factors × sites × full depth × within-layer processing × multiple models. A narrow role-preserving control is one discriminating experiment in that atlas, not its replacement.

**Status / action:** Provenance recorded. No new benchmark or inference authorized. Preserve Codex's finished artifact and prospectively discuss any follow-up after reviewing its scientific design.


## Entry 006 — Atlas instrumentation succeeds; competence gate blocks held-out inference (18 October 2026)
**Origin:** Codex's [closed repaired continuation at commit `06e0332`](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/06e03321942c969e5182c40b603721e921601fe7/research/linguistic_development_atlas_continuation_2026-10-18/REPORT.md); [all failed answers](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/06e03321942c969e5182c40b603721e921601fe7/research/linguistic_development_atlas_continuation_2026-10-18/FAILURES.csv). **Status:** technically validated observational development dataset, FAILED frozen linguistic competence gate, zero held-out inference.

**Instrumentation achievement:** Qwen7B/Llama3B fixed-128-token captures pass original-prompt-end logit reproducibility, exact causal-prefix equality, pre-cue zeros, exact repeat, and numerical residual/attention/MLP block reconstruction (residual error ~0.00056 Qwen / 0.00052 Llama, frozen ≤0.002). Earlier Llama native-shape failure remains historical. Full 28-block participant/sentence/question positions and attention/MLP writes were captured and mapped; the study used 69/132 cumulative model calls. These controls license studying observational trajectories on these exposed examples but not semantic mechanism claims.

**Semantic competence:** Qwen7B 28/32, with Agent 8/8, Theme 8/8, Goal 8/8, Order 4/8; **No on every Order question**. Qwen's four failed Order rows were **all gold Yes**, with negative margins roughly −13.2 to −15.8 (looked up in FAILURES.csv); the other four gold No Order items became correct by coincidence with the all-No policy. Llama3B 22/32; Agent 8/8, Theme 7/8, Goal 2/8, Order 5/8; Llama produced Yes for all eight Goal questions and thus was only 2/8 correct. Two development families a01/a02, each 16 answers/model, **0/2 fully competent families for both**. The frozen rule required all 16 per family and at least one passing development family; a03/a04 untouched.

**Stimuli:** examples include “Ada recommended Boris and Clara to Neil” versus “Ada recommended Boris to Clara and Neil.” This changes **argument grouping, arity and local and/to attachment**. It is a linguistically interesting contrast, but a language-only interaction is not automatically a semantic-role effect.

**Developmental observation:** the prespecified interaction I=(across_swap−across_original)−(within_swap−within_original) was zero before the first eligible distinguishing conjunction/preposition and nonzero at eligible later sites. Sentence-period states were invariant to appended questions; this checks masking. Qwen late interaction strength was generally greater for Theme/Goal than fixed-Agent demand. On fixed-Agent questions, gold label Yes was identical across all four cells, yet prompt-end states had nonzero interaction: this rejects only a **state determined exclusively by a fixed answer label**, not answer confidence, distributed answer preparation, syntax/marker copying, nor a generic nonlinear process. Llama interactions occurred even for consistently wrong Goal answers, proving a sizable pattern alone does not establish successful linguistic function.

**What this means for OUR atlas:** the first working attention/MLP-resolved whole-layer map is useful as a **development-only instrument and phenomenon catalogue**, even if frozen held-out confirmation was correctly stopped. Preserve all mistakes, individual-family curves and same-answer comparisons. It is not legitimate to relax the old gate, recast 24/24 as a frozen pass, or inspect its sealed a03/a04 cases without a separate registered design. No automatic retry/approval is inferred.

**High-information next design question:** Why do these binary question interfaces collapse onto fixed response policies? Before ordering more inference, examine the precise prompts, gold semantics, top-token margins and instruction format already recorded. Develop a separately registered diagnostic interface (potentially direct name extraction, forced-choice with counterbalanced options, independent yes/no paraphrases and a matched order task), ensuring output identity can still be crossed with role/syntax; do NOT choose on old held-out results. Distinguish semantic-role preservation from word reordering using syntactically natural alternate placements and counterbalanced templates. Pilot with new development families, and evaluate original holdouts only under an approved new protocol and contamination accounting—prefer genuinely new held-out families.

**Scientific boundary:** failed confirmatory study, reusable observational instrument. No causal use, cross-model algorithm, new law, or paper-ready mechanism is established. Our original long-range linguistic developmental atlas plan continues; Codex's task is to devise rigorous testable designs, not prescribe the exploratory lab's hypotheses.
