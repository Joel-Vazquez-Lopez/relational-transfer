# COMPUT-027 — Independent Qwen7B/Llama3B frozen factorial error-provenance audit

**17 Oct 2026 | Historical pretrained model output RE-ANALYSIS (no new LLM calls) | Four synthetic lexical families.**

## Collision and permissions

I checked current #31 scientific gate and #33 wrong-event/frontier handoffs, [existing COMPUT-026](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/comput-falsification/research/comput/REPORT_026_SOURCE_PROVENANCE_IDENTIFIABILITY.md) (the three-event model-free source certificate has **already been done**), and the independently owned #34 Qwen/Llama causal-developmental frozen factorial report. Rather than duplicate another 3-event or model-run panel, COMPUT-027 independently audits the actual **already saved model generations** on one single-event grammar: lexical position, role, voice, lexical-family bias and cross-model *errors on matched semantic items*.

**Source-frozen original saved raw outputs, read in full through connected GitHub:**
- Qwen7B: `research/causal_developmental_alignment_2026-10-17/FACTORIAL_GATE_qwen7b.json`, Git blob `6a573e12340dedc6d935eb890809508f1f48b5a6`.
- Llama3B: `research/causal_developmental_alignment_2026-10-17/FACTORIAL_GATE_llama3b.json`, Git blob `73d8be699d53bbba41855d970d4c69918a76b91c`.
- Frozen original design `FACTORIAL_GATE_LOCK.json` Git blob `fe3b0e8ed0ba0b8ac99147f4faa899c66a83a2ca`; summary `FACTORIAL_GATE_SUMMARY.json` blob `78583c15adb46019eeb90d08778be76c6edbbd90`.

**Each model has 64 recorded conditions, but 16 repeated active-frame prompt records**, yielding **48 distinct semantic prompts**. Verify that duplicated active-frame outputs and score margins are exactly identical before counting distinct conditions; never call 64 independent prompts. Different families of model can have different chat template wrappers, so compare **semantic-condition keys** not literal full chat strings. The four lexical families are two `thanked`, two `invited`, with development/evaluation labels but **not an independent natural-language source-heldout evaluation**.

## New independent results from original stored generations

| Diagnostic (48 distinct cells/model) | Qwen 7B | Llama 3B |
|---|---:|---:|
| Correct model generations | **43/48** | **39/48** |
| When gold was the **first-mentioned** name | **21/24** | **24/24** |
| When gold was the **second-mentioned** name | **22/24** | **15/24** |
| Active voice | **16/16** | **12/16** |
| Canonical passive | **12/16** | **15/16** |
| Agent-first passive cleft | **15/16** | **12/16** |
| Original verb `thanked` | **19/24** | **20/24** |
| Original verb `invited` | **24/24** | **19/24** |
| Wrong responses choosing **first-mentioned** name | **2/5** | **9/9** |
| Wrong responses choosing **original family name A** | **5/5** | **7/9** |
| Wrong responses choosing the other role's filler | **5/5** | **9/9** |
| Grammar-aware *input-only surface parser* | **48/48** | **48/48** |

Across the **same 48 underlying semantic source-condition cells**, **34** cells are correct for both, **5** wrong only for Qwen, **9** wrong only for Llama and **0** wrong for both. These are dependent tiny-set counts and **not significant causal mechanism evidence**; the two models also use different chat wrappers and checkpoint families.

### Interpretation

- **Llama positional interference signature:** all nine *distinct* errors output the sentence's first-mentioned person even when the gold was second, while the 24 first-gold cases are all correct. This is compatible with a fall-back first-name policy or role-question default under lexical/format uncertainty. **An always-first raw-text rule itself only achieves 24/48 = 50%**, so it does *not* explain the model's 39/48 global competence. The hypothesis is a **selective** fallback, not that Llama always chooses first. Within-family correlated counts cannot show a population bias, and no internal causal computation is located.
- **Qwen condition interaction:** every error occurs with passive **`thanked`**; the two `invited` families and all 16 distinct actives have zero errors. **5/5 wrong results prefer original family A name** (a design metadata identity, *not a model-visible tag*); this includes both first- and second-mention errors and warns against reducing Qwen's issue to first-name order. Verb, lexical name pairs and tokenizer differences are **confounded**, so 'thanked causes failures' is not identified. All five failures are candidate-name predictions, not format-invalid outputs.
- **Strong negative control:** the literal visible grammar (`Agent Verb Patient`, `Patient was Verb by Agent`, `It was by Agent that Patient was Verb`) lets a simple template parser answer *all* 48 unique conditions without hidden labels. This is **not** a realistic arbitrary-language parser; it shows the benchmark's simple syntax does not force abstract event-binding representations.
- **Critical inferential ceiling:** two-person single-event examples cannot measure wrong-EVENT sourcing; every wrong in-vocabulary response is necessarily the other ROLE, so calling 14/14 'binding confusion' is tautological. These historical model failures cannot distinguish retrieval from prompt-interface or ordinary syntax learning. The `0/48` overlap of error conditions is descriptive only.

## Reproducibility and boundaries

Portable Python stdlib `comput027_frozen_factorial_audit.py` takes the TWO ORIGINAL saved owner JSON logs locally and scores original source grammar, active-row equivalence, first-mentioned diagnostics, cross-model semantic cells, score-vs-generation. It exports aggregates and **no prompt strings/raw sentences**. `test_comput027.py` has **11/11 passed** local synthetic fixture/integrity tests including corrupted duplicates, wrong gold, and grammar parsing. Original raw source **aggregate analysis actually executed with an independent in-session GitHub-connected JS reader**, not yet re-executed by the portable local Python CLI on the private raw JSON files; do not conflate these execution statuses.

No model inference, paid tokens, checkpoint download, hidden/test source access, novel human gold, sealed experiments, causal patch, copied other investigator's branch, approval change or paper/novelty claim. #34's frozen competence and transport **NO-GO** outcomes stand without revision. Original source logs are in that author's GitHub research branch; no copies included in Comput's user artifact.

## Next *conditional* experiment gate

**Do not run now.** To distinguish a competence-dependent first-person fallback from voice/syntax, pick *unexposed independent lexical families* and cross first-mentioned Agent/Patient position, verb, construction and role/question with canonical prompt and strict **full-name/format** scoring, demanding equal baseline competence before comparing Qwen and a distinct model family. For multi-event questions, first recover the original reported 64 items and require **three-event** source-disjoint human-validated source provenance; a syntax-aware name copier, a positional first-event fallback and an ambiguity-aware abstain baseline must all be prespecified. Coordinator #31 retains all inference approval; recorded exam pause until 23 Oct remains respected.

**Scientific signal: verified archived model error phenotypes and a strong surface-grammar negative control; no EUREKA.**
