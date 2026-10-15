# Compass Investigation 02 — Structural shortcut falsification (2026-10-15)

**Status:** Exploratory research; no new pretrained-model inference, training, or mechanistic claim. No sealed evaluation data used.

## Verified archived-data finding
A name-position lookup using only the construction and queried role recovers **1344/1344 gold answers** in the archived `strong_free` panel: seven configurations, 192 answer records per configuration. The panel contains four constructions, two names per event, two role questions, two wordings, two reversal directions and six reused name-pair groups. The gold roles are deterministic functions of the first/second mention position given the construction. Thus **even 100% accuracy here cannot differentiate a reusable relational computation from a template-plus-position shortcut**. It does not establish that Gemma actually uses a shortcut.

Leave-one-name-pair-group-out lookup rules, trained to predict each model's *response position* from (construction, wording, role), match 181/192 Llama responses; 179/192 Phi-3; 155/192 Qwen 1.5B; 177/192 Qwen 7B; 192/192 Gemma. These are development-only behavioral compression results, **not evidence of internal strategy or held-out natural-language generalization**.

## Iterative anti-shortcut task design
- **Two independent events, four people**: 4096 generated questions. A global first/second name heuristic gets 2048/4096, but a local verb-plus-template position rule gets **4096/4096**. Insufficient; retain as a negative control.
- **Linked events, unique action verbs**: 3072 questions. Still shortcut-solvable through a single clause; insufficient.
- **Repeated-verb relational graph**: 12288 development questions from eight strongly reused lexical world groups. Six events: two `nudged`, two `thanked`, two `tagged`. Questions require following two relations to disambiguate between repeated verbs. Three independently crossed edge permutations preserve the *complete word multiset* while changing specific relational answers; 16 form patterns and four clause orders.

Deterministic generation checks passed for **36864 counterfactual edits**, including appropriate invariance for irrelevant edits and changes for causally relevant edits. The predeclared **single-clause** baseline gets **6144/12288 (50%)**. An event-aware, question-conditioned algorithm may still solve this panel, so successful behavior **does not establish pre-question computational reuse**.

## Scientific opportunity
Do **not** treat score improvements on template actor/patient pairs as evidence for structural understanding. The more informative hypothesis is that competing models differ in how they compose and selectively update event bindings, *even where aggregate answer accuracy is matched*. Develop negative-control systems with deliberately precomputed versus reconstructed bindings, matched behavior, and interventions whose predictions diverge.

## Scientific constraints
- Generated tasks are syntactic development examples, **not** human-validated or sealed confirmation.
- Only eight lexical world groups; thousands of transformed records are not thousands of independent observations.
- Identical token bags defeat bag-of-words shortcuts, **not all** surface heuristics.
- No pretrained forward passes, causal interventions, or new accuracy evidence has been produced in this investigation.
- The experiment must distinguish *reuse of a circuit* from *reuse of a precomputed relational value*.

## Reproducibility
Full standalone, stdlib-only package including archive parser, analysis, exact generated JSONL, manifests, validation checks and research note is `COMPASS_STRUCTURAL_COMPUTATION_LAB_2026-10-15.zip`, attached in the Compass research conversation. Its SHA-256 manifest was checked in the working container. The original research foundation archive was read-only. Scripts:

```bash
python shortcut_audit.py /path/to/COMPASS_RESEARCH_FOUNDATION_2026-10-15 --out results
python structural_challenge.py --out results
```

Literature adjacency: [Mondorf et al., ACL 2025](https://aclanthology.org/2025.acl-long.727/) studies modular circuit reuse, and [Geiger et al., JMLR 2025](https://www.jmlr.org/papers/v26/23-0058.html) supplies a general causal-abstraction framework. Novelty is NOT established.

## Next
1. Independent human review of generated question semantics; eliminate awkward/ambiguous cases.
2. Falsifiable native-path causal cut: transparent cached/reconstructed systems as controls, then only approved small-model interventions.
3. A genuinely untouched evaluation designed *after* the development task is frozen.
