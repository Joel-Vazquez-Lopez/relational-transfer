# COMPUT-015 — Generator-aware shortcut falsifies V2 aggregate accuracy as a strong benchmark

**Date:** 2026-10-15. **Status:** executed label-safe baseline on independent exact-structure reconstruction of `build_pairs_v2.py`; 7 local tests passed. Original upstream source inspected on `research/protocol-auditor-same-query-binding-2026-10-15`, but authoritative upstream Python was not executed byte-for-byte. No model inference or causal patching.

The V2 generator computes `s=(family_index + 2*voice_index + clause_order_index + query_role_index)%4`, then assigns four cyclically sorted participant identities. Its fixed 8-verb list selects the target verb at `VERBS[family_index%8]`, revealing `family_index%4` to a verb-index lookup. A **label-free predictor** can use only sorted four name identities, query verb identity, passive `was` marker, queried verb's clause index, and question role:

`predicted_index=(VERBS.index(query_verb) + 2*passive + queried_clause_index + 2*patient_question)%4`.

It does **not** read semantic actor/patient order, graph edges, sentence-local subject or gold labels. Numeric name suffixes are **not** used to derive the answer.

### Results: 512 paired cases, 1,024 answer opportunities

| Condition | Base correct /128 | Donor correct /128 |
|---|---:|---:|
| identity | 128 | 128 |
| target_role_swap | 128 | **0** |
| distractor_role_swap | 128 | 128 |
| predicate_swap | 128 | **0** |
| **All** | **512/512** | **256/512** |

**Aggregate 768/1,024 = 75%**, including 75% on the openly held-out-looking family partition because the same assignment formula transfers. Previously tested simple sorted-name+role rule on V2 scored 256/1,024 (25%); surface grammar parser scored 1024/1024. The new result finds a more capable **lexical/position/metadata generator artifact** despite simple name-order balancing. Exact fixture and independently written, input-only evaluator: `COMPUT_015_V2_GENERATOR_SHORTCUT.zip` (conversation artifact, SHA256 `07a0a7da6fb81b6961ef1c977291452f42927e46a9d41509c7c39ebcc22ccdd2`).

**Crucial limit:** This shortcut fails **all 256 gold-changing donor predictions**, so **does not** pass the actual causal binding-specific paired-transition test! Aggregate answer accuracy is not a valid substitute for matched donor transition accuracy. It is deliberately generator-aware and does not establish that any LLM learns this arithmetic. Baseline rules must be frozen before unseen data and models.

**Scientific integration alert:** Brain-O's newest Experiments 031–034 already test literal same-query, changed-source binding in trained synthetic Transformers. After layer 1, source-verb-state patch matches donor 300/300 and 289/300, but query-verb-state patch matches 0/300 and 6/300. After layer 2, query-verb state matches 300/300 and 299/300. Selective attention cuts reduce performance to ~chance. PR #20 and RELATE-001 should be reconciled with this *newer layerwise relay* evidence before running duplicate inference; protocol/checkpoint compatibility is not established. This result still does NOT demonstrate an abstract binding circuit or pretrained LLM generalization.

**Next design gate:** independently randomize name-role, verb-family and question/voice/position assignment; preserve same-query/lexical bag and gold-changing vs invariant controls; audit stress/true held-out family separation; use a competent source-local syntactic-parser negative control and exact frozen Brain-O vocab/model/cut; report intervention selectivity not aggregate task score. No sealed examples, weights or paid model runs used.

**Reproduction:** `python3 research/comput/comput015_v2_shortcut.py /path/to/open_binding_pairs_v2.jsonl`. Independent fixture generator/tests provided in the attached ZIP. Original V2 unchanged.