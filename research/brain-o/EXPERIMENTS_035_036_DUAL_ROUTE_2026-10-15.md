# Brain-O Experiments 035–036: Event selection vs participant retrieval
**15 October 2026 | Exploratory, synthetic causal intervention study; not pretrained-LLM mechanistic evidence.**

## Background
Prior E031–034 showed source-verb→query-verb→answer pathway for fixed-query predicate reassignment in trained 3-layer Transformers with local event-index supervision. Compass Investigation 12 observed broadly similar handoffs in nonrelational copying and modular addition. Our next question: Does **same-query actor/patient reassignment** travel through that same query-verb path?

## E035: role swap with identical query
200 newly generated disjoint cases per successful 3-layer seed (1016, 1017). Swap actor/patient in the queried event, preserving query verb, role token, syntax and input word multiset. Both models native-correct 200/200 base and 200/200 donor, with native answer changing 200/200. Distractor-event role swap leaves answer unchanged 200/200.

Donor-answer restoration after swapping group states:
| Patch | Seed 1016 | Seed 1017 |
|---|---:|---:|
| Source name states after L1 | 148/200 | 145/200 |
| Source name states after L2 | 74/200 | 87/200 |
| Query-verb state after L1 | 0/200 | 0/200 |
| Query-verb state after L2 | 0/200 | 0/200 |
| Source-verb states after L1 or L2 | 0/200 | 0/200 |
| Answer/role-token state after L2 | 96/200 | 16/200 |

**Negative result:** a query-verb representation at L2 that transfers verb-assignment changes in E032–033 does **not** transfer same-query actor/patient swaps. This rejects the simplistic claim that the complete event-role relation is packaged there. Whole-state splices can be off-manifold and the two seeds differ on other patches.

## E036: selective native attention-edge masks
240 fresh generator cases per overlap condition, per seed; below is disjoint-case accuracy, clean 100% in both models:
| Cut | Seed 1016 | Seed 1017 |
|---|---:|---:|
| L2 query verb cannot attend to source verbs | 50.42% | 46.67% |
| L3 answer cannot attend to query verb | 50.83% | 50.83% |
| L3 answer cannot attend to source names | 67.50% | 66.67% |
| L2 answer cannot attend to source names | 72.92% | 98.33% |
| L2 or L3 answer cannot attend to source verbs | 100% | 100% |
| L2 **and** L3 answer cannot attend to source names | 37.08% | 60.83% |

Interpretation: task-effective event selection and participant/name retrieval have **partly separable causal dependencies** under the current narrow grammar. It remains viable that these are ordinary lexical matching and copying circuits, not relation-specific mechanisms. Masking changes attention normalization; effects are not isolated information deletions.

## Next falsification
- Compare matched copying and modular-addition backbones to reject generic retrieval explanations (Compass 12).
- Test 3+ events, variable positions and unseen verb/name families; cluster by independent generation family.
- Apply same-query target/distractor and predicate swaps from protocol-auditor PR #20 with full clean-competence reporting and activation validity checks.
- Seek competence-matched natural-language training/heldout tests only after controlled mechanism passes adversarial checks.

## Reproducibility
Chat artifact `BRAIN_O_DUAL_ROUTE_035_036_2026-10-15.zip` (SHA-256 `abeff7b1d63c536f1d9de1a5423f779e70224223b8a8b1c10301756b5178e59d`, archive integrity and result invariants verified; both experiments independently replayed after extraction with exact JSON metric reproduction); includes executable scripts, complete metric JSON, E031-34 source and frozen 3L checkpoints. Binary ZIP is **not uploaded into the GitHub branch**. No new model training, no sealed data, no modifications to other branches.
