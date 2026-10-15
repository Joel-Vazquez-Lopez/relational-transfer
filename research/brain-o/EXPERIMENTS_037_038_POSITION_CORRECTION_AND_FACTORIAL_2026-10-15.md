# Brain-O E037–E038: correct token-position bug and test crossed causal routes

**Date:** 2026-10-15. **Status:** executed exploratory analysis on two frozen three-layer synthetic Transformer checkpoints. No new training, paid inference, sealed data, or pretrained-LLM mechanism claim.

## E037: critical ERRATUM to E035 and E036
The original participant-name patch and attention-mask group `[1,4,7,11]` is **wrong for mixed active/passive input**: actual second participant positions are clause-relative +3 in active and +4 in passive. For second clause, +3 is position 10 (NOT 11). The original group sometimes included grammar and omitted a name. Reinterpret all earlier source-name effects as **partially mislabeled**.

Rerunning exactly 200 disjoint same-query role-swaps per seed with per-example *true* source-name positions:

| Patch donor-answer match after L1 | 3L 1016 | 3L 1017 |
|---|---:|---:|
| Old fixed group | 148/200 | 145/200 |
| Correct four name tokens | **200/200** | **197/200** |
| Correct two names in queried event | **200/200** | **197/200** |
| Query verb alone | **0/200** | **0/200** |

Native L3 answer→name attention blocking on 240 new disjoint test cases per seed: **100%** clean; **67.50%/66.67%** old fixed mask; **55.00%/53.75%** corrected *per-example four-name* mask. 4-head, batch-specific masks; attention normalization remains a confound. This strengthens evidence that participant-name states contribute causally; no unique semantic role circuit established.

## E038: crossed predicate-event and participant-role factorial
Within the **same original training grammar and vocabulary** with disjoint entities, construct 200 worlds, seed 2026101538. Same query and source word multiset throughout. A swaps source predicate positions, B swaps actor/patient in BOTH events; A and B commute, and the four corresponding gold answers are four distinct participant names.

- **A alone:** replace L2 query-verb hidden state with A-donor query-verb representation; hold base other states.
- **B alone:** replace all four *actual* source-name token states after L1 with B-donor states.
- **A+B:** first perform B name-state patch after L1; advance to L2; then transplant A query-verb state after L2; continue final layer. AB target is **not** used as activation donor.
- Control: base self-patch at both cuts and distractor-event-only role swap.

| Correct expected answer / 200 | 3L 1016 | 3L 1017 |
|---|---:|---:|
| Native base / A / B / AB (each) | 200 | 200 |
| A-only patch | 200 | 200 |
| B-only patch | 200 | 200 |
| **Combined A+B patch predicts fourth gold name** | **200** | **197** |
| Sham and distractor retain base prediction (each) | 200 | 200 |

Mean probability assigned to AB gold under combined patch: ~0.999 (seed 1016), ~0.976 (seed 1017). The corrected patch results are numerically reproducible; raw per-condition answer-category breakdown, gold logit margins and descriptive mixed differences are in JSON.

**What this supports:** two separately manipulable, approximately composable causal dependencies for selecting a source event and retrieving the participant in two learned synthetic networks. **What it does not:** exclude competent lexical lookup/copying or question-time reconstruction, establish a semantic abstraction, transfer to 3+ events or natural language, prove a minimal depth requirement, or demonstrate ACL-level novelty. Compass I12 showed generic copying/addition also exhibits layerwise causal transport.

**Next gate** requested by the coordinator (GitHub #12): matched trained competent lexical-key copier, then repeated predicates, varying event counts/layout, held-out families, masked-attention normalization controls, independent review. Auditor's v2 generator is shortcut-leaky and not tokenizer compatible as-is; do not treat it as a sealed benchmark.

## Reproduction
Conversation artifact `BRAIN_O_POSITION_AND_FACTORIAL_037_038_2026-10-15.zip`, SHA256 `3264057483169e0cb8f2d24b293097755584b2c92babfa569d52cc8e36238283`. Contains `position_integrity_audit.py`, `crossed_factorial.py`, JSON results, exact replay logs, methodological note and manifest. Depends on frozen `BRAIN_O_DUAL_ROUTE_035_036_2026-10-15.zip` source/checkpoints in this conversation. The artifact is **not** included in this GitHub branch. Both scripts were rerun and the ZIP passed CRC. No other investigators' files changed; no merge to main.
