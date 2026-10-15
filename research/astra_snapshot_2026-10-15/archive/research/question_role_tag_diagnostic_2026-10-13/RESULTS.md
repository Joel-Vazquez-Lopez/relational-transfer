# Requested-role tag and candidate-support diagnostic

**13 October 2026.** This is a prospectively frozen, no-training development diagnostic. The protocol, code, item list, checkpoint hashes, and positive screen were fixed in `PROTOCOL.json` before any of these model evaluations. The 40 items are new QA-SRL Bank original-train sentences, sentence-disjoint from the earlier 160 training and 40 development sentences and excluded from the earlier Gold probe. The actor/patient conversion is automatic and has **not** been independently reviewed by a human. The sealed 60-item review set and older active/passive challenge were not used.

Two existing Qwen2.5-0.5B starts were tested for each of the role-first and answer-only arms. Each condition comprises 40 items × actor/patient question × two candidate orders = 160 prompts per checkpoint. Greedy decoding, checkpoint, item, question, and scoring were held fixed. The explicit condition adds `Requested role: ACTOR/PATIENT` before the otherwise unchanged question. The no-candidate condition removes the answer-candidate line; it differs from the checkpoints' training format.

| Start / arm | Ordinary exact | Explicit tag exact | No candidates exact | Tag + no candidates exact |
|---|---:|---:|---:|---:|
| fresh1 / role-first | 88/160 | 85/160 | 34/160 | 36/160 |
| fresh3 / role-first | 85/160 | 91/160 | 36/160 | 30/160 |
| fresh1 / answer-only | 96/160 | 100/160 | 74/160 | 72/160 |
| fresh3 / answer-only | 100/160 | 99/160 | 76/160 | 62/160 |

For role-first, the tag reduced identical final answers to paired actor and patient questions from **64/80 to 49/80** and **68/80 to 44/80**. However, it also reduced correctness of the requested generated role field from **128/160 to 106/160** and **125/160 to 107/160**, and stable role fields across question pairs from **76/80 to 51/80** in both starts. The same-output *oracle-selector* route from printed fields fell from **128/160 to 106/160** and **125/160 to 107/160**. The direct-minus-routed gap remained substantial under the tag: 21 and 16 prompts. All-four-variant direct success remained low: ordinary **2/40, 2/40**; tagged **1/40, 4/40**.

The frozen positive screen required the role-first tag to gain at least 16/160 direct answers in **both** starts, while routed accuracy fell by at most 8/160. It **failed**: direct gains were −3 and +6, with routed changes of −22 and −18. An explicit cue does not, by itself, provide a reliable repair in this test. The lower identical-answer rate does show that it changes behavior, but the simultaneous field deterioration makes a simple question-role localization invalid.

Removing candidates reduced role-first direct accuracy to 34/160 and 36/160 and routed accuracy to 66/160 and 68/160. This indicates strong dependence on the candidate-supported interface under these conditions. Because both arms were trained with candidate lists, the no-candidate result is an out-of-distribution prompt test, **not** proof of a candidate shortcut. The no-candidate exact-answer comparison is also disadvantaged by generative span matching. The answer-only arm did better than role-first on this development bank, so these data do not support a natural-language direct-answer benefit from role-first supervision.

`ANALYSIS.json` contains every cell's exact-answer, joint four-variant, parse, role-field, candidate-position, and routed scores. `EVAL_*.json` retains every prompt-level output with protocol and checkpoint hashes. The item set is one source corpus with provisional semantic labels, the starts share items, and no independent generalization or internal-state mechanism follows from this result.
