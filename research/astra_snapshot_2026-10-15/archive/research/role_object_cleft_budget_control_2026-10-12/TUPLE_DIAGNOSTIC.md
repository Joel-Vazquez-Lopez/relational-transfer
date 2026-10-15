# Post hoc ordered-tuple diagnostic (12 October 2026)

This is an exploratory output-level check on the already inspected 36-event constructed bank. The model scored two completions, `(actor, patient)` and its reversal, after one prompt asking for both roles. The correct tuple was preferred when its teacher-forced log score was higher. A sentence-omitted control exposed name/order preference. No weights were changed.

| Start | Checkpoint | Active, sentence present | Object cleft, sentence present | Sentence omitted, either form |
|---|---|---:|---:|---:|
| fresh1 | ordinary SFT | 27/36 | 21/36 | 18/36 |
| fresh1 | role-coupled | 27/36 | 24/36 | 18/36 |
| fresh3 | ordinary SFT | 27/36 | 24/36 | 18/36 |
| fresh3 | role-coupled | 27/36 | 26/36 | 18/36 |

The sentence helps tuple preference in object clefts, but less than in active clauses. The object-cleft direct patient answer was poor even on training events, and repeated object-cleft exposure under the matched budget did not repair strict two-role accuracy. Taken together, these checks motivate testing whether explicit event-role supervision plus a link from role slots to answers improves transfer. They do **not** prove that a correct role tuple is represented in an internal state, or that question routing is the sole cause of direct-answer failures. The tuple prompt and score format differ from direct QA, and this bank was already inspected.

Frozen protocol: `TUPLE_PROTOCOL.json`. Implementation: `tuple_diagnostic.py`. Raw files: `TUPLE_fresh1_old_sft.json`, `TUPLE_fresh1_old_coupled.json`, `TUPLE_fresh3_old_sft.json`, `TUPLE_fresh3_old_coupled.json`.
