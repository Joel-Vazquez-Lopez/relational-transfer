# Matched role/subject decoupling pilot: a real trade-off, not a repair

**Date:** 12 October 2026. **Status:** completed prospective exploratory test with frozen items and success rule. The reserved human-review active/passive challenge was not used.

## Why this experiment

The preceding [Qwen role-equivariance diagnostic](../qwen_role_equivariance_diagnostic_2026-10-12/RESULTS.md) found that continuing two Qwen2.5-0.5B adapters on 256 active QA-SRL events improved an active natural-language proxy but shifted answers toward the subject of the target clause. Active examples do not separate subject from actor. This experiment asks whether adding sentences that do separate them improves role transfer beyond the same amount of additional role training in aligned constructions.

Both arms began from the same natural-role adapter within each of two earlier starts. Both saw the same 48 directed training events (six name pairs, four verbs, both event directions), answered actor and patient questions, and used 96 updates with six labeled examples per update at LR 5e-5. The **aligned** arm saw active, subject-cleft and object-cleft sentences, in which the target clause's subject is the actor. The **decoupled** arm saw active, passive and object-cleft sentences; the passive makes the grammatical subject the patient. The two arms shared active and object-cleft items and differed in one construction. Prompt-token totals over two epochs were 27,936 versus 27,744, a 0.7% difference. Both had 576 labeled examples.

Before training, I froze a 36-event test with **disjoint names and verbs**. It used active, passive, both clefts, and two never-trained relative-clause forms. Each event was asked once for actor and once for patient under fixed prompts. The primary endpoint was both roles correct on the patient-relative construction, in which the subject of the relative clause is the patient while the actor is the first named participant. The [protocol](PROTOCOL.json) required a gain of at least six events in *each* start, with retention of the actor-relative and active results, for a positive screen.

## Results

Values are events with **both roles correct**, out of 36 for each form. “All prompts” combines 432 individual answers.

| Start / checkpoint | Active | Passive | Subject cleft | Object cleft | Actor relative | Patient relative | All prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh1 source | 9 | 2 | 16 | 5 | 8 | 10 | 255/432 |
| Fresh1 aligned | 25 | 15 | 28 | 10 | 27 | 21 | 341/432 |
| Fresh1 decoupled | 22 | **30** | 22 | 16 | 25 | **25** | 348/432 |
| Fresh3 source | 13 | 5 | 14 | 3 | 11 | 13 | 251/432 |
| Fresh3 aligned | **26** | 4 | **29** | 11 | 24 | 12 | 319/432 |
| Fresh3 decoupled | 17 | **32** | 24 | 13 | 22 | **16** | 328/432 |

Decoupled training improves the primary patient-relative count by **+4/36 in each start**, and passive accuracy improves substantially. But active individual-answer accuracy falls by **11.1 and 20.8 percentage points** relative to aligned training. It also loses some actor-relative and subject-cleft performance. The frozen positive screen **fails in both starts**.

The trained checkpoints sometimes returned `The` as the first answer token on **patient questions in the relative-clause forms** rather than returning a candidate name. Invalid first tokens count wrong under the frozen scoring rule: 2 versus 4 in Fresh1 aligned versus decoupled, and 10 versus 17 in Fresh3. Neither source checkpoint produced an invalid token. This answer-format difference may account for part of the relative-clause trade-off, so the exact semantic error rate cannot be inferred from first-token correctness alone. A separate candidate-normalized diagnostic would be needed; it would be post hoc and could not replace the frozen result.

The two-arm contrast is evidence that the *distribution of training constructions matters*. It does not show a successful general repair. It is compatible with the model learning different construction-specific answer rules. Since the curricula differ by which construction replaces the other, it does not isolate subject/actor decorrelation from exposure to passive morphology.

## Decision

The matched arms revealed a coverage trade-off. A separate prospective [mixed-form follow-up](../role_subject_balanced_followup_2026-10-12/PROTOCOL.json) was frozen after these results. It retains active and passive examples while alternating subject and object clefts under the **same 96-update, six-target budget**. A second disjoint lexical test determines whether that simple coverage change removes the trade-off. This follow-up is exploratory because its design was informed by the table above.

## Audit

- [Frozen protocol and source hashes](PROTOCOL.json), [training and evaluation code](experiment.py), [token budget audit](TRAIN_TOKEN_AUDIT.json), [derived analysis](ANALYSIS.json), and [analysis code](analyze.py).
- Immutable checkpoint directories and six `EVAL_*.json` files contain every prediction.
