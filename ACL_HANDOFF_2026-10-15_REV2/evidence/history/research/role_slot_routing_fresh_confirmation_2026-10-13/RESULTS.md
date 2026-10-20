# Fresh lexical confirmation of slot-first role answers

**Completed 13 October 2026.** The protocol, all three existing checkpoints in each of two starts, and the new lexical bank were fixed before inference. No checkpoint was retrained or selected after seeing this bank. The reserved human-review challenge was not accessed.

## Question and test

Does the slot-first gain over both answer-first and a candidate-list control persist for new names and verbs? The six checkpoints share a Qwen2.5-0.5B-Instruct base. The three arms were trained on the same 48 directed events, prompts, answer and role labels, 96 updates, and 8,640 target tokens per checkpoint. They differed in output order: `Answer; Actor; Patient`, `First candidate; Second candidate; Answer`, or `Actor; Patient; Answer`.

This confirmation used six previously unused participant pairs, four previously unused reversible verbs, both event directions, four sentence forms, and five question wordings: **48 events × 4 forms × 5 questions = 960 outputs per checkpoint**. The primary score counts a crossed form/question event-form case only when both actor and patient answers are correct. It covers passive, subject cleft, and object cleft: **144 cases per checkpoint**. The prespecified positive screen required slot-first to beat *each* control by at least 12/144 in *each* start, lose no more than 5/48 on active cases, and gain at least 6/48 on object clefts.

## Frozen outcome

| Start | Answer-first /144 | Candidate-first /144 | Slot-first /144 | Gain over candidate-first |
|---|---:|---:|---:|---:|
| fresh1 | 25 | 59 | **106** | **+47** |
| fresh3 | 36 | 46 | **103** | **+57** |

**The prespecified screen passed in both starts.** Every output parsed to one of the candidate names. Slot-first beat candidate-first in all six pair families in each start, although these families came from one constructed generator and are not independent language domains.

| Start and arm | Active /48 | Passive /48 | Subject cleft /48 | Object cleft /48 |
|---|---:|---:|---:|---:|
| fresh1 answer-first | 13 | 0 | 21 | 4 |
| fresh1 candidate-first | 14 | 16 | 31 | 12 |
| fresh1 slot-first | **39** | **25** | **45** | **36** |
| fresh3 answer-first | 12 | 0 | 31 | 5 |
| fresh3 candidate-first | 19 | 4 | 30 | 12 |
| fresh3 slot-first | **40** | **24** | **47** | **32** |

On two novel question wordings across all four forms, the both-roles-correct score was 47/192, 76/192, and **143/192** for fresh1 answer-first, candidate-first, and slot-first; fresh3 gave 38/192, 57/192, and **141/192**. Passive remains a substantial weakness even in slot-first: **25/48 and 24/48** on the crossed score.

## Interpretation and limits

This independently chosen lexical bank confirms a large **constructed-language answer-behavior** gain over two matched output-format controls. The candidate-first arm matters: merely listing both names before the answer did not reproduce the slot-first score. The result now spans two evaluation banks, two training starts, and six fresh pair families in the confirmation bank.

The mechanism is narrower than “the model learned roles internally.” When a deterministic rule selects the requested role from the **answer-first model's own generated actor/patient fields**, that model reaches **98/144 and 99/144**, near slot-first's direct answers of 106/144 and 103/144. The same routing rule applied to slot-first's generated fields reaches 116/144 and 111/144. This rule uses a hand-known classification of each question as actor or patient and is therefore a diagnostic, not an end-to-end repair. It suggests that a major failure is connecting available role information to the requested answer. It does not establish a new internal representation or causal pathway.

The placebo was designed after the earlier 36-event pilot, but **before** this bank and all six evaluations here. This is a prospective lexical confirmation, not an independent training replication. All items remain synthetic, all checkpoints share one 0.5B model family, and the six pairs share the same template generator. The natural-language and cross-family claims required for a stronger paper are still open.

## Next decisive tests

1. Test the three fixed arms on a separately sourced, human-annotated natural-language role set. Keep agent/patient separate from grammatical subject/object, and report passives, ambiguous cases, and response-format errors separately. Existing provisional EWT labels and the reserved human-review challenge cannot serve as verified gold.
2. Replicate training on at least one different instruction-model family with frozen data, budgets, prompts, and metrics.
3. Test output-format crossover and no-role structured controls to isolate semantic labels from sequence order and answer routing. Passive examples are an especially important stress test.

## Audit trail

`PROTOCOL.json` was frozen with test-row and checkpoint hashes. `confirm.py` generated the six `EVAL_*.json` files, containing **5,760 raw outputs**. `analyze.py` and `ANALYSIS.json` contain per-form, per-family, secondary-wording, routing, and positive-screen results. The protocol and inference script hashes were checked during analysis. See the [earlier pilot](../role_slot_routing_pilot_2026-10-12/RESULTS.md) for training and post hoc control details.
