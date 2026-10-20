# Slot-first answer training: positive pilot, with a crucial routing caveat

**Completed 13 October 2026.** The main two-arm Qwen2.5-0.5B protocol was frozen before training or inference on a new 36-event lexical bank. Two starts were run. A candidate-list placebo was designed **after** the main results were seen and is exploratory. No reserved human-review challenge was accessed.

## Question and design

Does producing the event roles before answering help transfer to a sentence-form/question pairing absent from training? Both main arms saw the same 48 directed training events, the same actor, patient, and answer labels, the same nine prompts per event, 96 updates, and **8,640 target tokens** per checkpoint. Only output order changed:

- **Answer-first:** `Answer: X; Actor: A; Patient: B`.
- **Slot-first:** `Actor: A; Patient: B; Answer: X`.

The test used six new participant pairs and three new verbs in both event directions (36 events), four forms and five question wordings (720 outputs/checkpoint). The primary score counts an event-form only when both actor and patient questions are correct. It covers passive, subject cleft, and object cleft: **108 cases/checkpoint**. In each construction, one of those question/form combinations was withheld during training. All output formats, questions, metrics, and the positive screen were fixed before training.

## Frozen result

| Start | Answer-first /108 | Slot-first /108 | Difference | New-wording joint /144, answer-first → slot-first |
|---|---:|---:|---:|---:|
| fresh1 | 28 | **77** | **+49** | 37 → **106** |
| fresh3 | 31 | **72** | **+41** | 35 → **101** |

The **frozen positive screen passed in both starts**: gains exceeded +10/108, active joint accuracy did not regress, and object-cleft joint accuracy improved. Object-cleft crossed joint changed **6→30/36** and **9→27/36**. Active crossed joint changed **19→33/36** and **19→32/36**. The slot-first arm generated 720/720 and 719/720 valid parsed answers in the two starts; the one invalid answer was retained as wrong. These are exploratory constructed-language results in one model family, not a general LLM repair.

## Post hoc controls and what they change

To test whether merely placing names before the answer caused the gain, a **candidate-first** control produced `First: candidate 1; Second: candidate 2; Answer: X`. It used the same training events, question schedule, source adapters, 96 updates, and **8,640 target tokens**. It labels candidate order, not semantic roles. This control was added after seeing the main result, so it does not count as independent confirmation.

| Start | Answer-first direct /108 | Candidate-first direct /108 | Slot-first direct /108 | Answer-first routed from its generated slots /108 | Slot-first routed from its generated slots /108 |
|---|---:|---:|---:|---:|---:|
| fresh1 | 28 | 48 | **77** | 75 | **91** |
| fresh3 | 31 | 41 | **72** | 71 | **90** |

Candidate-first shows a generic answer-last/name-priming benefit, but remains **29 and 31 cases below slot-first**. Slot-first beats it in all six participant-pair families in both starts, though the families share one constructed generator. The role labels appear useful in this setting.

The **routing caveat is important**. Answer-first often generated correct actor/patient slots *after* a wrong first answer. If a deterministic rule ignores that answer and selects the requested role from its generated slots, it reaches **75/108 and 71/108**, nearly matching the slot-first model's direct answers. Conversely, routing from slot-first's generated slots reaches **91/108 and 90/108**. These routed scores use a hand-known actor/patient classification of each question and are *derived* from the already generated outputs; they are not end-to-end free-form question understanding. They show that much of the gap is in getting the answer to use available role information, rather than simply discovering roles.

## Scientific interpretation

This is the strongest local **training-format** result so far. Showing the model actor and patient before the answer greatly improves answer behavior on new names, verbs, and crossed form/question combinations. It is more than merely putting both names first under this control. It is **not yet evidence** that the model learned a new, stable internal linguistic representation: output order and role labeling changed together, the generated roles were already useful in the answer-first arm, and the study is synthetic with two related starts. The earlier logit-level transport penalty did not help; explicit sequential role output did.

This also narrows the next experiment. A fixed, three-arm comparison on a **new** lexical bank should include answer-first, candidate-first, and slot-first from the outset, plus a format crossover or matched dummy-structure control. A human-verified natural challenge and a second model family are needed before claiming general training repair or conference-level novelty. Structured semantic prediction and inference already have strong precedents, especially [Mehta et al. (2024)](https://aclanthology.org/2024.naacl-long.7/) and [QASem Parsing (2022)](https://aclanthology.org/2022.emnlp-main.528/). The possible contribution is the tightly controlled role-transfer and answer-routing effect, if it survives those tests.

## Audit

- Main: `PROTOCOL.json`, `run.py`, four checkpoint directories with `TRAINING.json`, four `EVAL_*.json` files (2,880 raw outputs), `analyze.py`, `ANALYSIS.json`.
- Post hoc placebo: `../role_slot_routing_placebo_2026-10-12/PROTOCOL.json`, `placebo.py`, two checkpoint directories with traces, two `EVAL_*.json` files (1,440 raw outputs), `analyze.py`, `ANALYSIS.json`.
- All six runs used 96 updates and 8,640 target tokens each. Source adapters and the reserved human-review challenge were unchanged.
