# QA-SRL Gold natural-sentence probe: direct repair does not transfer

**Completed 13 October 2026.** The item set, six existing checkpoints, native prompts, 48-token decoding limit, exact-span scoring, and positive screen were frozen before inference. All six runs finished. This is a diagnostic on one 0.5B Qwen family, not publication validation.

## Data and test

The source is [QA-SRL Gold](https://github.com/plroit/qasrl-gs), whose question-answer spans were consolidated from trained crowd-worker annotations. Starting from an existing rule-filtered candidate file, a stricter rule retained **all 51 predicates in 49 natural sentences** for which the gold question metadata identified a subject-like answer and a direct-object answer, with no prepositional or second-object role. The mapping from these questions to `actor` and `patient` is automatic and **has not been independently human-reviewed**. A few verbs may not express canonical agent/patient roles.

For each predicate, the model saw the original sentence, the two annotated answer spans as candidates in both orders, and the original actor-like or patient-like gold question: **51 × 2 roles × 2 candidate orders = 204 prompts per checkpoint**. The three existing answer-first, candidate-first, and slot-first training arms were tested in each of two starts. Output formatting followed each arm's training. Exact span scoring normalizes case, punctuation, and whitespace; malformed outputs and other spans count wrong.

The prespecified primary score was the number of predicates for which **all four answers** were correct. The positive screen required slot-first to exceed both controls by at least 6/51 in each start and parse at least 80% of its outputs.

## Frozen result

| Start | Arm | All four correct /51 | Individual exact answers /204 | Parsed /204 |
|---|---|---:|---:|---:|
| fresh1 | Answer-first | 0 | 90 | 199 |
| fresh1 | Candidate-first | 2 | **94** | 204 |
| fresh1 | Slot-first | 1 | 84 | 203 |
| fresh3 | Answer-first | 1 | **92** | 204 |
| fresh3 | Candidate-first | 1 | 88 | 204 |
| fresh3 | Slot-first | 2 | 81 | 203 |

**The positive screen failed in both starts.** Slot-first's direct answers are below the controls on the 204-prompt score. The all-four score is near zero for every arm. This result does not support claiming that synthetic slot-first fine-tuning repairs role answers in natural sentences.

## Error structure and post hoc diagnostic

Candidate order is a major confound. Across the six checkpoints, the generated answer matched the **second** candidate 91–117 times out of 204, versus the **first** candidate 51–75 times; the remaining answers matched neither. Actor accuracy tended to rise when the actor was listed second, while patient accuracy tended to rise when the patient was listed second. Restricting to the **26 items with one-word spans for both participants** did not remove the failure: slot-first got only 40/104 and 39/104 individual answers, with just 1/26 all-four-correct item in each start. Thus multiword span extraction alone does not explain the result.

The role fields reveal a narrower, potentially useful signal. If a deterministic diagnostic selects the requested actor/patient from the model's own generated fields, slot-first reaches **112/204 and 109/204** correct answers, versus its direct **84/204 and 81/204**. Its all-four-correct score rises from **1→15/51** and **2→12/51**. Answer-first's generated fields yield **89/204 and 91/204**, with **8/51 and 7/51** all-four-correct. The role fields therefore contain more useful information than the slot-first direct answer on these items. This is a **post hoc oracle-role-routing diagnostic**: it uses our knowledge of which role each gold question requests, so it is not an end-to-end answer system or evidence of a repaired internal representation.

In a post hoc paired bootstrap over the 49 source sentences (10,000 resamples, fixed seed), slot-first routed-minus-direct accuracy was +13.7 percentage points in each start, with approximate 95% intervals **+3.4 to +23.6** and **+3.4 to +24.0** points. These intervals describe this selected set and routing rule; they do not solve the missing human review or out-of-domain problem.

## What this changes

The constructed lexical confirmation remains valid for its own controlled setting. The natural probe shows a clear **boundary**: outputting roles first on synthetic training data does not by itself produce robust natural role answers. The shared candidate-position bias suggests the model is often choosing by presentation rather than the annotated relation. Its generated role fields sometimes provide a better route to an answer, but the resulting 109–112/204 exact accuracy is still modest.

The next repair test should use **disjoint natural training examples** with both candidate orders and verified semantic roles, while keeping a fresh natural source set untouched for final confirmation. Compare ordinary QA, candidate-order controls, role-first supervision, and explicit answer-from-slot training under matched budgets. Do not train on these 51 QA-SRL Gold items or treat a new result on them as independent confirmation. A second instruction-model family is needed for a broader LLM claim.

## Audit trail

`PROTOCOL.json` hashes the item list, original gold CSV files, script, candidate file, and six fixed checkpoints. `ITEMS.json` contains every selected item. Six `EVAL_*.json` files hold **1,224 raw outputs**. `analyze.py` and `ANALYSIS.json` report every arm, order, role, parsed output, candidate-position match, one-word subset, and diagnostic role routing. The reserved human-review challenge was not accessed.
