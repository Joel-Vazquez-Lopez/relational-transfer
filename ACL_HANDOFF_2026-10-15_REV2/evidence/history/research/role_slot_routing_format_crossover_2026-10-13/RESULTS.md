# Output-format crossover: the location of the answer matters

**Completed 13 October 2026.** This is a post hoc mechanism diagnostic following the fresh lexical confirmation. It reuses the same 960 constructed prompts per checkpoint and the same four answer-first/slot-first checkpoints; it does not retrain a model or test new language. The crossover protocol and script hashes were frozen before the four crossover inference runs.

## Test

Each checkpoint was evaluated once with its trained output format and once with the opposite format requested in the system prompt. Thus the answer-first model was asked to generate `Actor; Patient; Answer`, and the slot-first model was asked to generate `Answer; Actor; Patient`. The sentence, question, checkpoint, decoding settings, and scoring items stayed fixed.

Two scores are needed. **Strict** accepts an answer only if the complete requested format parses. **Prefix-tolerant** reads the first token as the answer when the prompt asks for answer-first, or reads a tagged `Answer:` field when the prompt asks for slot-first. A separate **role-field routing** diagnostic picks the requested actor or patient from the generated fields. The latter uses the known role type of each question and is not an end-to-end solution.

## Outcome on crossed form/question cases (/144 per checkpoint)

| Checkpoint | Native strict answer | Opposite-format strict answer | Opposite-format prefix-tolerant answer | Opposite-format role-field routing |
|---|---:|---:|---:|---:|
| fresh1 answer-first | 25 | 11 | 11 | 80 |
| fresh3 answer-first | 36 | 17 | 17 | 85 |
| fresh1 slot-first | **106** | 0 | 24 | 2 |
| fresh3 slot-first | **103** | 0 | 19 | 0 |

When asked for an answer *first*, the slot-first checkpoint's first-token score falls to **24/144 and 19/144**, close to the native answer-first checkpoint's 25/144 and 36/144. Its strict full-format score is zero because it almost never generates the requested `Actor:` field after that first token. This is not evidence that its role information vanished: the new prompt places the answer before any explicit role fields, which is an out-of-training-format demand.

When asked for roles *first*, the answer-first checkpoint produces an actor and patient field on all 960 outputs per start, and deterministic routing from those fields scores **80/144 and 85/144**. But it omits the `Answer:` field on 444/960 and 303/960 outputs, so direct answers fall to **11/144 and 17/144**. Its generated roles remain useful even though its answer generation is brittle under the changed output contract.

## Scientific interpretation

The strong slot-first result depends on **producing the requested answer after explicitly producing role fields in a format the model was trained to use**. A prompt-only crossover does not recover that behavior from the answer-first checkpoint; in fact, both crossovers cause severe schema failures. Together with the prior candidate-first control, this supports a narrower claim: in this controlled setting, training a role-first answer interface helps the model use event roles for held-out questions and constructions. It does **not** prove a changed internal representation or show that this technique transfers to natural language or another model family.

The crossover is a distribution shift in the requested output syntax. It cannot cleanly separate training effects from learned format adherence, so the zero strict scores must not be presented as a pure causal effect of output order. The important diagnostic is that useful role fields can coexist with poor direct answers, and moving the answer before those fields removes the slot-first checkpoint's measured advantage. A future matched training comparison should keep the output format stable while separately manipulating whether role labels are semantically correct, whether they precede the answer, and whether the answer is trained to copy the requested slot.

## Audit

`PROTOCOL.json` hashes the script, both checkpoints per start, prior test rows, and prior protocol. Four `EVAL_*_cross.json` files hold **3,840 raw outputs**. `analyze.py` produces `ANALYSIS.json` with strict and prefix-tolerant scoring, generated-field routing, field presence, and sample raw strings. The six native results remain in the [fresh confirmation](../role_slot_routing_fresh_confirmation_2026-10-13/RESULTS.md).
