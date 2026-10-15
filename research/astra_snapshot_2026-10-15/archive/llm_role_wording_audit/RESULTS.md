# Fresh actor-question wording audit (10 October 2026)

The [protocol](PROTOCOL.json) and [512 prompts](ITEMS.json) were fixed before inference. They cover 64 new events: eight disjoint name pairs, four reversible verbs, and both directions. Each event appears in active and passive voice with four actor questions. All names are single tokens in both local Qwen2.5 tokenizers. The [1.5B](RESULT_1.5b.json) and [0.5B](RESULT_0.5b.json) raw files contain every prompt, exact greedy next-token answer, and correctness flag. Every output was one of the two names.

| Model | Question | Active correct / 64 | Passive correct / 64 |
|---|---|---:|---:|
| Qwen2.5-1.5B-Instruct | Who did the action? | 8 | 64 |
| Qwen2.5-1.5B-Instruct | Who performed the action? | 51 | 60 |
| Qwen2.5-1.5B-Instruct | Which person carried out the action described in the sentence? | 40 | 62 |
| Qwen2.5-1.5B-Instruct | Who is the actor in this sentence? | 5 | 64 |
| Qwen2.5-0.5B-Instruct | Who did the action? | 60 | 11 |
| Qwen2.5-0.5B-Instruct | Who performed the action? | 59 | 25 |
| Qwen2.5-0.5B-Instruct | Which person carried out the action described in the sentence? | 52 | 48 |
| Qwen2.5-0.5B-Instruct | Who is the actor in this sentence? | 23 | 20 |

For the 1.5B model, changing only “did” to “performed” improves active accuracy by 43/64 = 67.2 percentage points, close to the prior fresh-set estimate of 67.5 points. This improvement appears in all eight disjoint name pairs (4–7 of eight events per pair). A third, explicit wording (“carried out”) also improves active accuracy by 32/64 = 50 points over “did,” in all eight pairs (3–5 of eight events). Thus the issue is not confined to one pair or to the “performed” phrasing. However, the apparently technical “actor” wording yields only 5/64 active correct, showing that simply replacing an awkward question with an explicit role term does not solve the evaluation problem.

The 0.5B model behaves differently: it is strong on active “did” (60/64) and weak on passive “did” (11/64). It improves most on passive “carried out” (48/64). Only 3/64 events are correct under all eight prompts for 1.5B, and 4/64 for 0.5B. Agreement and accuracy must be reported separately; a stable wrong answer is not relational success.

These are model-specific, template-English behavioral findings. They do not identify the internal cause, establish that the models never saw related sentences during pretraining, or demonstrate a training repair. The new contrasts among “carried out” and “actor” were exploratory, without a predeclared significance threshold. They strengthen the case for evaluating *which* linguistic transformations should preserve the answer and which should change it. A future fine-tuning study should freeze a balanced, independently checked set of question wordings and held-out forms before training, then compare matched training arms on fully correct event groups as well as average accuracy.
