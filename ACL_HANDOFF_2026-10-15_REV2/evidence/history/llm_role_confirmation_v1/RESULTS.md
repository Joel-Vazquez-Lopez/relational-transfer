# Pretrained-language-model role evaluation (10 October 2026)

## What was run

The main confirmation used the unmodified local Qwen2.5-1.5B-Instruct snapshot. The [protocol](PROTOCOL.json), [120 events and 480 prompts](ITEMS.json), and scoring rule were saved before inference. Three sets each used five new name pairs, four new reversible verbs, and both actor directions. Each event appeared in active and passive form with the two actor questions used in the earlier diagnostics. Names, verbs, and pairings were new to those panels; the question wordings were already chosen. Exact raw first-token answers and actor-versus-patient candidate margins are in [RESULT.json](RESULT.json), with a [machine-readable summary](SUMMARY.json). All 480 answers were one of the two names.

| Model | Active, “did” | Active, “performed” | Passive, “did” | Passive, “performed” | All four correct |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 10/120 | 91/120 | 120/120 | 119/120 | 10/120 events |
| Qwen2.5-0.5B-Instruct | 86/120 | 81/120 | 63/120 | 102/120 | 42/120 events |

For 1.5B, changing only the actor-question wording increased active accuracy by **67.5 percentage points**, with gains of 70.0, 77.5, and 55.0 points in the three separate sets. Passive accuracy stayed near ceiling. The predeclared replication rule of at least 20 points in every set therefore passed. The wording-by-voice interaction was +68.3 points. An exact two-sided sign-flip over the 15 disjoint name pairs gives *p* = 0.000061; a pair-resampling 95% interval is [60.8, 75.0] percentage points. Those uncertainty calculations condition on the chosen templates and verbs, which are reused within sets. The independent units are not 480 unrelated prompts.

Under the generic question, 1.5B chose the **last name in 230/240** active and passive prompts combined. That positional shortcut explains much of this particular contrast; it is not evidence that the model lacks role information everywhere. Earlier subject-cleft diagnostics also showed that it does not always choose the last name across sentence forms.

The 0.5B size check used [the identical items](ITEMS.json), the same prompts and scoring, and its own [frozen protocol](SCALE_PROTOCOL.json). Its [raw answers](SCALE_RESULT.json) show a **different** pattern: the generic active wording was relatively strong, while passive performance was weaker. The 0.5B model answered all four versions correctly on 42 events, compared with 10 for 1.5B. These are two models from the same family, trained separately; the comparison does not identify parameter count as the cause and does not generalize to LLMs as a class.

## Authentic sentences

From [QA-SRL Gold](https://github.com/plroit/qasrl-gs), a human-annotated corpus sampled from Wikinews and Wikipedia, a fixed rule produced [100 candidate predicates](QASRL_CANDIDATES.json). Source and annotation inspection before model inference selected [63 distinct sentences](QASRL_ITEMS.json). Each was asked about the actor and a second participant with two fixed wordings and both A/B option orders: 504 prompts. The [protocol](QASRL_PROTOCOL.json), [raw choices](QASRL_RESULT.json), and [summary](QASRL_SUMMARY.json) are saved.

Qwen2.5-1.5B selected the annotated actor in **209/252** actor prompts and the annotated second participant in **151/252** other-role prompts. It got all four actor variants correct for **32/63** sentences, and all eight actor/other-role variants correct for **6/63**. The option order mattered: for the wording “Which listed phrase is the actor of the target verb?”, actor accuracy was **63/63** when the actor was option A and **34/63** when it was option B. This is evidence that the evaluation format itself changes measured performance. All 504 responses were valid A/B choices.

This authentic check is **provisional**. The corpus annotations are human produced, but the 63-item selection was screened by an AI assistant and has not been independently checked by the student. The [review sheet](QASRL_REVIEW.csv) has the exact sentences and spans for that verification. Some second participants are recipients or other non-agent roles rather than strict semantic patients; the actor measure is cleaner. The original corpus may overlap the models' pretraining, and the forced-choice format differs from the constructed-sentence task.

## Internal readout

A separate [fixed readout](INTERNAL_PROTOCOL.json) examined the 1.5B model's final prompt position after blocks 0, 4, 8, 12, 16, 20, 24, and 28 on the same 480 prompts. At each point, a fixed final norm and output head compared candidate actor and patient name scores. No probe was fitted and no weights were changed. The [raw margins](INTERNAL_RESULT.json) and [summary](INTERNAL_SUMMARY.json) are saved. Final-layer margin signs agreed with the saved behavioral answers on **480/480** prompts; the largest numerical margin difference between the two runs was below 0.00009.

In the failing active/generic condition, the readout favored the actor on **62/120** prompts after block 20, **6/120** after block 24, and **10/120** at the final block. The other conditions were also near 50% at block 20 and separated sharply later. Thus this readout does **not** show a consistently accessible correct role in the middle of the model under the failing prompt. It shows a late, prompt-dependent change in candidate-name preference. A logit lens is an approximate readout, and neither it nor the earlier whole-state patches isolates an abstract actor/patient mechanism.

## Scientific boundary and next experiment

The confirmed LLM result is **model-specific evaluation instability**: equivalent actor questions and sentence forms can change answers dramatically, and the failure pattern differs between two local Qwen2.5 sizes. The authentic test adds an option-order sensitivity that must be controlled in future role evaluations. These results make RQ3 substantive without assuming that the small Transformer's graph-rewiring effect or ordered-role repair transfers to a pretrained LLM.

Next, the student should independently verify the authentic labels. A second model family and a new set of linguistic constructions would test the scope of the behavioral result. To make a mechanistic claim, a role signal must generalize across held-out names and constructions with actor position balanced, followed by a selective intervention. A matched LLM training-repair experiment is a separate step after those checks.

![Behavior and readout overview](RQ3_OVERVIEW.png)
