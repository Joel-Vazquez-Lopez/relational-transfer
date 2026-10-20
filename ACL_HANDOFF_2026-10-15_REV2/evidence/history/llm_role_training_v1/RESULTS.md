# First local language-model training pilot

**Completed 7 October 2026. Exploratory; no LLM repair demonstrated.** The [protocol](PROTOCOL.json) was fixed before any of these three runs. The [runner](runner.py), [analysis script](analyze.py), and [machine-readable results](ANALYSIS.json) preserve the prompts, raw responses, and every scored item.

## Question and matched comparison

Can the ordered-role auxiliary training that helped in the [controlled Transformer study](../role_transfer_phase01/role_aux_confirmation_v1/RESULTS.md) improve a locally fine-tuned open-weight language model? This is a first feasibility test, not a direct replication of the synthetic result. It uses Qwen2.5-0.5B-Instruct, one 10-name relation graph, one training start, and constructed active/passive sentences.

Both fine-tuned arms received the **same 480 sentence/question/answer examples** from 20 observed name pairs, including both directions, three verbs, two voices, and actor and patient queries. The model and main answer targets were identical. A rank-four LoRA adapter was trained for 480 updates. The auxiliary targets differed: one arm predicted the two participant identities without order; the other predicted actor and patient identities in order. Both arms started with the **same trainable parameter hash**. An untouched pretrained model was evaluated as a reference.

The evaluation used every one of 45 name pairs in both directions, three verbs, two voices, two queried roles, and two question wordings: **2,160 prompts per arm**. The 25 withheld pairs account for 1,200 of these prompts. They were absent from this fine-tuning set, though nothing here establishes that related language was absent from pretraining. All generations produced a parseable one-name answer.

## Results

| Model | Observed pairs, 960 prompts | Withheld pairs, 1,200 prompts |
|---|---:|---:|
| Pretrained, no fine-tuning | 56.25% | 56.50% |
| Unordered-identity auxiliary | 55.94% | 56.75% |
| Ordered-role auxiliary | 58.44% | 57.83% |

The ordered arm exceeded the unordered arm by **1.08 percentage points** on withheld pairs (694 versus 681 correct), and the untouched model by 1.33 points. The ordered-minus-unordered difference was positive on **16 of 25** withheld pairs, zero on two, and negative on seven. An exploratory pair-resampling 95% interval is **−0.50 to +2.58 points**. With one graph and one training start, these data do not support a robust LLM improvement claim.

Prompt behavior is a stronger finding in this pilot. On withheld pairs, active/passive versions gave the same answer in **266/600** matched cases for ordered-role training, versus **305/600** for unordered training; both versions were correct in 180 versus 193. Reworded questions gave the same answer in **310/600** cases for ordered training, versus **269/600** for unordered training; both were correct in 202 versus 175. Thus ordered training improved wording agreement here but worsened voice agreement. Neither form of agreement is close to sufficient reliability.

| Withheld-pair subset | Unordered | Ordered |
|---|---:|---:|
| Active, actor, generic question | 99.3% | 93.3% |
| Active, actor, verb-specific question | 28.0% | 25.3% |
| Active, patient, generic question | 26.0% | 48.7% |
| Active, patient, verb-specific question | 100.0% | 100.0% |
| Passive, patient, generic question | 0.7% | 0.7% |

The large differences among semantically equivalent ways of asking show why average accuracy alone would conceal the failure mode. The model can answer one wording almost perfectly and another almost never, even when both ask for the same participant. That pattern is present in both trained arms. The low passive-patient result also cautions against interpreting the small average gain as relational understanding.

## Interpretation and next decision

The local training and full evaluation are feasible on this machine. The **controlled-model role-training effect did not transfer convincingly** under this first language-model setup. A later LLM study needs an evaluation with more independent relation graphs, training starts, names, and verbs; it should score voice and wording consistency as outcomes rather than rely on mean accuracy. It should also inspect whether the auxiliary role decoder actually learned its task. The current answers do not establish any change to the LLM's internal role representation.

This pilot gives a concrete evaluation finding: prompt form can dominate the measured role answer. It also suggests that the controlled internal-state mechanism and an LLM training repair should be treated as separate claims until a stronger matched experiment connects them.
