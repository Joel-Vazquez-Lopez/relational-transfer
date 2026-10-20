# Fresh-graph confirmation: ordered-role training and patient-first clefts

## Frozen question and design

Does an auxiliary objective that predicts the **ordered actor and patient** reduce the loss of actor accuracy caused by ordinary answer fine-tuning when a held-out pair appears in a patient-fronted cleft (for example, “It was Bob whom Alice chased”)? The answer is evaluated in one local model, Qwen2.5-0.5B-Instruct.

The [protocol](PROTOCOL.json) was saved before these replication runs. It specifies three new 10-name graph layouts, two matched starts per layout, and the same 240 active/passive training examples for both arms. Each layout withholds 25 of 45 unordered name pairs. Both arms receive 120 training updates. The ordered arm adds actor and patient classification losses at the final prompt state. Each trained run answers 3,000 fixed prompts on held-out pairs: two actor directions, five verbs, four sentence forms, and three actor-question wordings. Two verbs, both cleft forms, and two question wordings were absent from fine-tuning.

The **primary measure** is exact actor-name accuracy on the 750 patient-fronted cleft prompts per layout/start. The frozen directional rule requires ordered training to beat ordinary fine-tuning in at least five of six matched units, with an equal-unit mean gain of at least 10 percentage points. A mean loss of more than two points on familiar active/passive prompts is a material trade-off.

## Results

| Graph / start | Ordinary patient cleft | Ordered patient cleft | Gain | Ordinary familiar forms | Ordered familiar forms | Familiar change |
|---|---:|---:|---:|---:|---:|---:|
| fresh1 / 830211 | 295/750 | 619/750 | +43.2 pp | 1500/1500 | 1490/1500 | −0.67 pp |
| fresh1 / 830212 | 332/750 | 471/750 | +18.5 pp | 1500/1500 | 1457/1500 | −2.87 pp |
| fresh2 / 830211 | 397/750 | 558/750 | +21.5 pp | 1499/1500 | 1470/1500 | −1.93 pp |
| fresh2 / 830212 | 389/750 | 572/750 | +24.4 pp | 1499/1500 | 1454/1500 | −3.00 pp |
| fresh3 / 830211 | 422/750 | 614/750 | +25.6 pp | 1500/1500 | 1482/1500 | −1.20 pp |
| fresh3 / 830212 | 483/750 | 565/750 | +10.9 pp | 1500/1500 | 1463/1500 | −2.47 pp |
| **Equal-unit mean** | **51.51%** | **75.53%** | **+24.02 pp** | **99.98%** | **97.96%** | **−2.02 pp** |

The frozen directional criterion is met: **6/6 units positive** and a **24.02-point mean gain**. The familiar-form cost also crosses the prespecified threshold, narrowly: **2.02 points** versus a threshold of more than two. Across the six runs' 150 held-out pair/start combinations, 124 favor ordered training on patient clefts, 18 tie, and eight favor ordinary training. These combinations are correlated and are descriptive, not 150 independent replications.

The untreated pretrained model scored **628/750**, **644/750**, and **674/750** on patient-fronted clefts in the three name sets (mean **86.49%**). Its familiar active/passive scores were **1011/1500**, **1016/1500**, and **1024/1500** (mean **67.80%**). Thus ordinary fine-tuning makes familiar forms nearly perfect but degrades patient-fronted clefts sharply. Ordered-role training protects much of that cleft performance, yet its **75.53%** mean remains below the untouched model's **86.49%**. The intervention is a **partial protection against fine-tuning regression**, not an improvement over the unmodified model.

### Secondary checks

| Form on held-out pairs | Ordinary | Ordered |
|---|---:|---:|
| Active | 4498/4500 (99.96%) | 4466/4500 (99.24%) |
| Passive | 4500/4500 (100.00%) | 4350/4500 (96.67%) |
| Actor-fronted cleft | 4500/4500 (100.00%) | 4495/4500 (99.89%) |
| Patient-fronted cleft | 2318/4500 (51.51%) | 3399/4500 (75.53%) |

The gain is specific to the patient-fronted form, where ordinary training fails despite ceiling-level accuracy on actor-fronted clefts. It is present for each of the three fixed question wordings and for both seen and unseen verbs. On patient-fronted clefts, the ordered-minus-ordinary gains by wording are **+205/900** (“Who performed the action?”), **+240/900** (“Who did the action?”), and **+220/900** (“Which person carried out the action described in the sentence?”) for the three seen verbs; for the two unseen verbs, they are **+148/600**, **+156/600**, and **+112/600**, respectively. Across events, all three wordings were correct on the patient-fronted form for **953/1500** event/start instances with ordered training versus **612/1500** with ordinary training. All 18,000 greedy first-token predictions per trained arm were valid name tokens; the study did not score full generated responses.

## Verification and interpretation

The [analysis](ANALYSIS.json), produced by [analyze.py](analyze.py), checks that all six pairs have the same 3,000 test items, matched initial adapter hashes, 120 updates, and 240 training examples. It checks the frozen protocol hash in every result. The original [raw results](.) include the prompts, answers, correctness, training diagnostics, and saved adapters. The training source uses attention masks for left-padded batches; the earlier v2 run that omitted them remains invalidated and is excluded.

This is strong **within-model, controlled replication** of a specific training interaction. It does not prove that the classifier heads caused the model to encode general semantic roles internally. The added objective and the ordinary objective have different losses, and there is no causal intervention on internal states here. The cleft templates are synthetic; natural sentences, other linguistic relations, larger models, and another model family remain untested. Two starts on each graph share names and edges, so the six units are not fully independent. No p-value is reported for individual prompts because prompts within the same graph, pair, and event are dependent.

The next decisive study should test whether the effect survives **a second model family and natural role-labeled sentences**, while keeping matched ordinary and ordered arms and an untouched baseline. It should also test whether a smaller ordered-loss weight or another training schedule preserves the cleft gain with less passive-form loss. That would establish how far the current result extends and whether the trade-off can be improved.

**Follow-up:** A separately frozen [participant-only auxiliary control](../llm_role_unordered_control/RESULTS.md) also protects patient-cleft accuracy and exceeds the ordered arm on average. Thus the present result does **not** isolate ordered semantic-role supervision as the necessary cause. The original directional comparison and its thresholds remain unchanged.
