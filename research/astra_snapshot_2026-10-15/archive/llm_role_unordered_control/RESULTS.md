# Participant-only auxiliary control

## Why this control was run

The [frozen six-run confirmation](../llm_role_repair_confirmation/RESULTS.md) established that adding ordered actor/patient labels during fine-tuning improves accuracy on held-out patient-fronted clefts relative to ordinary fine-tuning. It did **not** establish that semantic-role order caused the improvement. This follow-up tests that narrower explanation.

The new arm has the **same two classification heads**, model, initial weights, 240 training examples, 120 updates, auxiliary-loss weight (0.5), and 3,000 test prompts in each of the six graph/start units. Its heads predict the two people in fixed ID order (lower, higher), regardless of who acts. The language-model answer objective is unchanged. This teaches participant identity but gives no actor/patient order as an auxiliary target. The [follow-up protocol](PROTOCOL.json) was saved before its training began. The control was designed after the ordered result was inspected, so it is a follow-up explanation test, not part of the original frozen replication.

## Matched results

Patient-fronted cleft accuracy is out of 750 per unit; familiar active/passive accuracy is out of 1,500.

| Graph / start | Ordinary cleft | Participant-only cleft | Ordered-role cleft | Ordinary familiar | Participant-only familiar | Ordered-role familiar |
|---|---:|---:|---:|---:|---:|---:|
| fresh1 / 830211 | 295 | 618 | 619 | 1500 | 1488 | 1490 |
| fresh1 / 830212 | 332 | 619 | 471 | 1500 | 1420 | 1457 |
| fresh2 / 830211 | 397 | 543 | 558 | 1499 | 1449 | 1470 |
| fresh2 / 830212 | 389 | 616 | 572 | 1499 | 1438 | 1454 |
| fresh3 / 830211 | 422 | 679 | 614 | 1500 | 1479 | 1482 |
| fresh3 / 830212 | 483 | 697 | 565 | 1500 | 1446 | 1463 |
| **Mean accuracy** | **51.51%** | **83.82%** | **75.53%** | **99.98%** | **96.89%** | **97.96%** |

Participant-only supervision beats ordinary fine-tuning on the cleft test in **all six** units. It beats ordered-role supervision in **four** units; ordered wins in two, by one answer in one start and by 15 in the other. The participant-only mean cleft advantage over ordered training is **8.29 percentage points**. The cost is lower familiar-form accuracy: **96.89%** versus **97.96%** ordered and **99.98%** ordinary. On active, passive, and actor-fronted clefts separately, participant-only scores **98.31%**, **95.47%**, and **99.18%**; ordered scores **99.24%**, **96.67%**, and **99.89%**. The cleft benefit trades off against these easier forms.

The untouched model averaged **86.49%** on patient-fronted clefts and **67.80%** on familiar active/passive prompts. Participant-only training therefore nearly restores, but on average does not exceed, the untouched cleft score while greatly improving familiar-form accuracy. It exceeds the untouched cleft score on the two third-graph starts, but this is not consistent across layouts.

## Interpretation

The control **falsifies the narrow claim that ordered actor/patient auxiliary targets are required** for the observed protection against cleft regression. Predicting which participants are present, without assigning their roles in the auxiliary labels, produces a comparable or larger cleft gain in this experiment. It does not prove the model ignores roles: the main answer loss still trains actor answers, and the underlying pretrained model already has role knowledge. It also does not isolate whether the useful factor is participant identity, extra gradient signal, regularization, or preservation of pretrained behavior. Those require further controls, such as matched auxiliary tasks with non-participant labels or altered loss weights.

The more defensible result is a **fine-tuning trade-off**: ordinary answer training makes familiar forms nearly perfect while damaging a patient-first form; auxiliary supervision retains more of the pretrained model's performance on that form, with a small familiar-form cost. Ordered-role labels have a somewhat better familiar-form score, while participant-only labels have a higher cleft score at the tested weight. These are one-model, synthetic-template results, not a demonstrated repair for natural language or larger LLMs.

The [analysis file](ANALYSIS.json) and [analysis script](analyze.py) verify that the six new runs use identical test rows, matched adapter and head initializations against the ordered arm, and the same training count. Full first-token predictions and training diagnostics are saved in each run's `result.json`. The control source and original experiment sources match the hashes frozen in the follow-up protocol. No independent-prompt p-value is claimed because prompts within each graph and event are related.
