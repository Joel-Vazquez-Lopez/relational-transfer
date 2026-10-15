# Can earlier ordinary fine-tuning explain the auxiliary gains?

## Question and fixed design

The preceding [three-arm result](../llm_role_unordered_control/RESULTS.md) showed that ordered-role and participant-only auxiliary training retain more accuracy on patient-fronted clefts than ordinary answer fine-tuning, but lose a little accuracy on familiar active/passive forms. This experiment tests whether simply stopping ordinary fine-tuning earlier reaches the same trade-off.

The [protocol](PROTOCOL.json) was saved before these trajectory runs. We repeated ordinary fine-tuning on the same three graph layouts and two starts per layout, with the same examples, learning rate, optimizer, model, and 120-update order as the earlier ordinary arm. At updates **20, 40, 60, 80, 100, and 120**, we saved an adapter and scored all 25 held-out pairs, both actor directions, five verbs, and active/passive/patient-cleft forms using the fixed “Who performed the action?” wording. That is 250 patient-cleft and 500 familiar prompts per checkpoint and graph/start. All six 120-update adapters exactly matched the earlier ordinary adapters, so pausing for evaluation did not change the training endpoint.

The checkpoint selection used **fresh1 and fresh2 only**, two starts each. For each auxiliary comparison, an ordinary checkpoint had to meet that arm's mean familiar-form accuracy. Among eligible checkpoints, the rule picked the one with highest patient-cleft accuracy, breaking ties toward fewer updates. Both comparisons selected **80 updates**. The chosen checkpoint was then tested on fresh3 with the two other question wordings, which were excluded from selection.

## Development trajectory

| Ordinary update | Patient cleft, fresh1+fresh2 | Familiar active/passive, fresh1+fresh2 | Meets ordered familiar floor? | Meets participant-only familiar floor? |
|---:|---:|---:|:---:|:---:|
| 20 | 67.4% | 98.25% | No | No |
| 40 | 41.6% | 100.0% | Yes | Yes |
| 60 | 44.5% | 100.0% | Yes | Yes |
| **80** | **45.4%** | **100.0%** | **Yes** | **Yes** |
| 100 | 44.2% | 100.0% | Yes | Yes |
| 120 | 43.2% | 100.0% | Yes | Yes |
| Ordered-role auxiliary, 120 | 67.5% | 99.8% | — | — |
| Participant-only auxiliary, 120 | 74.5% | 98.95% | — | — |

The 20-update ordinary checkpoint keeps much more cleft accuracy than later ordinary checkpoints, but its familiar accuracy misses both auxiliary floors. Every sampled ordinary checkpoint that meets those floors has substantially lower cleft accuracy than the corresponding auxiliary arm on these development graphs. This does not exclude a better ordinary checkpoint between sampled steps, especially **20–40**, where the steepest change occurs.

## Selected-checkpoint check on unused wordings

Patient-cleft accuracy is out of 500 and familiar-form accuracy out of 1,000 per start. The rows are identical across arms and use only the “did” and “carried out” question wordings.

| fresh3 start | Ordinary, 80 updates | Ordered roles, 120 updates | Participant-only, 120 updates |
|---|---:|---:|---:|
| 830211, patient cleft | 314/500 | 421/500 | 458/500 |
| 830211, familiar forms | 1000/1000 | 982/1000 | 979/1000 |
| 830212, patient cleft | 358/500 | 398/500 | 471/500 |
| 830212, familiar forms | 1000/1000 | 964/1000 | 947/1000 |
| **Both starts, patient cleft** | **67.2%** | **81.9%** | **92.9%** |
| **Both starts, familiar forms** | **100.0%** | **97.3%** | **96.3%** |

The selected ordinary checkpoint does **not** reproduce either auxiliary arm's cleft accuracy on the unused wordings. Each auxiliary arm improves cleft accuracy while giving up some familiar accuracy; none of these three fixed settings is better on both measures. The participant-only arm has the highest cleft score and lowest familiar score among them. This makes a **retention-versus-adaptation trade-off** a stronger empirical description than a unique ordered-role repair.

## Limits and next falsification

This rules out the simple explanation that **one of these six sampled ordinary checkpoints** already gives the same development trade-off. It does not rule out a checkpoint at 25, 30, or 35 updates, a different learning rate, or another ordinary regularizer. The held-out checkpoint choice used fresh1/fresh2, but the fresh3 graph's final-arm scores were known before this follow-up was designed. These are synthetic templates in one small model family; no natural-language or cross-model repair follows from them.

The next economical falsification is a **dense 20–40 update sweep** with the same familiar-accuracy-floor rule. If an ordinary checkpoint in that interval matches the auxiliary cleft score at its familiar accuracy, the auxiliary benefit may be explainable by training duration. If the gap remains, we should test a second model and authentic role-labeled sentences before making a broader claim.

The [machine-readable analysis](ANALYSIS.json) verifies every matched test row, the frozen selection, all six trajectory endpoints, and the reserved-wording comparison. [choose_checkpoint.py](choose_checkpoint.py) and [confirm_selected.py](confirm_selected.py) reproduce the selection and final scoring from the saved predictions and adapters.

**Completed follow-up:** The separately frozen [dense 25/30/35 update sweep](../llm_role_dense/RESULTS.md) tested the steep interval and selected ordinary checkpoints at 25 and 30 updates. Neither reproduced its matched auxiliary arm's patient-cleft accuracy on the reserved wordings. The limitation concerning untested steps and learning rates remains.
