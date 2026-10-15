# Matched local LLM role-training pilot (10 October 2026)

## Design and audit

The [protocol](PROTOCOL.json), [60 training groups](TRAIN_GROUPS.json), and [5,400 test prompts](TEST_ROWS.json) were saved before version 3 training. The model was Qwen2.5-0.5B-Instruct. A ten-name graph supplied 20 observed and 25 withheld pairs. All trained arms received the **same 240 distinct examples** (20 pairs × 3 verbs × 2 actor directions × active/passive), presented for two epochs in the same order (120 group updates). The four arms had identical initial adapter weights, model, optimizer, examples, and test items. Raw answers, adapter weights and loss traces are in each arm's `result.json` and `adapter.pt` files. All 5,400 outputs in every trained arm were one of the two names.

The arms were ordinary supervised fine-tuning; the same task with active/passive answer-distribution consistency; the same task with ordered actor/patient auxiliary heads; and ordered heads plus both voice consistency and role-swap equivariance. The untouched pretrained reference used the identical test items. Its raw result is in [version 2](../llm_role_repair_v2/pretrained/result.json). Version 2 **trained** arms are [invalidated](../llm_role_repair_v2/INVALIDATED.md) because training omitted attention masks for left-padded batches. Version 3 corrected this before any of its training runs. A masked batched prompt and the corresponding unpadded prompt then agreed in candidate logits to within 0.00002.

## Held-out-pair results

All scores below concern the **25 pairs absent from fine-tuning**. “Familiar forms” means active/passive; “new forms” means actor-fronted or patient-fronted clefts. The two new verbs were also absent from fine-tuning. All three actor-question wordings are included; only “Who performed the action?” appeared during fine-tuning.

| Condition | Seen verbs, familiar forms /900 | Seen verbs, new forms /900 | New verbs, familiar forms /600 | New verbs, new forms /600 | All held-out prompts /3,000 |
|---|---:|---:|---:|---:|---:|
| Untouched model | 560 | 736 | 450 | 544 | 2,290 |
| Ordinary fine-tuning | **900** | 699 | **600** | 490 | 2,689 |
| Generic consistency | **900** | 692 | **600** | 485 | 2,677 |
| Ordered roles | 892 | 792 | 588 | 551 | 2,823 |
| Ordered roles + transformations | 894 | **801** | 592 | **554** | **2,841** |

Ordinary fine-tuning already answers **900/900** primary prompts correctly on withheld pairs with familiar verbs and forms. Ordered-role training cannot be credited with repairing that primary case; it makes 8 errors there. The substantive difference appears in the new sentence forms. The combined arm exceeds ordinary fine-tuning by **166/1,500 = 11.1 percentage points** across both cleft forms, while losing 14/1,500 on familiar forms across seen and new verbs. Generic active/passive consistency does not produce that transfer gain.

The new-form gain is concentrated on the **patient-fronted cleft**, as in “It was Alexander whom Stephen chased.” The fronted name is the patient, while Stephen is the actor. On all 750 held-out-pair prompts in this form:

| Condition | Correct /750 | Events correct under all three questions /250 |
|---|---:|---:|
| Untouched model | **614** | **172** |
| Ordinary fine-tuning | 439 | 118 |
| Generic consistency | 427 | 113 |
| Ordered roles | 593 | 159 |
| Ordered roles + transformations | 605 | 165 |

Ordered roles rescue 179 ordinary errors and introduce 25 errors on this form, a net gain of **154/750 = 20.5 points**. The gain is positive on 21 of the 25 withheld pairs, zero on two and negative on two. The combined arm gains another 12 correct answers over ordered roles, but that small increment is not a demonstrated benefit of the role-swap constraint specifically: this arm adds both voice and swap losses. Importantly, the untouched model still scores 9/750 above the combined arm on this form. The result is therefore best described as **role-aware supervision mitigating a transfer regression caused by ordinary fine-tuning**, while preserving most of the gains fine-tuning brings to familiar forms.

## Interpretation and limits

This is an **exploratory one-graph, one-start, one-model-family pilot**, not a confirmed LLM repair or an internal-mechanism result. The 25 withheld pairs share the same ten names and training graph; they are not 25 independent training replications. The cleft constructions are grammatical but constructed, and the model's pretraining exposure is unknown. The training objective's ordered-head loss remains nonzero at the end, so we cannot claim a fully learned internal role representation. The combined arm is only modestly better than ordered heads alone.

The result changes the next experiment: preregister the patient-fronted transfer contrast and its familiar-form trade-off; replicate on new relation graphs, independent initializations and a second model family; then use independently checked natural examples. Mechanistic claims require a role signal that generalizes across names and forms plus a selective intervention. The present evidence supports a narrower behavioral hypothesis: explicit ordered-role supervision can protect some unseen syntactic forms from a shortcut acquired during task fine-tuning.
