# Scientific report — frozen pretrained diagnostic, 15 October 2026

## Decision

We have an exploitable extraction-versus-answer gap in untouched Qwen2.5-1.5B on this small natural-language development subset. We do not yet have persuasive evidence that withholding the question improves extraction across model sizes. Do not launch a repair-training sweep or hidden-state mechanism claim on this evidence.

Completed 240 primary generations and 96 follow-up generations, plus 10 technical smoke generations. Both language models were untouched local pretrained instruction weights; no adapters or LLM optimization were used. The secondary question selector was fit on the same separate training questions as the earlier typed-goal study. All evaluation data were already development data.

## Primary outcomes

| Model | Direct original QA: exact /48 | Query-visible fields + generated answer: exact /48 | Query-blind fields + deterministic routing: exact /48 |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 17 | 1 | 8 |
| Qwen2.5-1.5B | 19 | 2 | 31 |

| Model | Direct QA: both answers /24 | Query-visible generated answers: both /24 | Query-blind routed answers: both /24 |
|---|---:|---:|---:|
| 0.5B | 0 | 0 | 2 |
| 1.5B | 2 | 0 | 11 |

The 1.5B blind pipeline improves individual exact answers by 12/48 (25 percentage points) and both-answer accuracy by 9/24 (37.5 points) over direct QA. The descriptive paired-event bootstrap interval for the latter is [16.7,58.3] points. This interval is development-only, does not adjust for several events sharing source documents, and does not establish cross-model or syntactic transfer.

In 0.5B, the pipeline loses individual accuracy despite a small joint-answer improvement. This rules out a general success claim from this screen.

The initial structured condition omitted the final Answer field in 46/48 responses for 0.5B and 45/48 for 1.5B. It is unsuitable for diagnosing answer-use computation without accounting for schema failure. Many counted correct-field/wrong-answer cases are missing fields, not explicit contradictory answers.

## Remove the extra-output confound

A separately frozen follow-up requested only Subject and Object fields, keeping the extraction system instruction, output prefix, model, target predicate and decoding identical to the blind condition. It added the original question to the user input. This follow-up was designed after the primary results and evaluated the same development events; it is not independent confirmation.

| Model | Question visible, fields only: exact /48; both /24 | Question blind: exact /48; both /24 | Blind-minus-visible joint accuracy |
|---|---:|---:|---:|
| 0.5B | 12; 3 | 8; 2 | −4.17 pp; interval [−16.67,+8.33] |
| 1.5B | 26; 9 | 31; 11 | +8.33 pp; interval [−8.33,+25.00] |

In 1.5B, blind extraction gains three joint-correct events and loses one relative to visible extraction. In 0.5B, it gains one and loses two. The effect is uncertain and does not repeat in direction across sizes.

The printed tuple changes between the two questions on 8/22 fully parsable event pairs for 1.5B, and 8/14 for 0.5B. This establishes output sensitivity under these prompts, not its cause or harmfulness. A change can correct an earlier error. Blind extraction is reused, so its consistency is guaranteed by construction and cannot be presented as learned invariance.

## Routing without gold role selection

A post hoc secondary analysis used the pre-existing character-TFIDF/logistic classifier recipe, fit solely on 160 training events. It correctly classified all 48 original questions here, preserving the routed results. This reduces dependence on oracle question slots for these examples. It does not establish general question interpretation: 18/48 evaluation question strings occurred verbatim in the classifier's training questions, and the question grammar is restricted.

With deterministic routing, the original query-visible condition reaches 27/48 individual and 6/24 joint answers in 1.5B; the field-only version reaches 26/48 and 9/24. The apparent gain from blind extraction is therefore much smaller after controlling how the final answer is produced and how many output fields are requested.

## Data correction

The earlier 100-event candidate-free sample included prepositional questions labeled patient, for example `Who did someone finish behind?` Such original QA-SRL questions can be legitimate while the semantic patient remapping is unsupported. Exactly 56 events passed the new simple active subject/direct-object question-slot filter; the diagnostic used the first 24 in the previous deterministic ordering. The excluded 44 are not all asserted to be annotation mistakes.

The present task is grammatical subject/direct-object question answering, not verified semantic agent/patient classification. All selected events are active and subject-first. Results cannot show passive robustness, participant-order robustness, or novel entity-pair transfer. Original consensus answer spans and alternative consensus spans were used. A prediction-blind semantic review packet is available.

## What is justified next

1. Retain the untouched 1.5B extraction/answer discrepancy as a development lead. Maintain separate accounts of extraction errors, question-role interpretation, schema compliance and final selection.
2. Before training, validate a balanced natural-language evaluation and equivalent prompt tasks. Use original QA annotations where possible. Verify semantic role assignments and any transformed sentences independently.
3. For question-conditioning tests, compare identical field-only outputs with meaningful versus irrelevant, length-controlled question context. Include errors corrected by the question, not just changes. Confirm on new development documents before proposing a repair.
4. Use a conventional extraction-plus-router pipeline as a strong baseline. Its improvement alone is not a novel method. A new intervention must improve independently verified transfer beyond this baseline and matched data/format controls.
5. Test a meaningfully different pretrained family before treating this as a general LLM phenomenon. Hidden-state causal work is justified only after the behavioral failure and measurement are stable.

The proposed causal interference mechanism remains unestablished. This narrows the research direction and prevents a format-driven performance difference from becoming a mechanistic claim.

## Provenance and reproduction

Primary protocol SHA256: `6f5623f0f2cf4be0d6b4b07db7b158292c3c82a83adcb867535647a9ecc0f214`.

Follow-up protocol SHA256: `cd163fbf884c8eac8a249e0b0d9938b6ee56ee28b9b0490856fa18699fdd538e`.

Read README.md, SMOKE_NOTES.md and DATA_AUDIT.md. Original protocols and prediction files are preserved. Analysis source timing is recorded in ANALYSIS_FREEZE.json and FIELD_ANALYSIS_FREEZE.json; the latter was written during follow-up execution, before inspecting completed larger-model outputs. These are transparent development analyses, not a claim of preregistered confirmation.

Commands with the existing ai-fingerprint Python environment:

```
python run.py run --model 0.5b
python run.py run --model 1.5b
python analyze.py
python secondary_routing.py
python field_only_ablation.py run --model 0.5b
python field_only_ablation.py run --model 1.5b
python analyze_field_only.py
```
