# Authentic-sentence transfer check: completed exploratory result

All 19 evaluations completed: one untouched Qwen2.5-0.5B-Instruct model and three existing fine-tuning arms across three graph/name worlds and two matched starts. Each evaluation used the same 63 selected QA-SRL Gold source sentences and 504 prompts (actor/patient × two wordings × both A/B option orders). The frozen protocol and every result file were validated by `llm_role_authentic_transfer.py analyze`. Raw per-prompt records and source hashes are retained in this directory.

## Primary and secondary scores

| Condition | Units | Actor correct | Patient correct | All eight prompts correct for an event |
| --- | ---: | ---: | ---: | ---: |
| Untouched pretrained | 1 | 168/252 (66.7%) | 104/252 (41.3%) | 0/63 |
| Ordinary fine-tuning | 6 | 1125/1512 (74.4%) | 548/1512 (36.2%) | 0/378 |
| Ordered-role auxiliary | 6 | 1059/1512 (70.0%) | 591/1512 (39.1%) | 0/378 |
| Participant-only auxiliary | 6 | 1068/1512 (70.6%) | 579/1512 (38.3%) | 0/378 |

Matched against ordinary fine-tuning, ordered-role training changes actor accuracy by **−4.37 percentage points** and patient accuracy by **+2.84 points**, averaged equally over six graph/start units. Participant-only training changes them by **−3.77** and **+2.05 points**. Ordered-role actor accuracy is lower in all six units; its patient accuracy is higher in all six. These signs describe this sample and should not be treated as six independent datasets or a significance test.

## Post-hoc option-order diagnostic

The 504 prompts have exactly 252 gold A and 252 gold B answers. Nevertheless, the model often answers B: 438/504 (86.9%) for untouched Qwen; 2241/3024 (74.1%) for ordinary; 2474/3024 (81.8%) for ordered roles; and 2431/3024 (80.4%) for participant-only. All outputs were valid A/B tokens.

Requiring **both option orders** to be correct for the same sentence, role, and wording exposes the problem:

| Condition | Actor: both orders correct | Patient: both orders correct |
| --- | ---: | ---: |
| Untouched pretrained | 42/126 (33.3%) | 1/126 (0.8%) |
| Ordinary fine-tuning | 418/756 (55.3%) | 50/756 (6.6%) |
| Ordered-role auxiliary | 328/756 (43.4%) | 16/756 (2.1%) |
| Participant-only auxiliary | 341/756 (45.1%) | 23/756 (3.0%) |

The auxiliary arms' apparent patient gain comes with **worse option-order stability**. A stronger B preference can raise accuracy when the patient is B while lowering it when the patient is A. The result therefore **does not support a natural-sentence role-repair claim**. The robust conclusion from this run is that this A/B test is strongly sensitive to option placement, and that the synthetic gain has not yet transferred to a reliable authentic-sentence measure.

## Limits and next experiment

These 63 sentences were used in an earlier, separate Qwen2.5-1.5B pilot. Their selection and prior results were already known; this run is exploratory. The QA-SRL annotations and AI-screened candidate selection have not been independently checked by the student; the blank `HUMAN_REVIEW.csv` records that required review. Natural sentences are longer, contain more participants, and use A/B answers rather than the name-output objective used in fine-tuning. The small 0.5B model's severe A/B preference may reflect that format as well as role understanding. Pretraining overlap is unknown. No internal-mechanism claim follows from these scores.

The next evaluation should remove the A/B position cue: use independently reviewed sentences, ask for the original annotated answer span, and score that span with a fixed exact-match rule. A separate candidate-likelihood comparison can be a diagnostic, but it must control for span length and candidate prior. Freeze those choices before running the saved adapters again. Only then does a second model test become informative for transfer.

Files: `PROTOCOL.json` and `ITEMS.json` freeze this run; `ANALYSIS.json` contains the prespecified summary; `DIAGNOSTICS.json` and `diagnose_results.py` contain the explicitly post-hoc option-order audit. Raw result JSON files contain every answer.
