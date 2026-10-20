# Direct QA-SRL span check: completed exploratory result

This is the follow-up to the A/B option-order confound in `llm_role_authentic_transfer/RESULTS.md`. The same 63 previously selected QA-SRL Gold sentences were evaluated with **no A/B options**. Each source sentence has two original QA-SRL questions: one seeking the active subject and one seeking the other direct participant. We use **subject answer** and **other-participant answer** because the latter is not uniformly a semantic patient (for example, recipients of *told* and partners of *married*).

The prompt explains QA-SRL's `someone`/`something` placeholders and asks the model to copy the exact answer phrase. Greedy generation is capped at 18 new tokens. Before the full batch, `PROTOCOL.json` fixed the 126 prompts, exact normalization, models, and comparisons. All 19 evaluations finished and the frozen protocol/results validated. The four- and six-item prompt smoke tests were nonreportable and are stored separately. I also checked all 126 source answer ranges against the original sentence tokens; all matched their recorded spans.

## Exact-span results

| Condition | Matched units | Subject answer | Other-participant answer | Both roles right for one event |
| --- | ---: | ---: | ---: | ---: |
| Untouched Qwen2.5-0.5B | 1 | 13/63 (20.6%) | 14/63 (22.2%) | 1/63 |
| Ordinary fine-tuning | 6 | 80/378 (21.2%) | 79/378 (20.9%) | 1/378 |
| Ordered-role auxiliary | 6 | 79/378 (20.9%) | 84/378 (22.2%) | 1/378 |
| Participant-only auxiliary | 6 | 82/378 (21.7%) | 82/378 (21.7%) | 3/378 |

The equal-unit mean difference from ordinary fine-tuning is **−0.26 percentage points** for subject answers and **+1.32 points** for other-participant answers with ordered-role training. Participant-only training gives **+0.53** and **+0.79 points**, respectively. Each unit has only 63 questions per role; the changes amount to a few individual answers and are not consistent enough to support a transfer claim. The six units share one authentic sentence set and are not six independent datasets.

## Error audit, performed after the full run

| Condition | Exact opposite participant given | Same answer for both roles | Literal `Someone`/`Something` |
| --- | ---: | ---: | ---: |
| Untouched | 32/126 | 41/63 events | 18/126 |
| Ordinary | 180/756 | 239/378 events | 119/756 |
| Ordered-role | 179/756 | 235/378 events | 120/756 |
| Participant-only | 177/756 | 230/378 events | 113/756 |

The low exact score is not solely a strict span-boundary problem. The model frequently supplies the *other* annotated participant or gives the same response to both role questions. Boundary errors are also real: for ordinary fine-tuning, 88 answers are a shorter part of the gold phrase, and 55 include the gold phrase with extra words. Examples in the raw results include `Deputy` for `a Deputy`, but also `Webster` for `A police official` and `Someone` for `she`. These counts are descriptive, post-hoc diagnostics; they are not a revised primary score.

## Interpretation and limits

The direct-answer evaluation removes the A/B position cue, yet the three fine-tuning methods remain near the untouched model and near one another on this selected sample. **We still have no evidence that the synthetic auxiliary-training benefit transfers to authentic role answering.** The earlier controlled synthetic results remain valid within their own task.

This natural test is also hard for the small model. QA-SRL question notation appears to cause confusion even after explanation, and exact copying is demanding for longer spans. The 63 selected items were previously used in another pilot; they are not a fresh blind sample. Their source annotations and span presence were checked in code, and an AI assistant inspected the sentences, but independent student annotation review is still blank in `../llm_role_authentic_transfer/HUMAN_REVIEW.csv`. Do not call these labels independently verified or extrapolate to larger LLMs. Pretraining overlap is unknown.

Next, review the labels and role categories without viewing arm-specific answers, then use a small development set to choose a natural, placeholder-free prompt. Freeze that prompt and evaluate untouched models first as a validity gate; only a task they can perform above the observed floor can meaningfully test whether fine-tuning preserves or changes authentic role knowledge. Keep the option-order failure as a separate evaluation-robustness result.

`ANALYSIS.json` holds the prespecified comparison. `DIAGNOSTICS.json` and `diagnose_results.py` hold the post-hoc error audit. Each model result JSON keeps its exact generated answer for every item.
