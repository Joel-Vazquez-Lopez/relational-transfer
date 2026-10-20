# Fixed-field answer continuation: adaptive exploratory follow-up

**13 October 2026.** This no-training follow-up was chosen **after** seeing the failed explicit-tag screen, so it is an adaptive exploratory analysis on the **same 40 nonsealed development items**. Its `PROTOCOL.json` and evaluator code were frozen before this follow-up's inference. Actor/patient labels are automatically mapped QA-SRL labels without independent human review. Neither sealed set was evaluated.

The role-first model's original output contains printed `Actor`, `Patient`, and `Answer` fields. For each existing ordinary prompt, the evaluator cached the model's printed actor and patient fields, prefixed the assistant continuation through `Answer:`, and generated only the final answer. It crossed the unchanged question versus an explicit `Requested role` tag with original versus swapped cached fields. Sentence, question wording, candidates, checkpoint, and decoding stayed fixed. The ordinary/original continuation reproduced the prior full-generation answer **160/160** in both starts, validating replay for this condition.

| Start | Cached fields | Ordinary final-answer exact | Explicit-tag final-answer exact | Explicit-tag minus ordinary |
|---|---|---:|---:|---:|
| fresh1 | Original | 88/160 | 97/160 | +9 |
| fresh3 | Original | 85/160 | 98/160 | +13 |
| fresh1 | Swapped | 77/160 | 62/160 | −15 |
| fresh3 | Swapped | 81/160 | 59/160 | −22 |

The original cached requested-role field itself matches the provisional label in **128/160** and **125/160** prompts. Among those prompts, the tag improves exact final answers from **73/128 to 88/128** and **72/125 to 90/125**. Among prompts where that cached field is wrong, exact answers fall from **15/32 to 9/32** and **13/35 to 8/35**. Thus, the cue makes the final answer follow the requested printed field more often, but this helps accuracy only when the field is correct. This differs from the full-generation tag test: when the tag is given *before* the fields are generated, field quality falls and the direct-answer benefit disappears.

Swapping the cached role fields changes final answers in **64/160** and **57/160** ordinary prompts, and **80/160** and **79/160** tagged prompts. Under the tag, final-answer agreement with the requested printed field is **109/160 versus 109/160** in fresh1 for original versus swapped fields, and **115/160 versus 103/160** in fresh3. Accuracy falls when the visible fields are swapped, although it does not collapse completely: the model sometimes retains a sentence- or question-based answer. This is evidence that the **visible text fields influence output**; it is not evidence that the same causal path or an abstract role state operates during unconstrained generation.

**Interpretation.** The experiments separate two coupled problems. In the full-generation setting, a new role cue changes the fields as well as the answer and does not improve accuracy reliably. With the fields fixed, the cue improves selection of a correct requested field, but also encourages following an incorrect one. A training method would therefore need to preserve or improve role extraction and answer selection together. This follow-up does not establish a natural-language repair, generalization beyond these items, or a novel internal mechanism.

`ANALYSIS.json` contains paired rescue/harm and field-following counts. `EVAL_*.json` retains all 1,280 continuations and provenance hashes. The 40 items and two starts are shared across cells; they are not independent replications. Field swapping changes the coherence of the printed explanation relative to the sentence, so it is a behavioral intervention on text, not a surgical intervention on internal computation.
