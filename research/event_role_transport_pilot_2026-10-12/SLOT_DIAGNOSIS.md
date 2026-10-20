# Held-out slot output versus direct role answers

**12 October 2026. Post hoc diagnostic.** The 36-event lexical bank and direct QA results had already been inspected. This diagnostic used the same Qwen2.5-0.5B checkpoints without further training. It did not access the reserved human-review challenge.

Each model received a held-out event in each of four forms. The structured prompt used in training ended with `Actor:`; the model greedily generated the rest of `Actor: NAME; Patient: NAME`. A strict match required both names in the right roles. The direct QA comparison requires both *held-out* actor and patient question wordings to be answered correctly on that same event-form. All 576 generated tuples parsed; 2,880 direct QA answers were valid candidate names.

| Start | Arm | Object-cleft exact tuple /36 | Object-cleft direct both /36 | Active exact tuple /36 | Active direct both /36 |
|---|---|---:|---:|---:|---:|
| fresh1 | Slot SFT | **27** | 2 | 23 | 4 |
| fresh1 | Transport | 24 | 1 | 24 | 4 |
| fresh3 | Slot SFT | **24** | 2 | 25 | 5 |
| fresh3 | Transport | 21 | 2 | 23 | 4 |

For the ordinary slot-supervision arm, **26/36** fresh1 and **22/36** fresh3 object-cleft items had a correct generated tuple while at least one direct held-out role answer was wrong. The same mismatch also occurs for active items (22/36 and 23/36). Thus the failure is not unique to cleft syntax. The additional transport losses did not resolve it and slightly reduced object-cleft exact tuple accuracy.

Candidate-relative scoring independently preferred the correct object-cleft tuple in 22–27/36 cases across arms and starts. With the sentence replaced by `[omitted]`, this preference was exactly 18/36 in every arm/start. This control argues against a *global* correct-answer preference from candidate order alone, though it does not exclude item-level lexical or prompt effects.

## What the gap means

The same checkpoint can often **produce both correct roles in a trained structured format** while failing to answer two differently worded role questions about the same sentence. The result locates an important *behavioral* gap between structured role output and direct answer transfer. It does **not** prove that the model has a stable internal event representation. The structured slot prompt appeared during training, whereas the two direct question wordings were held out; the tasks also differ in output length and context. The result therefore supports a narrower conclusion about transfer across response formats and wordings.

One immediate two-stage system could parse the generated tuple and return the requested role. On these same items its both-role correctness would equal the exact-tuple numbers above. That would be an inference-time bridge, not a new training result, and it costs an additional generation step. It is useful as a diagnostic upper path for this particular prompt, not a publishable repair by itself.

## Next decisive experiment

Train a *slot-conditioned answer* format that requires the model to produce the two roles and then answer the question, and compare it with identical QA and slot labels presented as separate examples. Hold out crossed combinations of sentence construction and question wording, plus new names/verbs, from the start. A positive result would require the slot-conditioned format to raise direct or two-stage exact role accuracy without degrading ordinary forms; a negative result would show that simply exposing an intermediate tuple is insufficient. This is still exploratory until confirmed on human-verified natural sentences and a second model family.

If a one-step direct answer is essential, a later model could train an explicit role readout and a question-to-role router, with the readout shared across surface forms and role reversal. That is a distinct architecture experiment and should be judged against a parameter-matched role-label baseline.

This diagnostic alone is **not** a novelty claim. [Mehta et al. (2024)](https://aclanthology.org/2024.naacl-long.7/) already show that independently prompted linguistic structure components can be inconsistent and that structural inference can help. [Raghav and Jana (2025)](https://aclanthology.org/2025.ijcnlp-srw.21/) evaluate LLM semantic roles through QA, and [QASem Parsing (2022)](https://aclanthology.org/2022.emnlp-main.528/) studies structured QA-based semantic representations. Our potential contribution would need a stronger, controlled role-transfer intervention and new evidence on held-out transformations and natural language.

## Audit

`SLOT_CHECK_PROTOCOL.json`, `slot_check.py`, and `SLOT_*.json` record the candidate-relative diagnostic and sentence-omitted control. `SLOT_GENERATE_PROTOCOL.json`, `slot_generate.py`, and `GENERATE_*.json` record exact free generation. The original direct QA predictions are in `EVAL_*.json`; all checkpoints and frozen training results are unchanged.
