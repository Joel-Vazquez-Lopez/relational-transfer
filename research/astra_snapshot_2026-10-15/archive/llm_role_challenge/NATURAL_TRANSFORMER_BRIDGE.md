# Natural-language Transformer bridge

**Status:** design note, 12 October 2026. A first frozen-encoder diagnostic has now been run; see [its results](natural_bridge_probe/RESULTS.md). That diagnostic tested voice exposure, not the ordered-role intervention proposed below. The harder evaluation still requires independent human review.

## Why this bridge is needed

The controlled study trains a small Transformer on an artificial world where the correct roles and observed relations are known exactly. The Qwen studies change several things at once: natural syntax, pretraining, model size, and generated answers. A smaller pretrained Transformer trained to extract roles from natural sentences can test whether role-focused training still helps when the *language* becomes realistic. This does not by itself isolate architecture from pretraining; it narrows the gap between the two studies.

The existing natural Qwen continuation is already evidence of partial transfer: on a 130-event QA-SRL Bank test, candidate role contrast improved from 97/130 to 121/130 in one start and from 93/130 to 121/130 in another. Exact answers for *both* roles reached only 41/130 and 49/130. These are exploratory results on a single model family. The EWT subject/object gains are a separate diagnostic because its participant order and semantic labels do not support a clean patient-generalization claim.

## Question and predictions

**Question:** Does explicit, ordered role supervision help a pretrained Transformer assign roles across natural wording and participant order, beyond the benefit of seeing more natural examples?

- If it improves both-role accuracy on unseen natural sentences and on verified active/passive alternations, role-focused learning is useful beyond the artificial task.
- If it improves ordinary QA-SRL but fails passive or reversed-role cases, the model may have learned common surface patterns rather than a transferable relation.
- If structured extraction succeeds while Qwen's generated answers fail on the same items, answer generation or prompting becomes a plausible bottleneck; that comparison alone does not prove an internal mechanism.
- If neither model succeeds, inspect label quality and the linguistic transformation before inventing another training loss.

## Smallest interpretable experiment

Use the locally available `distilbert-base-uncased` checkpoint as a pretrained encoder Transformer. Mark the target event predicate in each sentence. Attach two span readouts, one for agent/actor and one for the second participant. Score exact spans and whether **both** spans are right. Use the term *patient* only for examples independently verified as patients; QA-SRL direct objects are otherwise called second participants.

Train matched starts with the same natural examples, optimization budget, and backbone:

1. **Ordinary answer training:** one shared span readout conditioned on an actor or second-participant question.
2. **Matched auxiliary control:** the same two-readout architecture as the next arm, trained to identify the first and second participant **by surface order**.
3. **Ordered role supervision:** the two readouts instead identify actor and second participant **by event role**. This is the closest natural-language analogue of the controlled ordered-role versus unordered-identity comparison.
4. **Transformation objective (conditional):** add same-event active/passive stability and true role-swap sensitivity only after enough training examples have been checked for grammaticality and meaning. Compare it against an equal-data arm without that objective.

The first three arms answer the immediate bridge question. The fourth is a later repair test, contingent on a clean failure mode. **Training examples must include verified cases where actor order differs from surface order**; otherwise the second and third arms have nearly identical targets, and their comparison is meaningless. The current 256 selected QA-SRL events are active subject/object examples, so the first step is to assemble that additional training material from a separate source and verify its roles. A from-scratch Transformer trained on only 256 selected natural examples would also be difficult to interpret because the data are too limited to learn general English from scratch.

## Data and evaluation

- Draw training and development items only from the original QA-SRL Bank training partition. The existing extraction found 1,154 candidate training events; the earlier Qwen continuation used 256 of them. Document-level and normalized-event overlap must be checked.
- Use the existing 130-event Bank test as an **exploratory comparison** because it has already influenced model choices. Do not call another run on it independent confirmation.
- Treat the 75 EWT items as a subject/object diagnostic, not a semantic agent/patient test. Report subject position and answer-position baselines.
- Reserve the 32-pair active/passive challenge for final testing after blind human review and protocol freeze. The AI provisional review is a screening aid, not verified ground truth. If too few clear reversible events remain, build a larger human-checked set before making a generalization claim.
- Compare the small Transformer and existing Qwen checkpoints on the **same verified sentences**. Report actor and second-participant exact accuracy, both-role accuracy, active/passive agreement **and correctness**, reversed-role sensitivity, and results by construction. Keep candidate-relative scores separate from answers generated without candidates.

## Decision after the bridge

If ordered roles help natural sentences but not changes in voice or participant order, develop the transformation objective. If they help the small Transformer but not Qwen on matched data, test Qwen with a structured role readout before altering its language-model objective. If the gain disappears under verified active/passive items, focus the paper on the boundary of transfer and on why familiar evaluations overstated it. Either outcome addresses the main question without claiming that the present controlled repair already fixes LLMs.
