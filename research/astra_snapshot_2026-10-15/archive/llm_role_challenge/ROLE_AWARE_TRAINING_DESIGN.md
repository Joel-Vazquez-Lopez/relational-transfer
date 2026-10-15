# Candidate next intervention: role aware pattern learning

**Status:** design proposal only, 12 October 2026. No model has been trained with this objective. Keep the new 32-pair active/passive challenge untouched until human review and the current checkpoints are evaluated.

## Question

Can a pretrained decoder model learn a reusable mapping from linguistic form to event roles, beyond learning to answer the particular actor/object prompts used in adaptation?

The existing natural role adaptation gives the model actor and direct-object answers. The 75-item EWT gain is encouraging, but those items almost always put the subject first and many objects are not semantic patients. The new objective must teach what **should remain stable** when wording changes and what **must change** when the event roles reverse.

## Training unit

For one reversible event, form a four-sentence block with verified labels:

| Form | Sentence | Agent | Patient |
| --- | --- | --- | --- |
| Original active | Mira photographed Taro. | Mira | Taro |
| Same event, passive | Taro was photographed by Mira. | Mira | Taro |
| Roles reversed, active | Taro photographed Mira. | Taro | Mira |
| Reversed event, passive | Mira was photographed by Taro. | Taro | Mira |

These are illustrative sentences, not examples selected for training or evaluation. Real training blocks must be checked for grammaticality, plausibility, and whether the changed sentence truly reverses the intended event. The training data and all generated variants must be disjoint from the frozen challenge by normalized sentence and source event. Do not call every direct object a semantic patient: use verified agent/patient examples for the patient objective and label other relations separately.

## Model target

Attach a small role readout to the decoder states for the target event. It predicts a distribution over the **participant spans** for two slots: agent and patient. Train this readout together with a small adapter using:

1. **Role supervision:** the agent and patient slot predictions must choose the verified spans. Keep the existing direct-answer objective as a separate baseline.
2. **Meaning-preserving consistency:** active and passive versions of the *same event* should produce the same agent and patient predictions, even though the words occupy different positions.
3. **Role-swap sensitivity:** when the participants genuinely exchange roles, the slot predictions must exchange accordingly. This prevents a model from satisfying consistency by always choosing the same name.

Constrain the **role readout**, not the model's entire hidden state. Active and passive sentences legitimately differ in syntax; forcing all hidden states to match could erase useful information.

## Matched comparison

Use the same base checkpoint, number of examples, optimizer updates, and adapter capacity for each arm. Use at least two matched random starts.

| Arm | What it tests |
| --- | --- |
| Ordinary role-answer training | Existing intervention and reference level. |
| More role-answer examples, same compute | Whether gains come from additional labeled practice. |
| Generic same-event consistency | Whether agreement alone helps or produces repeated-answer shortcuts. |
| Role slots + active/passive consistency | Whether explicit roles stabilize across surface form. |
| Role slots + consistency + reversed-event sensitivity | Full proposed method. |

The minimal first pilot can use the existing 0.5B model and a small set of training blocks, but model selection must use separate development events. Keep the 32-pair human-reviewed challenge as a final test. If the challenge reveals that the current model already transfers robustly, revise the target problem before training this method.

## Evidence required

Primary: exact **both-role-correct** accuracy, separately for active, passive, and two question wordings, plus correct behavior for both same-event and reversed-event pairs. Report actor and patient separately, candidate likelihood contrast separately, and the gap between them. Check unseen participant pairs, unseen verbs, and held-out constructions as distinct splits. Retain familiar-language performance to detect a trade-off.

A successful training result would improve these measures across matched starts and independent event blocks, beyond the extra-data and generic-consistency controls. Internal-role readouts or causal interventions could then test mechanism; behavioral accuracy alone would not establish a unique internal representation.

## Relation to prior work

[Wilson et al. (2023)](https://aclanthology.org/2023.tacl-1.78/) already report active/passive argument-structure generalization and linear-order bias. [Elazar et al. (2021)](https://aclanthology.org/2021.tacl-1.60/) study consistency under meaning-preserving paraphrases. [Guo et al. (2024)](https://aclanthology.org/2024.findings-acl.680/) address reversal with semantic-unit permutation training. The proposed contribution would have to be the **paired role-invariant and role-changing objective, its matched controls, and its transfer result**, not the general idea of consistency or reversal. Novelty remains a hypothesis pending fuller related-work review and a successful independent test.
