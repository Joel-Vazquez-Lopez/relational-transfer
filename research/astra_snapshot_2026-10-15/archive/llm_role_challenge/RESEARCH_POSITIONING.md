# Paper positioning while the blind test awaits review

**Status:** research framing, not a novelty claim or a result from the new challenge.

Two relevant findings are already in the literature. [Wilson, Petty, and Frank (2023)](https://aclanthology.org/2023.tacl-1.78/) studied LLM generalization across argument-structure contexts including active and passive, and reported a linear-order bias when the relevant structural relation was not already supported by pretraining. [Hu and Levy (2023)](https://aclanthology.org/2023.emnlp-main.306/) found that metalinguistic prompts and direct probability measurements can disagree. Therefore the paper should **not** claim to discover either active/passive role difficulty or prompt/measurement divergence for the first time.

Our possible contribution is more specific: connect a controlled role-transfer intervention to a pretrained decoder model, compare matched source and adapted starts plus a copy control, and ask whether an improvement appears in **both participant answers at once** under natural active/passive variation and alternate question wording. The existing 75-item EWT results support the training effect on a narrow active subject/object task, but the nearly fixed participant order and provisional semantic labels prevent a general role-transfer claim. The independently reviewed active/passive comparison is the decisive next observation.

Three possible paper outcomes remain scientifically useful:

1. **Transfer survives voice and wording:** A bounded positive result for the training method, with exact both-role answers and source-event uncertainty. A larger dataset and another model family would then be the most valuable extension.
2. **Candidate scores improve but generated answers fail:** An evaluation-validity result about the gap between relative evidence and usable role answers. The training method would need a more precise objective or output interface.
3. **The gain vanishes on passive items:** Evidence that the EWT gain was tied to the mostly fixed surface order. The right next experiment would isolate voice, participant order, and role reversal before changing training.

The 15-credit report can be solid if it answers a bounded question with clean controls and honest limits. A stronger publication claim depends on the reviewed challenge and later replication, not on treating the present EWT ceiling as a finished repair.
