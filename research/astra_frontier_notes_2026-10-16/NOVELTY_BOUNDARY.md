# Targeted novelty boundary — 16 October 2026

This is a targeted primary-source check, not an exhaustive novelty review. It does not change the running frozen output-order diagnostic.

Verified primary abstracts:

- Long et al., NAACL2025, 'LLMs Are Biased Towards Output Formats! Systematically Evaluating and Mitigating Output Format Bias of LLMs': https://aclanthology.org/2025.naacl-long.15/ . Separates performance under format compliance from performance irrespective of compliance; evaluates multiple format types and prompting/fine-tuning mitigation. Thus format sensitivity and separating correctness/compliance are not novel claims.
- Kumar, Kanishka, Mandora, 'Structure Tax: How Structured Output affects LLMs Performance', arXiv2610.12056, submitted8Oct2026: https://arxiv.org/abs/2610.12056 . Abstract reports reasoning-first vs answer-first field order effects, especially in smaller models. ArXiv comment says AACL IJCNLP2026 Main; venue not independently verified here. Broad field-order and reasoning-preservation claims have close precedent.
- 'Quantifying the Impact of Structured Output Format on Large Language Models through Causal Inference', Findings EACL2026: https://aclanthology.org/2026.findings-eacl.91/ . Relevant prior work for any causal language about structured-output effects. Full-method comparison still needed.

## Narrow candidate, not established novelty

Explain and predict when the sign of a training-objective comparison changes across semantically equivalent role tests, distinguishing input position, requested output order, construction, and direct question wording. Current tests are on reused synthetic0.5B development material. A robust paper would need held-out linguistic conditions, strong baselines, independent model families, and independently adjudicated natural language. No such transfer follows from the current control.

Possible subsequent intervention must be selected from the observed failure, and compared with known format diversity training, ordinary matched training, explicit syntax parsing, deterministic routing, and training-data augmentation. Do not rename generic consistency or schema randomization as a novel method.

The experiment now running is a discriminator, not a paper result guaranteed in advance. Its outcome may eliminate output-order copying, support it descriptively, or show a mixture. Behavioral agreement with a rule does not identify a neural circuit.
