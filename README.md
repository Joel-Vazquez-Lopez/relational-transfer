# Relational Transfer in Transformers and Language Models

Research investigating how Transformer-based models learn, represent, and generalize semantic relationships, particularly actor–patient roles.

## Research questions

1. Under what conditions do small Transformers transfer semantic roles to participant pairs not encountered together during training?
2. Can role-aware training improve generalization to unseen relational combinations?
3. How reliably do pretrained language models identify semantic roles across different sentence constructions and question formulations?

## Research approach

The project progresses from controlled synthetic experiments to increasingly complex language settings.

1. **Synthetic relational tasks:** Construct controlled datasets with observed and withheld participant pairs.
2. **Controlled Transformers:** Investigate how training-pair structure affects generalization.
3. **Role-aware learning:** Compare ordinary training with explicit supervision of semantic roles.
4. **Pretrained language models:** Investigate role attribution under sentence and prompt variation.
5. **Targeted interventions:** Study possible improvements to relational robustness.
6. **Generalization:** Evaluate on unseen prompts, new lexical combinations, larger datasets, authentic language, and multiple models.

## Repository structure

- `proposal/` — Research proposal, abstract, and presentation.
- `preliminary_research/` — Experimental code, protocols, analysis, and preliminary findings.
- `research/` — Future experiments and research development.

## Research status

This repository contains an evolving research project. Preliminary experiments are exploratory and do not establish that any intervention generalizes to all pretrained language models.

## Reproducibility

Experimental records should document model versions, training configurations, random seeds, evaluation splits, and results. Reserved evaluation material and private data are not published.

## Author

Joel Vázquez López
