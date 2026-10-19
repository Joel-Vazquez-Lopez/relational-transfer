# Targeted prior-work audit and novelty boundary

19 October 2026 local time. Conducted while the fixed panel was collecting; it changes no frozen hypotheses, tests, prompts or interventions. This is a targeted primary-source review, not an exhaustive novelty search. No positive result from this panel by itself establishes a new mechanism or a publishable general law.

## Direct precedent: filtering changes the population under interpretation

[Jenner et al., NeurIPS 2024, *Evidence of Learned Look-Ahead in a Chess-Playing Neural Network*](https://papers.nips.cc/paper_files/paper/2024/file/37d9f19150fce07bced2a81fc87d47a6-Paper-Conference.pdf), §2.2 (PDF p3) and the continuation on p4, explicitly report that their difficulty/success filters change the frequency of coincident first/second move target squares from 47% to 83%. They also discuss stronger versus weaker interpretability results across these subsets and retain both. This is direct precedent for filtering changing task structure and the distribution relevant to mechanistic conclusions. Our broad idea that competence/difficulty selection can change the interpreted population is therefore **not new**. Their filter combines weak-model difficulty and strong-model success, not our two answer rules on unchanged LLM scores. Our narrow prospective question concerns syntactic construction admission, fixed external readout, and a cross-checkpoint intervention comparison.

[Nikankin et al., ICLR 2025, *Arithmetic Without Algorithms: Language Models Solve Math with a Bag of Heuristics*](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c5f30296296d2ae402ebbd09aaa9c12-Paper-Conference.pdf), Appendix G (PDF p19), explicitly use all arithmetic prompts to estimate ablation means, maintaining operator balance rather than restricting those means to correct prompts. Their answer probes use correctly completed prompts. This already recognizes that the appropriate sampling policy depends on the estimand. It would be wrong to characterize all previous studies as oblivious to selection or to claim that retaining all cases is our new methodological invention.

[Fierro et al., ACL Findings 2025, *How Do Multilingual Language Models Remember Facts?*](https://aclanthology.org/2025.findings-acl.827/), PDF footnote6, uses correctly predicted completions to study factual recall and reports unequal fact counts across languages. Such restriction can be well motivated for a conditional question. Our panel cannot show that their conclusions are biased or refute their mechanisms; that would require their data and a specified target population.

## Readout and assay dependence

[Zhao et al., ICML 2021, *Calibrate Before Use*](https://proceedings.mlr.press/v139/zhao21c.html) already studies answer biases and external calibration. Our old exposed-data midpoint is not their content-free estimator, and is neither a new calibration algorithm nor proof of a model-internal posterior.

[Zhang & Nanda, ICLR 2024, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*](https://arxiv.org/abs/2309.16042) studies how patching metrics and corruption methods change interpretability findings. Here the underlying score and intervention are held fixed and sample eligibility changes; that is a narrower source of variation, not discovery that methodology matters in general.

## Binding mechanism boundary

[Dai, Heinzerling & Inui, EMNLP 2024, *Representational Analysis of Binding in Language Models*](https://aclanthology.org/2024.emnlp-main.967/) identifies low-rank binding-related subspaces and causally edits retrieval. Our binary panel does not isolate an equivalent semantic binding variable, and the earlier failed period-donation signature remains failed. A threshold/screen result supplies no rival binding mechanism and does not establish priority over Dai et al.

## What could survive

A successful fresh confirmation would establish a reproducible, prompt/checkpoint-specific construction-selection effect under an independently fixed readout rule. If the secondary causal comparison also changes, it would quantify a concrete inferential sensitivity on this linguistic panel. Novelty would still require broader constructions, answer conventions, checkpoints and a more comprehensive literature comparison. If that secondary effect fails to reproduce, the report must narrow to sample admission. Significance of the sole conditional family sign test, if present, does not settle these other questions.
