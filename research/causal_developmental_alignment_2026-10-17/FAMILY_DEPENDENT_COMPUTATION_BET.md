# Research bet: family-dependent computational equivalence

17 October 2026. Design analysis only. No new model inference, patching, API use, or claim of discovery. The previous factorial and transport studies remain stopped under their frozen rules.

## Question

Do different transformers preserve a *task-conditioned causal dependency pattern* while differing in its geometric route, location, and depth? “Family” has two axes: (1) model lineage/architecture, and (2) input construction or task. These must be crossed rather than conflated. Similar output behavior is a prerequisite, not evidence of identical internal computation.

The most plausible shared object is a **partial order of functional dependencies**, not a universal vector path. For event-role retrieval, candidate variables are participant identity, semantic role, event identity, question target, and answer emission. A model could compute them in different coordinates or at different relative depths. However, the proposed order is a hypothesis; neither layerwise probe decodability nor a factorial interaction proves it.

## Why this is different from the failed studies

The saved final-token path Grams were more similar for unrelated sentences than role reversals; lag-8 role agreement fell to +0.024 Qwen / -0.011 Llama. The existing subject/object opposition is confounded by exchanging names. The factorial competence gate passed 2/4 Qwen families and 0/4 Llama families. The later event-period transplant passed competence on one common development family but selective role contrast was only +0.195/+0.036 nats, below the frozen +0.5 criterion; a same-answer unrelated donor had a larger Qwen effect. These results reject the present *measurement* as evidence for semantic-role transport, not all conditional invariants.

The new discriminating endpoint is whether a simple dependency pattern fitted on one **construction family** predicts *native causal effects* on untouched construction families and models, beyond output/position baselines. Do not use the exposed examples as confirmatory items and do not move the failed patch site within the stopped study.

## Candidate observable: an intervention-response fingerprint

For each competent model and input family, define a fixed, small panel of perturbations to **inputs** (rename one participant, reverse semantic roles, preserve roles while changing voice, change event reference, swap the queried role), and a fixed panel of **model-native** interventions (predeclared token sites and block bins). Record a vector of changes in full-answer margins for each intervention. Keep behavioral responses, hidden-state geometry, and causal responses separate.

A fingerprint is a matrix indexed by named intervention and readout, with depth treated as an ordered axis. Compare matrices across models in this *shared experimental label space*, avoiding direct hidden-coordinate alignment. Test partial orders by intervention-and-rescue: a proposed upstream variable should affect its predicted downstream readouts, while a downstream rescue should selectively restore the answer. A stage label counts only if it predicts held-out interventions; a probe score or visible activation peak alone does not establish an operation.

The key quantity is **excess predictive transfer**: improvement in held-out causal-effect prediction over (i) normalized-depth schedule, (ii) output-only/logit trajectory, (iii) lexical identity, surface order and tokenization, (iv) same-answer unrelated donor, (v) generic perturbation norm, and (vi) shuffled operation labels. Report both per-document and per-answer outcomes; bootstrap by lexical family or document, never by correlated layer or answer row.

## Crossed design and falsifiers

1. Freeze fresh, independently checked natural passages or controlled constructions with source labels, named participants, and unambiguous answers. Include at least two construction families whose role and position cues disagree (for example canonical active/passive and an agent-first passive frame). Hold out entire verbs, names, and construction families. Natural data should be a separate external check, not silently mixed with synthetic families.
2. Screen every model on every question before causal analysis. Declare competence thresholds and failed-family counts in advance. Given the prior Llama Patient failure, consider an easier task with reliable **both-role** answers or make Agent-only scope explicit. Do not cherry-pick only successful rows after screening.
3. Cross at least one within-lineage model pair and one across-lineage pair, *if local models pass competence*. Existing local assets include Qwen 0.5B/1.5B/7B, Llama 3B and Phi-3; smaller Qwen models may fail this task, so a within-lineage claim is conditional on an honest gate. A two-model Qwen 7B/Llama 3B study alone cannot estimate a model-family effect.
4. Freeze the fingerprint, stage mapping, controls, baseline models, and primary metric on development data. Test untouched constructions and models once. No flexible time warping or site search on holdout. A successful result must predict selective causal effects, not merely classify transformation labels provided to the method.
5. Explicit falsifiers: unrelated same-answer controls predict as well as role controls; surface order predicts as well as the proposed relation; the fingerprint predicts only within lexical families; the within-lineage comparison is no better than cross-lineage after task competence is matched; or patches have no measurable selective effect. In these cases report a boundary/negative finding, not a shared computational path.

## Smallest useful next step

**Do not run another patch grid now.** First perform a preregistered, no-cost feasibility audit of the existing local models and a fresh source set: can at least two models from one lineage and one other lineage answer all required questions in enough independent families? If no, the intended family interaction is not identifiable with current local resources; choose a more tractable task or frame the course work as a methodological boundary study. If yes, freeze one prospective causal design. The audit must be new and separately logged; it does not reopen the prior stopped experiments.

## Prior art and novelty boundary

- [Bhardia et al., *Mechanistic Analysis of Universality* (ACL SRW 2026)](https://aclanthology.org/2026.acl-srw.84.pdf) already find strong within-Qwen numerical-comparison circuit consistency and qualitatively different other-family implementations. Thus “family dependence exists” is not novel.
- [Aljaafari et al., *Emergence and Localisation of Semantic Role Circuits* (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.1964.pdf) already compare semantic-role circuits across scale/family and report more component than edge preservation. Thus shared role circuitry cannot be claimed from a generic cross-model similarity.
- [Sun and Toneva, *Tracking Equivalent Mechanistic Interpretations* (2026)](https://arxiv.org/abs/2603.30002) directly study interpretive equivalence across model families. [*Differentiable Faithfulness Alignment* (2026)](https://arxiv.org/abs/2604.24302) studies cross-model circuit transfer. [Cross-language reasoning invariance (2026)](https://arxiv.org/abs/2608.23809) reports that higher geometric alignment does not uniformly improve functional interchangeability.

The possible contribution is narrower: a **prospective, task-conditioned map of where computational equivalence holds and breaks**, using held-out predictions of controlled causal responses across both construction and model families. This is a research hypothesis, not a result. It could yield a solid methodological/empirical paper only with enough independent families, competent models, genuine selective intervention effects, strong baselines, and replication.
