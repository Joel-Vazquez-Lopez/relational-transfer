# Status-conditioned participant retrieval: cross-model and controlled update

17 October 2026. All inference here used existing local MLX 4-bit weights and greedy generation. No API charges or model downloads. This update does not revise the original bounded causal experiment or claim a new internal mechanism.

## Question and audit trail

An earlier six-pair natural-sentence development pilot gave Qwen2.5-7B 13/24 joint answers, with a 10/12 versus 3/12 target-first advantage. Its sentences were selected by one reviewer from annotation-matched documents, some predicate senses/objects differed, and one positive sentence was reused. [Original report](EPISTEMIC_COMPETENCE_DEV_REPORT.md).

The [cross-model protocol](CROSS_MODEL_DEV_PROTOCOL.json) was frozen before running the same **exposed** six pairs on existing Qwen2.5-1.5B and Llama-3.2-3B. [Script](cross_model_dev.py), [all 48 raw outputs](CROSS_MODEL_DEV_RAW.jsonl), and [machine-readable report](CROSS_MODEL_DEV_REPORT.json):

| Model | Joint correct /24 | Target first /12 | Target second /12 | All-four pairs /6 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B (earlier development) | 13 | 10 | 3 | 0 |
| Qwen2.5-1.5B | 13 | 9 | 4 | 0 |
| Llama-3.2-3B | 14 | 7 | 7 | 0 |

This is an exploratory capacity/family comparison on the *same items*, not independent replication. It does not show a general Qwen mechanism.

## Fresh controlled diagnostic

The natural source gate was not met. The [strict retrieval audit](STRICT_SOURCE_FEASIBILITY.json) found 38 lexical leads in 33 distinct unexposed nonfactual documents under one same-type/stem recipe, before semantic validation. This is **not** a global upper bound on all possible pairs. The leads include mismatched event senses, questionable Agent labels and sparse reported/possible examples. There is no independently adjudicated 30-pair natural set. We therefore did not run that confirmatory protocol on Qwen. [Audit code](source_feasibility_gate.py).

Instead, a separate [controlled protocol](CONTROLLED_STATUS_PROTOCOL.json) froze 24 **new synthetic** pairs before inference: four verbs × negated/failed attempt/unverified report × two unique name pairs, 48 distinct names; both sentence orders and both target questions. The frozen prediction was a Qwen-family target-second deficit. Qwen7B also received four separate component questions per pair. [Script](controlled_status_binding.py), [items](CONTROLLED_STATUS_ITEMS.json), [all 384 raw outputs](CONTROLLED_STATUS_RAW.jsonl), [machine-readable report](CONTROLLED_STATUS_REPORT.json).

| Model | Joint exact /96 | Target first /48 | Target second /48 | Complete pairs /24 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | **95** | 48 | 47 | 23 |
| Qwen2.5-1.5B | **60** | 43 | 17 | 1 |
| Llama-3.2-3B | **95** | 48 | 47 | 23 |

For Qwen1.5B, 18 pair-level first-minus-second differences were positive, six tied, and none negative. The mean first-minus-second accuracy difference was 0.542, with a pair-bootstrap 95% interval of 0.396–0.688 (10,000 draws, frozen seed). Qwen7B and Llama3B each had one positive pair difference and 23 ties; their difference was 0.021, bootstrap interval 0–0.063. Qwen1.5B's deficit appeared in all three status conditions. These are synthetic results, and models differ in family *and* size.

The Qwen7B component probe scored **75/96** exact: Agent 48/48, status 27/48. Nearly all status failures were plainly completed positive sentences answered `UNKNOWN` (21/24 positive status labels missed), despite the joint answer being correct in 95/96 prompts. Only 3/24 pairs passed all four component questions, so the predeclared component-competent subset is not useful. This is a prompt/measurement failure or task-specific inconsistency; the data do not identify which. We did not revise the prompt or rescore after inference. Every response remains in the raw file.

## Decision

The broad claim that Qwen7B cannot combine event status and participant identity in a clean two-sentence setting is **contradicted** by the controlled diagnostic. The six natural pairs cannot support an event-selector mechanism: semantic mismatch, additional entities, sentence complexity, and positional salience remain live explanations. Qwen1.5B does show strong position sensitivity on clean items, but that alone is not a novel event-specific mechanism or a strong paper. No causal patch was run.

The best remaining scientific target is the **boundary between easy status-conditioned retrieval and failures in natural discourse**. A credible paper would need independently reviewed natural or carefully controlled naturalistic passages that vary one property at a time (extra entity, embedding, predicate/object alignment, status scope), fixed competent component prompts, a stronger-model replication, and a causal manipulation whose outcome differs from syntax-aware copying and position heuristics. This is a prospective direction, not a result. The prior [30-pair contract](NEXT_SINGLE_TEST_CONTRACT.md) remains unexecuted and should not be described as confirmation.

## Prior-work boundary

Cross-event argument confusion is already discussed in [Shuang et al. 2024](https://aclanthology.org/2024.findings-acl.328/), positional bias in [Zhang et al. 2024](https://aclanthology.org/2024.findings-acl.487/), negated-entity tracking in [Schuster and Linzen 2022](https://aclanthology.org/2022.naacl-main.71/), event factuality/arguments in [MAVEN-FACT](https://aclanthology.org/2024.findings-emnlp.651/), and causal binding in [Dai et al. 2024](https://aclanthology.org/2024.emnlp-main.967/) and [Mixing Mechanisms 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf). We cannot claim priority for “position matters,” “models confuse events,” or “negation affects retrieval.”

FactBank has richer factuality annotation but is licensed through [LDC](https://catalog.ldc.upenn.edu/LDC2009T23); [NIST BETTER](https://ir.nist.gov/better) offers Agent/Patient event data but its downloads currently redirect to login. Neither was silently substituted for a validated open natural-source corpus.
