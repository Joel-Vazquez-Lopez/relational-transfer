# One prospective test after the development pilot

17 October 2026. **Design contract, not a frozen execution protocol.** No new items have been independently adjudicated, so this test must not be run yet. The six development pairs and both Qwen runs are permanently excluded from confirmation.

## Scientific question

Does a model that can separately identify an event's epistemic status and named Agent still fail to combine them when selecting an Agent from two competing event descriptions? If so, does swapping sentence order predict the failure better than status or lexical content?

The [development run](EPISTEMIC_COMPETENCE_DEV_REPORT.md) saw 10/12 correct when the requested Agent was in sentence one and 3/12 when it was in sentence two. A [post-pilot component diagnosis](EPISTEMIC_COMPONENTS_DEV_REPORT.md) scored 10/12 status and 11/12 Agent in isolation. These are exposed, small, prompt-confounded observations. The test below would be the first independent predictive check, not a replication claim based on the same six sentences.

## Source and annotation gate

Before model inference, assemble **30 pairs** from at least **60 distinct source documents**, excluding every development document and every positive sentence reused in development. Each pair contains two original, intact source sentences with the **same predicate lemma** and distinct, locally explicit Agents: one certainly occurred; the other is a clear CT−, failed/attempted, or reported/possible mention. Target ten pairs in each nonfactual category. Pairing itself is experimenter-constructed and must be labeled as such. Reserve any untouched original paragraphs for a separate external-validity analysis.

Two reviewers independently label both sentence-level Agents, event mention spans, completed/negated/unknown status *as expressed in that sentence*, temporal scope, and whether a same-predicate question has one unambiguous answer. Resolve disagreements before Qwen sees the items; publish exclusions, agreement, and all source IDs. No source sentence may require an unstated pronoun antecedent or contain a competing actor completing the same predicate. The [candidate generator](build_epistemic_pair_candidates.py) supplies leads only; its lexical ranking is not gold.

If 30 valid pairs cannot be produced, **stop**. Do not lower the threshold after seeing model outcomes or recycle the six development pairs. A course report may still present the source audit and exploratory pilot; the proposed publishable study would remain undone.

## Frozen questions and measures

The final item manifest must include exact prompts and gold strings before any held-out inference. For each source sentence alone, ask (i) whether its target event is explicitly completed, explicitly not completed, or not established, and (ii) who its named Agent is. With both sentences, ask which Agent actually completed the predicate and which Agent is mentioned without established completion. Present factual first and nonfactual first, with both question targets in each order. Use natural role wording that applies equally to people, teams, governments and vehicles; avoid the old person-versus-team conflict. A second, predeclared structured prompt is a robustness measure, not a replacement after outcomes.

Report every full response under an exact predeclared alias table. Primary denominator is **all 30 pairs**. A pair is *component competent* only if all four single-sentence answers are correct; never silently drop failures. On the component-competent subset, estimate the within-pair target-first minus target-second joint-answer accuracy. Use pair-clustered uncertainty intervals and a paired exact or permutation test; also report results by nonfactual category. The positional baseline always returns the first sentence's Agent; the syntax-aware status-and-role copier returns the status-appropriate Agent. A lexical overlap baseline and a last-sentence baseline are secondary.

**Prospective prediction:** component competence will exceed joint competence, and joint accuracy will be lower when the requested Agent is in sentence two, especially for reported/possible events. This is a claim about Qwen2.5-7B on this constructed corpus, not all LLMs. If fewer than **20 of 30** pairs pass the four-component screen, the conditional composition question is underpowered; report competence failure and stop. If component-competent pairs show no consistent target-position penalty, the development pattern did not generalize. If they show the penalty, describe an **order-sensitive composition failure**; do not infer an internal sequence of operations from behavior alone.

## Separate mechanistic gate

Only a successful, source-valid behavioral result justifies a new causal study. Its hypothesis must predict an intervention outcome that differs from a competent syntax-aware copier with matched information and edit budget. The previous full-residual question-label patch already occurred before role wording and cannot serve as new evidence. A status-only direction trained on development status questions, tested on held-out joint questions with norm-matched shams and off-target metrics, is one candidate. If no such witness can be specified, stop at the behavioral claim. Do not patch first and explain later.

## Literature boundary

The proposed contribution must be stated against [Schuster and Linzen 2022](https://aclanthology.org/2022.naacl-main.71/) on negated discourse entities, [Sulem et al. 2021](https://aclanthology.org/2021.findings-emnlp.385/) on competitive unanswerable event questions, [MAVEN-FACT](https://aclanthology.org/2024.findings-emnlp.651/) on factuality, [QAEVENT](https://aclanthology.org/2024.findings-eacl.126/) on event QA, [Dai et al. 2026](https://aclanthology.org/2026.acl-long.2194/) on binding interventions, and [Mixing Mechanisms](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf) on positional and lexical routes. “Models ignore negation” and “position matters” are already known. The possible new result is the **conditional conjunction failure**, established on validated status–participant event pairs with a prospective prediction and causal follow-up if identifiable.
