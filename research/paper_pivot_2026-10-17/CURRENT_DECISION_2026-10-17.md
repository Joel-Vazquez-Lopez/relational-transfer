# Research decision, 17 October 2026

**Best current paper bet:** conditional composition of event factuality and participant retrieval. The publishable question is whether a model that separately identifies a described event's status and participant fails to select the participant of the *actual* event when a nonfactual event competes, and whether event order prospectively predicts the failure. This is a hypothesis, not an established mechanism.

## Evidence acquired today

1. Public MAVEN-FACT validation joins exactly to MAVEN-Arg validation on all 710 document IDs/texts; 15,405 event IDs overlap. [Join code and audit](maven_fact_join_audit.py), [JSON counts](MAVEN_FACT_VALID_AUDIT.json).
2. Two disjoint one-reader, 36-document source samples produced 22 provisional source-valid single-event cards each: [round one](FACTUALITY_SOURCE_REVIEW_2026-10-17.md), [round two](FACTUALITY_SOURCE_REVIEW_ROUND2_2026-10-17.md). This is not a representative quality estimate or independently adjudicated gold. The within-document same-type distractor design has only two structural leads among round-one accepts and fails as a confirmatory path.
3. A constructed-sentence [Qwen development pilot](EPISTEMIC_COMPETENCE_DEV_REPORT.md) froze six pairs, two sentence orders and two target questions before local inference: **13/24** primary answers, **0/6** complete pairs. Requested Agent in first sentence: **10/12**; second sentence: **3/12**. This is exploratory, prompt-confounded and source-selected.
4. A separately frozen, **post-pilot** [component diagnosis](EPISTEMIC_COMPONENTS_DEV_REPORT.md) on the same exposed sentences scored **10/12 status** and **11/12 Agent** in isolation. This makes a composition question plausible, but cannot establish it independently.

## Stop and go

**Stop now on activation patching and unique event-selector claims.** The earlier full-residual pre-role patch is already development evidence; the new paired Qwen items fail the joint competence screen and offer no discriminating causal witness. Do not tune prompts on these six and call it confirmation.

**Next go gate:** [one fresh prospective test contract](NEXT_SINGLE_TEST_CONTRACT.md): 30 independently reviewed matched sentence pairs from 60 distinct source documents, status and Agent components plus joint selection, order counterbalance, first-sentence and syntax-aware baselines, and pair-level reporting. If source validity or component competence fails, the course report can still present a rigorous negative investigation. A paper-level mechanistic claim needs a separate, predeclared causal witness and preferably replication in another model or corpus.

**Novelty limit:** Negated discourse entities, competitive unanswerable QA, event factuality, event QA, positional binding and binding interventions all have substantial prior work. The possible increment is a *prospectively predicted conditional conjunction failure* and its selective causal pathway. [Literature and prediction wager](EPISTEMIC_BINDING_NOVELTY_WAGER.md). No claim of priority or acceptance is warranted yet.
