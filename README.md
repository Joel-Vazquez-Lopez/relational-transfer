# Relational Transfer in Transformers and Language Models

**Project:** Explaining Language Through Science · **Research deadline:** 15 December 2026  
**Scientific lead:** Joel Vázquez López · **Status:** exploratory, independently audited research

We study how Transformers **encode, select, compose, and use relational information**: who did what to whom, in which event, and how these answers change under sentence structure, question wording, and context. Our central question is no longer simply whether a model returns correct actor–patient answers:

> **When relational information is available, what determines whether the answer-generation pathway uses it correctly—and what evidence can distinguish an executed relational computation from a shortcut or post-question reconstruction?**

A correct answer, decodable representation, or successful activation patch is **not**, by itself, proof of an internal algorithm.

## Key findings and boundaries (15 October 2026)

All numerical results below are **attributed to their workstream reports**. Except where explicitly stated, they are *not independent replications by Mira*. Different experiments, tasks, interfaces, and models must not be pooled as evidence of one shared mechanism.

| Evidence | Reported finding | What it does **not** establish |
|---|---|---|
| **Brain-O: event information vs answer use** | On a structured synthetic two-event task, a two-layer Transformer learned 100% query-event identification but only 53.25% disjoint-answer accuracy. Matched three-layer local-auxiliary training reached 2,000/2,000 generated QA cases; query-verb-state transplant changed answers in 500/500 cases in the successful model versus 1/500 in the two-layer control. | No unique depth requirement, general pretrained-LM mechanism, or natural-language generalization. [Report](research/brain-o/EXPERIMENTS_021_029_CAUSAL_ROUTING_2026-10-15.md) |
| **Astra: the interface changes the apparent ability** | In a completed synthetic Qwen2.5-0.5B development diagnostic, patient-cleft joint correctness for untouched Qwen was 3.1% with two direct role calls versus 90.6% with a joint extraction request; rankings among adapters also shifted across interfaces. | Interfaces differ in prompting/number of calls and were not compute matched. This is not proof that extraction retrieves a preserved abstract binding. [Astra snapshot](https://github.com/Joel-Vazquez-Lopez/relational-transfer/tree/codex/astra-research-2026-10-15/research/astra_snapshot_2026-10-15/completed_adapter_diagnostic) |
| **Comput: historical response shortcut** | An independent replay of the original `llm_role_internal_v2/RESULT.json` found 240/384 correct answers and 368/384 agreement with a text-derived voice × question-wording positional rule; synonymous actor questions diverged on 48/64 matched active sentences. | Behavioral compression is not an identified neural circuit; the 384 responses reuse a small number of pairs and verbs. [Report](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/comput-falsification/research/comput/REPORT_011_INDEPENDENT_RESPONSE_REPLICATION.md) |
| **Compass: probing false positives** | Supervised high-capacity probes memorized 144/144 labels from contextual states, raw reconstruction inputs, and even raw one-hot tokens on the finite training table. | Probe success alone cannot show that a model already computed the relation. [Report](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/compass-reuse/research/compass/INVESTIGATION_11_2026-10-15.md) |
| **Compass/coordinator: route calibrations** | Explicit cache and raw-reader routes can be behaviorally indistinguishable yet selectively affected by known-port interventions. | Artificially exposed ports are not naturally isolated pretrained-Transformer circuits. [Coordinator calibration](https://github.com/Joel-Vazquez-Lopez/relational-transfer/tree/research/coordinator-mechanism-discriminator/research/coordinator_discriminator) |
| **Controlled ordered-role training** | Early graph effects weakened or reversed under stronger ability matching; the frozen strict-factorial/performance-matching gates failed. | An ability-independent ordered-role generalization benefit is **not established**. [Evidence registry](evidence/EXPERIMENT_REGISTRY.md) |

**Working hypothesis, not established discovery:** models may differ in *where relational information is available* versus *whether it causally contributes to answering*. This must be tested against competent shortcuts, reconstruction controls, and independently validated language.

## Datasets and evaluation policy

The research uses three **distinct** evidence layers:

1. **Programmatically controlled synthetic worlds** for exact gold labels and causal counterfactual checks.
2. **Generated naturalistic development examples** for linguistic stress tests—not equivalent to naturally occurring language.
3. **Authentic, independently semantically annotated text** for external-validity and eventual confirmation claims.

**Mira dataset development (external conversation artifacts, not automatically in GitHub):**

| Version | Status | Main lesson |
|---|---|---|
| V1 | Development-only; rejected as benchmark | Frequency shortcut and duplicate examples. |
| V2 | Development-only | Matched lexical bags, but anchor-blind rule succeeded on 320/320. |
| V3 | Development-only | Anchor sensitivity improved; verb-blind two-hop graph traversal still scored 320/320; punctuation leaked the split. |
| V4 | **Exploratory, awaiting independent Comput audit** | 2,592 questions / 60 families across five relational motifs. On the 1,152-question predicate-competition condition, a full programmed parser scored 100%, while first-/second-predicate-blind references scored 50% and both-blind 25%. These are **programmed baseline** scores, not LM results. |

V4 and its raw artifacts require an authorized external handoff; **do not infer access from this README**. The separately sourced English Web Treebank pilot has 20 candidate sentences and **zero independently human-verified semantic labels**. Grammatical `nsubj`/`obj` or PropBank numbered arguments must not be silently equated with universal actor/patient roles.

Unit of analysis is the independent **source family/event/document**, not the raw count of correlated prompts, paraphrases, seeds, or counterfactual edits. Freeze splits before tuning and keep sealed confirmation data untouched.

## Research workstreams

- **Scientific lead and GPT-6 coordinator:** research decisions, cross-workstream evidence gates, novelty and integration.
- **Astra:** adaptation, fine-tuning, matched checkpoint contrasts, interface and role-readout diagnostics.
- **Compass:** causal reuse vs question-time reconstruction, known-route controls and probe identifiability.
- **Brain-O:** falsifiable computational hypotheses and learned event-selection/routing controls.
- **Comput:** shortcut falsification, independent response replication and cluster-aware uncertainty.
- **Mira:** independent behavioral replication, dataset leakage/label audits and skeptical assessment of mechanistic claims.
- **Nova:** shared, reproducible data and evaluation infrastructure.

[Cross-workstream operations / dataset intake](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/16) · [Evidence and replication gate](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/11) · [Mira review](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/10) · [EUREKA reporting protocol](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/12)

GitHub issues coordinate work but **do not automatically notify separate ChatGPT conversations**.

## Next evidence gates

1. **Raw response replication:** independently recompute Astra's direct-QA/extraction paired transitions from its newly accessible unit-level outputs; separately score semantic correctness, answer format, validity, and prompt/interface contrasts. Verify hashes and cluster by original world/pair.
2. **Known-mechanism replication:** independently replay Brain-O's checkpoint and query-state transplants; verify that interventions are specific, in distribution where possible, and not merely high-capacity readout artifacts.
3. **Benchmark falsification:** Comput independently attacks Mira V4 with unseen shortcuts; Nova enforces split/provenance integrity. A held-out graph-structure split is still *development* if repeatedly inspected.
4. **External validity:** collect and adjudicate authentic, predicate-specific roles with document-disjoint held-out evaluation. Human-reviewed ambiguity is an outcome, not an exclusion to hide.
5. **Mechanistic identification:** only after known-route learned positive/negative controls succeed should anyone infer a native computation route in pretrained models.

No major new model inference, adaptation, or costly training without prior approval. Negative outcomes remain in the record. A **EUREKA** signal requires a decisive prediction and independent falsification/replication—not excitement alone.

## Repository map and reproducibility

- `proposal/` — original project proposal and presentation.
- `evidence/` — curated experiment registry, provenance, limitations, and raw-data access status.
- `historical_reports/` — prior research handoffs, checkpoints, and critical reviews.
- `research/` — workstream records; some current work lives on independent branches or in external archives.
- [Historical data access](evidence/HISTORICAL_DATA_ACCESS.md) explains which ZIPs contain raw evidence and which are *not* committed.
- [Reproduction guidance](evidence/REPRODUCTION.md) describes integrity checks and controlled reruns.

For every scientific claim, preserve original files; source SHA-256; code revision; model/checkpoint and tokenizer revision; prompts/decoding; seeds; raw predictions; gold-label source; group IDs; exclusions; exact per-event disagreement tables; and runnable commands. Archive hashes establish **file integrity, not scientific validity**. Do not upload restricted corpora, sealed evaluation examples, credentials, or licensed materials without authorization.

## Prior-art boundaries

Related work has already studied permutation/state composition, modular circuit reuse, and causal abstraction. Novelty is **not** established by a synthetic relational task, query-order effect, a fitted linear transformation, or a patch that changes an output. A publishable contribution must isolate a distinguishable mechanism or validated diagnostic and demonstrate a consequential result beyond the restricted development grammar.

## Author

Joel Vázquez López
