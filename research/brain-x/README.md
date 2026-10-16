# Brain-X — Independent Scientific Invention Laboratory

**Project:** Explaining Language Through Science  
**Scientific lead:** Joel Vázquez López  
**Deadline:** 15 December 2026  
**Workstream:** Brain-X  
**Branch:** `research/brain-x-theory`  
**Initial draft PR:** [#21](https://github.com/Joel-Vazquez-Lopez/relational-transfer/pull/21)

## Identity and role

Brain-X is an independent scientific invention and high-risk/high-reward computational theory workstream. It investigates *principles* of relational computation and designs falsifiable experiments that can overturn the project's present assumptions.

**Brain-O** investigates where relationally relevant information travels, when it is causally used, and whether its learned routing generalizes. **Brain-X** investigates whether a different mathematical object explains the phenomenon altogether: e.g. global consistency, computation scheduling, error correction, conditional inference, or learned transformation algebras.

Brain-X does not coordinate the team, validate other members by fiat, manage their branches, speak as a separately authenticated GitHub human/collaborator, or claim automatic communication with other conversations.

## Current research agenda

1. **H1 — Constraint-induced relational repair (priority):** investigate whether joint output can exploit globally informative constraints absent from isolated answers, with paired valid versus randomized constraint controls, matched compute and direct-answer baselines. An initial two-layer numerical Transformer pilot **failed** to outperform a local-evidence baseline despite Bayes-optimal exploitable evidence; the negative finding must remain visible.
2. **H2 — Conditional computation scheduling:** test dependency-depth predictions and whether Brain-O's relay simply reflects a generic computation critical path rather than relation specificity.
3. **H3 — Algebra of transformations:** test held-out compositions and noncommuting relational world updates while controlling against shortcuts and known group-equivariance prior art.

Details, assumptions, equations, rejection thresholds and limitations: [scientific notebook](NOTEBOOK_2026-10-15.md).

**Latest completed test (BX-002, 16 Oct 2026):** Repaired BX-001's unpaired test design. Same 4,000 latent worlds and questions, only informative versus randomized parity evidence changed. Two trained three-layer Transformers changed their predictions when constraints were randomized, yet achieved only 65.15% and 65.35% on valid constraints, at the 65.30% node-only baseline; simple parity voting obtained 69.875% and exact Bayes 73.125%. **Negative** evidence for useful native integration in this configuration; no language/pretrained generalization. Full [BX-002 report](BX002_RESULTS_2026-10-16.md), [code](bx002_paired_syndromes.py), [tests](test_bx002_paired_syndromes.py) and [summary](bx002_summary.json). The original per-case CSV is available in a separately checked local ZIP, not in GitHub.

**Day 2 first completed study — BX-003 (16 Oct 2026):** Read-only source-conditioned audit of BX-002's original 4,000 paired predictions. A Bayes reference that marginalizes the actual 80%-valid/20%-random training mixture scores **73.225%** on valid evidence; trained models remain at **65.15% / 65.35%**. Among **899 input-defined neighbor/local conflicts**, seed-17 and seed-29 models override local on only **155 and 90** cases, versus **552** for the calibrated Bayes reference. They gain **45/16** correct predictions on conflict cases but lose **51/14** on nonconflict cases, cancelling net value. This is a post hoc negative **synthetic** diagnostic, not a discovered universal mechanism. See [BX-003 report](BX003_CONFLICT_UPTAKE_2026-10-16.md), [reproduction source](bx003_release.py) and [machine-readable result](bx003_uptake_results.json). **Priority:** independent authentic-language corroboration/irrelevant-context comparison; avoid Astra's output-order and Compass's extended-composition experiments.

**BX-004 — authentic context utility readiness, 16 Oct 2026:** Created a no-inference, gold-isolated direct-answer scorer with document-clustered rescue/harm and source-intake checks. **12/12 local tests pass.** Directly audited Mira's 25-document authentic candidate manifest: **0 semantic-human-reviewed, 0 source-rights-cleared, 0 human-verified informative-vs-irrelevant contrast**, hence **0 presently eligible** for our test. This is expected for a candidate-selection pilot, not a criticism. [Protocol and limits](BX004_CONTEXT_UTILITY_2026-10-16.md), [executable scorer](bx004_evidence_utility.py), [regression tests](test_bx004.py), [machine-readable readiness](bx004_mira_readiness.json). Do **not** run or present authentic-model scores until Mira/Nova/Comput independent gates complete; no LLM results yet.

**BX-005 — evaluator self-falsification and repair (16 Oct 2026):** Proved that BX-004 v1 accepts altered role questions across context conditions while still reporting a mock **+50 percentage-point** context advantage. This is a protocol false-positive, **not an LLM result**. Added [strict canonical context/question/output contract](bx005_context_contract.py), [12 adversarial tests](test_bx005.py), and [failure/revision report](BX005_ADVERSARIAL_EVALUATOR_AUDIT_2026-10-16.md). Original 12 tests + new 12 tests: **24/24 pass locally**. The automated gate explicitly remains `CONTRACT_PASS_PENDING_MANUAL_SOURCE_REVIEW`: SHA commitments and declarations do not verify real humans, original corpus rights, or information gain. Never use BX-004 v1 alone for authenticity or efficacy claims. Same human-edited authentic source readiness remains blocked.

**BX-006 (Day 2, 16 Oct 2026) — existing human-annotated data metric audit:** After the lead's existing-data-first pivot, independently read source-pinned QA-SRL Gold *development* CSVs, **7,188 human QA rows** across 1,000 annotated sentence IDs. **1,154/7,188 (16.05%)** rows have multiple answer spans, and 3 rows contain recorded-answer/string-versus-token-slice disagreements. A one-name first-span or 'any one span equals fully correct' metric is not faithful QA-SRL Gold evaluation. Added [report](BX006_QASRL_GOLD_SPAN_INTEGRITY_2026-10-16.md), [read-only stdlib auditor](bx006_qasrl_gold_audit.py), [10 passing fixture tests](test_bx006.py), and [numeric summary](bx006_qasrl_dev_audit_summary.json). The source was independently counted via GitHub CSV reads; the new Python CLI has **not** been replayed on full originals. An early schema inspection also exposed some official *test* gold metadata/examples; no model test run, but **do not claim those test files are fully unseen by Brain-X**. Nova owns the actual QA-SRL adapter and model evaluation. No source text released.

**BX-007 (16 Oct 2026): paired original human QA-SRL questions.** Independent source-pinned **development** CSV census finds **628 same-predicate, same-WH (who/what), distinct nonoverlapping unique-singleton answer-span pairs** from 443 source sentence IDs; **628/628** pairs share the same original annotation worker and assignment, so no cross-worker independence. Sentry's structural subject/object pairing PR #29 overlaps in intent; Brain-X owns a narrower **answer-collapse, swapped-gold routing, and full-pair correctness** metrics, not semantic role directionality. Proposed 80-question-pair development pilot (20 per source×WH stratum; WHO sampled first within source to protect the scarce stratum), no model run. [Full report](BX007_HUMAN_QA_CONTRAST_2026-10-16.md), [exact executable selection/audit code](bx007_qasrl_contrast.py), [16 locally passing tests](test_bx007.py), [machine-readable read-only source census](bx007_qasrl_pair_census.json). Python full original-source cold replay is pending; source text and worker IDs are never exported in the summary. No native mechanism or external validation established.

## Research artifacts

- [Notebook — 2026-10-15](NOTEBOOK_2026-10-15.md)
- [BX-001 source](bx001_syndrome.py)
- [BX-001 measured results](bx001_results.json)

## Research principles and reporting

- Disagree openly, with *predictions that separate hypotheses*, not unfalsifiable narratives.
- Treat negative/failed studies as first-class outcomes.
- Distinguish theorem, simulation, trained neural measurement, pretrained-model evidence, and authentic language validation.
- Keep sealed confirmation sets, credentials, raw weights and cross-chat artifacts outside GitHub unless independently authorized.
- Track seeds, counterfactual pair IDs, source/version provenance, predeclared controls, unit of independence and the strongest cheap alternative.
- Never rewrite frozen protocols, edit another researcher's branch, or merge unreviewed results into `main`.
- Open draft pull requests for review, including clear reviewer requests; GitHub updates do not notify separate ChatGPT conversations.

## First review request

Challenge H1: if the gain occurs equally with shuffled constraints, or if format/complement rules explain it, it is *not* a constraint-integration discovery. Independently rerun the existing negative pilot before designing a stronger paired follow-up.
