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
