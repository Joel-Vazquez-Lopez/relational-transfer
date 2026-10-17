# Linguistic Development Observatory
### Our exploratory lab for looking inside the Transformer black box

**Scientific question:** How do linguistic distinctions influence the *complete development* of computations across Transformer tokens, layers, and within-layer operations? Which aspects of those developmental responses recur across related but separately trained language models, and which differences matter?

**Research lead:** Joel Vázquez López. **Exploratory collaborator:** GPT-6 / Armaggedon. **Planned active ground investigation:** from **23 October 2026** (not an automatic scheduled start). **Status:** workspace prepared; no new inference authorized or performed by this setup.

> Our original observation is NOT “two LLMs have the same vectors.” In Qwen 7B and Llama 3B, a controlled role reversal produces qualitatively related *up–down–continuity patterns* across subject, verb, object and question-token positions, with unequal amplitude and timing. Is that a meaningful regularity of linguistic processing, a general Transformer schedule, or some mixture? We want to **look at the entire development first** and discover how to tell.

**Lab discussion and running decisions:** [GitHub Issue #35](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/35). This is our dedicated ground-level laboratory; Codex's formal work remains in Issue #34.

## Why language, and why this scope?

Linguistic phenomena are our experimental microscope, not an incidental benchmark. We can define what must change and what should remain invariant without looking at the neural representations: who did what to whom; active/passive descriptions of one event; novel entities; negation, event choice and question-dependent retrieval. These give us meaningful perturbations with which to inspect how the black box works.

Start from **role reversal versus meaning-preserving voice change** with carefully handled name identity, tokenization and surface order. Unfamiliar names help reduce reliance on memorized entities but do not eliminate lexical and tokenization shortcuts. Keep the investigation small and comprehensible before expanding to discourse, factuality, style or generation.

We are interested in (i) the evolving hidden states, (ii) the attention, MLP and token-interaction contributions which generate their changes, and (iii) where those changes have demonstrable consequences for linguistic behavior. Full development means inputs → tokens → embedding → each layer/block → final readout; optionally successive generation tokens and training checkpoints as **separate** timescales.

## Division of labor

- **This workspace — research lead + Armaggedon:** ground-level, *open-ended exploratory* investigation: inspect all layer/site curves; hypothesize what drives their shared trends and differences; propose new mathematics; keep anomalous and messy results; design multiple credible explanations. Observational findings are allowed before causal identification as long as they are labeled descriptive.
- **Codex / method frontier:** independent adversarial experiments and rigorous prospective tests on [its existing branch](https://github.com/Joel-Vazquez-Lopez/relational-transfer/tree/research/causal-developmental-relations-2026-10-17), tracked by [Issue #34](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/34). We do not overwrite its protocols, use its holdouts for method selection, or reinterpret failed gates.
- **Other research agents:** retain their independent event-binding, falsification, datasets and theory branches. Communicate promising hypotheses in the shared issues, not by silently assigning work.

## Open documents

1. [VISION.md](VISION.md) — intellectual center, the original observation, questions and scope.
2. [EVIDENCE_AND_BOUNDARIES.md](EVIDENCE_AND_BOUNDARIES.md) — what existing plots, runs and negative controls actually establish.
3. [OBSERVATION_NOTEBOOK.md](OBSERVATION_NOTEBOOK.md) — living observations and a template for exploratory work.
4. [MATHEMATICAL_SANDBOX.md](MATHEMATICAL_SANDBOX.md) — possible representations of developmental similarity, their limits and toy counterexamples.
5. [ROADMAP.md](ROADMAP.md) — 23 October start, focused exploration, and handoff to validation.
6. [READING_MAP.md](READING_MAP.md) — neighboring literature and boundaries; references are leads, not novelty certificates.

## Evidence hierarchy

**Observation → explanation candidates → distinctive predictions → frozen controlled tests → independent confirmation.**

We can explore freely and record unexpected patterns without pretending they are mechanisms. Conversely, a failed confirmatory gate must stay failed. We value a strong narrow/compositional contribution (new use of known ideas) as much as an elusive foundational breakthrough.

**Do not claim:** identical algorithms from similar curves; semantic-role computation from token/name copying; computation onset from patch sufficiency; cross-model coordinate equivalence from separately fitted PCA; generalization from four exposed pairs; successful experiment from source hashes alone.

## Where data actually lives

The Codex branch holds the [role-path and factorial reports](https://github.com/Joel-Vazquez-Lopez/relational-transfer/tree/research/causal-developmental-relations-2026-10-17/research/causal_developmental_alignment_2026-10-17). Some full-dimensional NPZ state arrays are **local to the research machine** and are not implicitly present in this new branch. Treat raw-state access as unverified until paths and hashes are checked. Do not upload checkpoints, licensed corpora, private sources or large model weights to GitHub.

For approvals, novelty and downstream experiments see [Issue #31](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31), [Issue #30](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/30), [Issue #33](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33), and [Issue #34](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/34).

**This is a living laboratory, not a paper claim or a mandate to fit one preferred metric.**
