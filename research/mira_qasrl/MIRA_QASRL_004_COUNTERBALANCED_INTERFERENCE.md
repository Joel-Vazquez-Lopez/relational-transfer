# MIRA-QASRL-004 — Counterbalanced interference and shortcut identifiability

**Status: research design, not empirical result. 16 October 2026.**

## New evidence from team
Sentry PR #29: 576/620 structurally eligible original QA-SRL subject/object pairs have subject answer before object answer (92.90%), 620/620 pairs same worker. This is a source-order *gold distribution* statistic, **not measured heuristic accuracy**. Brain-X BX-007 same-WH contrasts are a separate overlapping question type; do not conflate denominators. Nova v0.3 finds 3 span mismatches and 5 repeated predicate-question keys with differing spans. Comput reports SemEval-2010 Task 8 human directional nominal-relation gold, which is NOT event actor/patient gold.

## Identification problem
A model that answers target queries using first-mention/last-mention/nearest-verb, or a deterministic local parser, can appear to handle event competition. Original QA-SRL development examples are not order-balanced. Synthetic fixture C1 and C4 are not length-, lexical-, or frequency-matched and MUST NOT be used to claim an interference effect.

## Counterbalanced 2×2×2 core design
Factor A: competing event relational overlap: LOW vs HIGH.
Factor B: target event source order: FIRST vs LAST.
Factor C: gold answer location relative to target predicate: BEFORE vs AFTER, **where natural grammar permits**. If impossible to realize a cell without changing grammatical construction or gold meaning, mark NOT IDENTIFIABLE and use a matched alternative rather than manufacturing a misleading factorial.

Control added length, entity count, answer frequency, target lemma, number of predicates, question text, target event identity, and output interface within each matched block. Each block has independent adjudication of truth conditions, event identity, answer spans, and coreference. Passive/cleft variation may alter grammar; stratify it explicitly, not as an invisible swap.

### Minimal contrast requirements
- For each HIGH-overlap item, pair LOW-overlap with same target and answer.
- Reverse target/distractor order and *recompute source indices*, not answer strings.
- If distractor contains same answer surface, record all occurrences; textual answer alone cannot distinguish target selection.
- Keep one counterexample where a first-mention shortcut predicts the wrong answer, and one where last-mention shortcut predicts wrong.
- Independently calibrate any pointer interface before event-identity claims.

## Metrics and interpretation
Primary: paired HIGH−LOW difference in official human-gold QA score, with cluster bootstrap at document/event-family level. Secondary: interaction of overlap×order and model-minus-best-syntax-aware baseline; question-only, first/last, nearest-predicate and name-frequency are mandatory. Report raw per-cell denominators, answer-surface ambiguity and exact-format failure.

**Interpretation gates:**
1. HIGH−LOW drop disappears on counterbalanced order → position/recency explanation survives; no relational-interference claim.
2. HIGH−LOW drop persists but local syntactic parser matches model → behavioral interference, not distinctive model mechanism.
3. Drop persists after parser and question-only controls, in ≥2 pretrained families and held-out source groups → candidate robust behavioral finding, not mechanistic proof.
4. Pointer interface fails calibration → event identity UNOBSERVABLE, even when surface answer is correct.
5. Gold adjudicators disagree → ambiguous item excluded from primary endpoint and disagreement reported.

## Adjacent alternative: SemEval directionality
Use as **separate complementary study**, never relabel nominal Rel(e1,e2) into agent/patient. Balance relation class × argument order × lexical entities; track train/test provenance and pretraining exposure. Comput owns source/rights verification; do not duplicate its corpus replay.

## Ownership and current blockers
Mira: design, natural-pair admissibility and matched-block review. Sentry: order-skew warning, source rights/protocol. Brain-X: QA span/pointer validity. Comput: shortcut baselines and SemEval. Nova: source cold replay and inference only after coordinator gate. Armaggedon: final acceptance.

**No natural matched blocks validated, no pretrained inference, no new scientific finding from this note.**
