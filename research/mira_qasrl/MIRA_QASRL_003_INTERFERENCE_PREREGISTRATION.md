# MIRA-QASRL-003 — Relational interference preregistration v0.1

**2026-10-16 | Design only | No experiment run | No model results**

## Research question
Does *similarity between competing events* impair predicate-grounded question answering beyond the effects of event count, text length, question format and lexical shortcuts?

### Hypotheses
- **H_interference:** Conditional on the same target predicate and equivalent answer demands, adding a competing event with shared predicate lemma and/or participants reduces target-specific performance more than adding an equally long dissimilar event.
- **H_length:** Matched additions hurt equally, regardless of relational overlap.
- **H_shortcut:** Apparent interference follows recency, answer-position, surface frequency, voice or candidate count and disappears after balancing them.
- **H_interface:** Effect appears in one output interface but not another; do not infer semantic binding failure from a format effect.

## Frozen paired design
Use existing human-annotated QA-SRL Gold development questions and accepted source spans as *original* task gold. A human-reviewed contrast extension may create paired contexts only with explicit rights and labels; synthetic edits must be tagged GENERATED, never represented as original human gold.

For each human-confirmed target event construct or select:
- **C0** target event alone (diagnostic, not always original context).
- **C1** target plus *dissimilar* distractor event matched on added tokens, entity mentions and position.
- **C2** target plus *same-lemma* distractor, disjoint participants.
- **C3** target plus *shared-participant* distractor, different lemma.
- **C4** target plus same-lemma and shared-participant distractor.
- **C5** C4 with target/distractor order swapped; preserve the identity of the queried target and gold answers.

All conditions must preserve answerability, original target predicate identity, question intent and gold span mapping. Reject any transformation that changes the semantic gold or creates unresolved coreference.

## Primary endpoint and observability
1. Official answer-set QA-SRL score with all accepted spans retained; report separately by WH question type.
2. Event-selection correctness **only when independently calibrated event pointer/explicit event choice is available**. A text-only answer is event selection UNOBSERVABLE.
3. Conditional answer accuracy among independently verified correctly selected events, with explicit selection-accuracy denominator.
4. Exact-format validity and abstention.
5. Heuristic-vs-model delta against first/last participant, nearest predicate, last-mentioned event, name frequency, syntax-aware role extraction and lexical copying.

**Do not treat oracle event index as ordinary model performance.** Oracle may provide a diagnostic upper bound.

## Estimand and analysis
Primary paired contrast: score(C4)-score(C1), on *identical target items*, where added length and position are matched. Secondary: C2-C1, C3-C1, C5-C4. Pre-register one primary question type or report all with correction. Bootstrap **document/event-family clusters**, not individual correlated QA rows; include confidence intervals and all exclusions. Report by source, question type, predicate voice, answer-span multiplicity and model family. Estimate contamination/previous-inspection overlap.

A difference is evidence of behavioral interference only if it survives matched length, frequency, position, syntax-aware copying and output-format controls. It is not by itself evidence of a native Transformer binding mechanism.

## Sampling gates
- Existing QA-SRL DEV only; test untouched.
- Source-pinned rows and immutable identifiers; avoid lexicographic prefix sampling.
- Separate source original human labels from any new human adjudications and from LLM SILVER.
- Screen rights for source-to-model use and derivative transformations.
- Human reviewers independently adjudicate same-event vs distinct-event, target predicate, accepted spans, answerability, and ambiguity before inference.
- Keep the 133 repeated-answer-surface candidate rows separate: their event identity is **not** verified.
- Do not choose final cases after inspecting model outputs.
- Ensure at least two distinct pretrained families; exact checkpoints and prompts frozen; paid compute requires approval.

## Stop/failure criteria
STOP if a condition changes gold semantics, document rights unresolved, gold spans unobservable, pointer uncalibrated, leakage detected, or another team owns the exact run. A negative result is retained.

## Team boundaries
Mira: contrast suitability, label/provenance/rights checks, source-pinned candidate matrix.
Nova: official adapter cold replay and inference.
Comput: rival shortcut baselines, matching audit.
Brain-X: span integrity and pointer calibration.
Compass: causal mechanism only after behavioral gate.
Armaggedon/Sentry: freeze, ownership and acceptance.

## Motivation
MIRA-QASRL-001 found 133 of 6033 exact eligible single-span DEV QA rows with repeated answer surfaces at multiple positions. These are **not** validated distinct-event pairs. Brain-O E051–055 shows why zero/random-source controls matter: engineered pointer rescue can succeed without a learned source-side relational representation.
