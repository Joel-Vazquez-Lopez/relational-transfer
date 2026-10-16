# Predicate-specific independent human-annotation protocol (DRAFT)

**Not yet executed:** Both independent annotator templates are blank; no model, LLM, UD grammar parser, or second AI agent is a replacement for two HUMAN annotations.

1. From the pinned upstream CoNLL-U file, resolve the **exact** `source_sentence_id` and `predicate_token_id`. Read adequate neighboring original context where source rights permit. Do not use predicted gold from labels or other annotators.
2. Define the *particular event* described by the predicate (including phrasal/light verbs). Record `predicate_scope_description` and `sense`; verb identity alone may be ambiguous.
3. Record **full referring expressions** for agent/causer and patient/undergoer where the intended semantic roles apply. If a verb assigns experiencer/theme, or if the subject is an event/inanimate cause, mark `one_applicable`, `neither_applicable`, or `uncertain` as appropriate and explain.
4. Values for `role_applicability`: `both_applicable`, `one_applicable`, `neither_applicable`, `uncertain`. Use `UNEXPRESSED` for omitted arguments, `UNRESOLVED` for reference needing unavailable context. Blank means **not annotated**, not empty semantic role.
5. Mark `ambiguity`, `coreference` as `yes`, `no` or `uncertain`; record confidence and rationale. Preserve ambiguous and out-of-scope items, including subject/object disagreement.
6. Reviewers A and B independently complete separate sheets before comparison. Only after both are locked may an adjudicator compare role applicability, spans, event scope and uncertainty. Track full agreement, partial overlap, discrepancies and retained unresolved status, with *human* adjudicator ID and date.
7. Reviewer-facing forms **must not** be prefilled with UD `nsubj/obj` heads, actor/patient labels, source generator graphs, or model outputs. Source text itself is fetched from approved source rather than duplicated in this repository.
8. Reserve new documents and prepare no LLM prompts until rights/PII clearance, full semantic adjudication and baseline plan are signed off. Syntactic extraction is only a candidate selection aid. Never equate `nsubj/obj` with universal agent/patient roles.

**Acceptance gate:** 25 candidate IDs/source IDs verified; all recorded scope and semantics human-authored; two independent review files; adjudication table with no silent exclusions; source-by-source rights and provenance; document-grouped error rates. Then a *separate*, untouched, document-disjoint confirmation design.
