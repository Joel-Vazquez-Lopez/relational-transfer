# MIRA-QASRL-002 — Event-identity observability gate (design only)

**Status:** Pre-inference design; no new dataset labels or model evidence. Owner: Mira successor. Date: 2026-10-16.

## Falsifiable question
When multiple source occurrences can yield the same answer string, does a correct *surface response* conceal selection of the wrong predicate/event? A correct answer string alone is insufficient to answer this question.

## Candidate strata (human adjudication required)
A. Same answer-token surface at two or more source positions; ask whether occurrences correspond to different event arguments, aliases of the same referent, or merely repeated text.
B. Two or more predicate indices with the same lemma, human-confirmed distinct event instances and distinct argument spans.
C. Same lemma but one event or scope ambiguity; retain as explicit negative/exclusion control.
D. Multiple predicates with different lemmas; generic multi-event control.

QA-SRL Gold original question, predicate token index, accepted span set and source provenance are primary gold. Do not infer universal actor/patient gold from QA-SRL WH or passive fields.

## Primary observability test
For each item, score three *separate* outcomes:
1. Gold answer **surface match** under the official QA-SRL evaluation contract.
2. **Source occurrence evidence**, only if output includes a verifiable span index or a reliable, independently calibrated pointer. If the model returns only text, mark occurrence **UNOBSERVABLE**, not correct or wrong.
3. **Event identity evidence**, only if the response is anchored to a human-verified target predicate/event, not inferred from a shared answer string.

Do not present model-generated token indices as trustworthy without an independently measured pointer-validity calibration. Span pointers are a separate interface intervention and may change task difficulty.

## Rival baselines and controls
- First/last named participant; nearest-to-predicate token; last-mentioned event; name frequency; surface-copy.
- Syntax-aware local predicate-argument extraction with passive handling (Comput owner).
- Repeated-answer-surface negative control with one genuine event.
- Same-name-bag, distinct-role orientation where human gold and rights permit.
- Original question text, answer set, and target predicate frozen across matched arms.
- No source IDs, gold spans or target answers leaked to model prompt; preserve prompt hashes and raw outputs.
- Sample by source document/event family, not independent QA rows. Record overlap with Astra's previously inspected 63 sentences.
- Existing QA-SRL development data first. Test partitions not used for discovery.

## Decision table
- Surface correct + occurrence unobservable: **surface success only**, not evidence of event binding.
- Surface correct + occurrence verified wrong: **demonstrated answer-event dissociation**, if pointer interface is independently valid.
- Surface correct + occurrence verified right: **event-specific behavioral success**, not proof of native pre-question binding.
- Event correct + participant wrong: **event selection / argument retrieval dissociation**, requires independent event and argument gold.
- Heuristic baseline matches model: **no advantage beyond baseline**.
- Event correctness tracks manipulated prompt cues but also recency/frequency: **confounded**, not mechanism evidence.

## Ownership and stop conditions
Mira successor: candidate stratification, manual provenance, label scope and exclusion accounting. Brain-X: span validity / pointer calibration. Comput: shortcuts. Nova: cold adapter replay and any future inference. Armaggedon: source rights, freeze, evaluation approval. Sentry: collision check.

STOP if rights/model-use are unresolved, gold does not distinguish events, pointers are uncalibrated, or cases are selected after inspecting model predictions. A null/ambiguous result is an acceptable outcome.

## Current evidence
MIRA-QASRL-001: 7,188 original DEV QA rows, 6,033 eligible exact single-span rows, 133 answer surfaces with multiple token positions. This is **not** a count of confirmed event collisions. No model run was conducted.
