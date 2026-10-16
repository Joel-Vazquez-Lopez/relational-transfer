# Authentic challenge supplement — Day 2, 16 October 2026

**NOT A BENCHMARK.** Read-only, source-pinned IDs for 24 potential advanced *development* event groups. **Zero human semantic gold**, zero complete rights clearances, zero model runs. The source of all event categories is UD syntactic structure, not independently adjudicated semantics.

## Audit and sampling
- Exactly **24 different source documents**, zero overlaps with the 25 documents in the initial [P2 manifest](candidate_manifest.jsonl); total combined **49 different documents**.
- Four purposefully hard candidate categories, six groups each:
  1. `passive`: source token has `nsubj:pass`. An overt agent is **not** guaranteed; annotate implicit, inapplicable or unresolved roles honestly.
  2. `two_overt_event_candidates`: two verbal heads each have grammatical `nsubj` and `obj`. These need human verification as *two distinct meaningful events*.
  3. `repeated_lemma_nearby`: the same lemma occurs in two sentences no more than three corpus sentence slots apart. **Not proof of two distinct semantic events**; two reference sentence IDs are mandatory and their full scope is reviewed together.
  4. `pronoun_event_scope`: includes pronoun and multiple verbs, with a verbal head having subject/object syntax. **Not labeled coreference**, just a candidate for event-scope ambiguity.
- Group count 24, source-sentence references **30** (six two-sentence groups), target predicate annotation slots **36** (two in repeated-lemma and two-overt-event groups).
- These 24 source groups were inspected for eligibility by automated heuristics, so **NONE** should later be rebranded sealed confirmation. They are purposefully selected, not a representative corpus sample. Genre balance is also not guaranteed in this supplement.
- Pinned source: [UD English EWT v2.18 May 2026](https://github.com/UniversalDependencies/UD_English-EWT/blob/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/en_ewt-ud-dev.conllu). Exact `sent_id`/target token/lemma reference checks were performed while generating this manifest, and review requires independent readback.
- No original sentences or role labels are committed, nor any LLM prompt containing source/gold metadata.

## Independent annotation contract
The two entirely blank forms `challenge_annotator_A.csv` and `challenge_annotator_B.csv` have **36 rows** each, one per syntactically identified predicate slot. A human should fill full spans, event sense/scope and role applicability. A repeated-lemma group must be reviewed as a **pair**. A person may annotate `not_distinct`, `inapplicable`, `unresolved`, missing actor or referent, and preserve cases requiring wider document context. These are **valid outcomes**, not errors to hide. No pseudo-gold from UD dependencies.

Before annotation or LLM evaluation, complete `challenge_rights_review.csv` for each of the **24 source documents** and record allowed original text access, privacy and model-processing status. The upstream annotation license CC BY-SA 4.0 does **not** establish blanket rights for source text from multiple third parties.

## Falsification gates and next experiments
- Independent source-verifier must confirm all 30 exact source IDs and 36 predicate token/lemma references; duplicated documents between P2 and challenge trigger FAIL.
- Human review must establish that repeated-lemma pairs really express different event occurrences, multi-event candidates are independently queryable, and pronoun-bearing examples have resolvable reference if included in scored evaluation.
- Ask Comput to freeze source-only repetition, subject/object/voice, closest-mention, and last-event name baselines *before* seeing any confirmed model results.
- Astra keeps its own controlled field-order experiment. Nova only runs two **different** untouched pretrained model families after format/competence, rights and double-human-label gates.
- Synthetic editing of these original sentences would produce **derived development stimuli**, not naturally authored data. Keep original authentic language and edits distinguished.

Status: **CANDIDATES BUILT / EXTERNAL RIGHTS AND TWO-HUMAN ANNOTATIONS BLOCKED**.
