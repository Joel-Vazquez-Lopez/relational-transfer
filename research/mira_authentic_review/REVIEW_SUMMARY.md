# Mira Day 2 — Authentic dataset suitability screen (16 October 2026)

**Status:** COMPLETED read-only investigator triage of 24 advanced EWT challenge groups; **not** independent semantic annotation; EWT text rights unresolved. This stacked review branch intentionally leaves the original 49-group candidate manifests untouched.

## Quantitative result

| Suitability class | Groups | Interpretation |
| --- | ---: | --- |
| PRIORITY for independent human annotation | **8** | Potentially informative, but include negative/abstention and non-canonical cases |
| CONDITIONAL: annotate only with explicit scope/modal/sense caveat | **8** | Potentially useful control; not allowed to be silently converted to a positive binary answer |
| EXCLUDE from main event-binding evaluation | **8** | Wrong scientific construct, same-event repetition, idiom, asserted/hypothetical mismatch or trivial pronoun |
| **Shortlist** | **16** | 16 different documents; **24 target predicate annotation slots**, all UNLABELED |

**Explicit disposition (preserves all negative data):**
- PRIORITY: `EWT-CH-002,004,007,009,011,016,019,024`
- CONDITIONAL: `EWT-CH-001,003,008,012,014,015,017,022`
- EXCLUDE from main outcome: `EWT-CH-005,006,010,013,018,020,021,023`

Every reason and the required human disambiguation is in `suitability_audit_24.jsonl`. Exclusions remain present for negative-control/corpus-reliability reporting. This is a **researcher-read semantic suitability triage**, not two independent blinded human judgments.

## Scientific insights uncovered

- **Not every repeated lemma denotes a distinct event:** `EWT-CH-005` uses the evacuation predicate across nearby clauses but appears to describe one activity; it is a strong *negative counterexample*, not suitable for testing event selection.
- **Syntax does not imply an applicable binary actor/patient label:** `EWT-CH-006` participial named, `EWT-CH-013` abstract give/afford, and idiomatic `EWT-CH-021` should not be forced into a naive semantic-role answer.
- **Factuality and modality matter:** `EWT-CH-008` negated confirmation, `EWT-CH-012` planned/obligatory pledge, `EWT-CH-017` asserted alert reception with alternative hypothetical forwarding, and `EWT-CH-018` conditional actions change whether the supposed event happened at all.
- **Informative negative controls are valuable:** `EWT-CH-007` and `EWT-CH-009` are passives lacking an expressed agent; a proper model can abstain rather than invent a named actor.
- **Pronoun presence alone is not genuine coreference complexity:** `EWT-CH-020,023` are weak; `EWT-CH-024` is more promising but needs wider context and independent verification.

## Files and uses
- `suitability_audit_24.jsonl`: original manifest fields plus one case-specific triage, preserved rejections, one required human test.
- `annotation_shortlist_16.jsonl`: ID-only, source-pinned 16-group candidate package; **not gold**.
- `shortlist_annotator_A_blank.csv`, `shortlist_annotator_B_blank.csv`: 24 distinct target-predicate rows each; no privileged UD subject/object heads or guessed semantics.
- `source_rights_triage_49.jsonl`: document-specific 49-row rights/ethics **PENDING** ledger, zero clearances.
- `RIGHTS_AND_ALTERNATIVES.md`: explicit EWT limitation and more predictable-license parallel source options.
- `BEHAVIORAL_PRE_SPEC.md`: predeclared falsifiable real-LM scope/interface study, but no inference budget, model identity or frozen exact prompts yet.

**Data origin:** UD English EWT dev `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`, Git blob `120d04140cf4833690117bc74c08c8da861bbce6`. Original 25+24 group files remain on the parent research branch. No raw original text is committed. Parent [PR #26](https://github.com/Joel-Vazquez-Lopez/relational-transfer/pull/26) must remain draft while rights, human annotation and independent baseline checks are open.

## Key decision
**EWT: keep as rights-pending open linguistic development, not an externally redistributable licensed benchmark.** Explore article-revision-pinned English Wikipedia, with page-specific rights/attribution and unchanged-authentic-versus-derived-context lineage, as a parallel route. The existing 16-group shortlist is feasible for asking real human reviewers to examine **once text-use rights are cleared**, but is NOT a completed annotation-ready unrestricted package.

## Independent replication still due
- External reviewer must rerun `verify_authentic_intake.py` against byte-exact pinned CoNLL-U and audit 24 manual case dispositions (neither check can be replaced by Mira's own review).
- Two genuinely independent human annotators and an adjudicator must fill separate locked files; preserve all disagreement, inapplicable and unresolved labels.
- Rights reviewer must sign off each actual original text source and intended model-processing/export path.
- Comput should attack the proposed contrasts with source-only/position/verb-choice baselines.
- Nova and Astra should not infer on EWT source text until rights, gold and calibration gates pass.

**Novel mechanistic finding:** NONE. **Pretrained evaluation:** NOT RUN. **Source rights approved:** 0/49.
