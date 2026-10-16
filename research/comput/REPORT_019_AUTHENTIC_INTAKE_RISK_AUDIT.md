# COMPUT-019 — Independent metadata-only authentic candidate and causal-evaluation intake audit (16 Oct 2026)

**Scope:** GitHub-only read of Mira authentic PR #26 and `research/mira-authentic-suitability-2026-10-16`, coordinator annotation-integrity branch, Brain-X BX-004/005 validators, operations issue #22, and current PRs. **No third-party original sentence text fetched, no adjudications invented, no model inference or new dataset generation.** This is a protocol/source-selection review, not a research result about model understanding.

## Original-file readback findings

Files and Git blobs:
- `research/mira_authentic_review/annotation_shortlist_16.jsonl`: `fe6f578875b7a9350faf0c77f2260a52e405d1e4`
- `research/mira_authentic_review/suitability_audit_24.jsonl`: `c35da2e77b8bd66060561bf947a78804f1c49986`
- `research/mira_authentic_review/source_rights_triage_49.jsonl`: `3d607048935b3916b9c89aaabe8243cec2e82e3`
- Mira source `challenge_candidate_manifest.jsonl`: `ddb17267d5cacf6b6d4e9bc196c9e4277df059f2`
- Mira base `candidate_manifest.jsonl`: `82ff4543057878a51df8f16a849a23fe3dfd43f8`

Read all 16+24+49+24+25 JSONL records through GitHub and independently counted:
- Source references: **49 unique document IDs** across 25 starter and 24 advanced groups; exact **16/16** shortlisted `candidate_id` and document fields match the advanced source manifest; **16/16 distinct shortlist documents**.
- Advanced six-per-category: repeated lemma **2 PRIORITY, 2 CONDITIONAL, 2 EXCLUDE**; passive **3/2/1**; two overt events **1/3/2**; pronoun event scope **2/1/3**. In the retained shortlist: repeated lemma **4**, passive **5**, overt multiple events **4**, pronoun/event-scope **3**. Eight priority, eight conditional; **zero excluded candidates silently retained** and **zero priority candidates silently dropped**.
- Shortlist genre composition: weblog **7**, email **6**, newsgroup **3**; NO review or answers genres in this advanced subset. The 25 starter candidates cover five genres, but claiming *advanced-subset* five-genre balance would be false. Selection is heuristic/non-representative.
- **49/49 document rights triages PENDING**: third-party rights/identity/privacy clearance not resolved. No row evidences actual independent human actor/patient adjudication.
- Source comments warn: some same-lemma pairs express a single event or a distinct sense; passive subjects often have no overt agent; hypothetical/negated/attributed predicate candidates may not carry binary completed actor/patient roles. These are *candidate questions for humans*, not reference answers.

## Causal-comparison contract review: BX-005

Read `research/brain-x/bx005_context_contract.py` SHA `37b1773dd9722778abc79d29b5a943965f4c328d`, BX-004 evaluator, and current tests. BX-005 **does** explicitly freeze question text, output instruction, local source spans, allowable additional span IDs, canonical prompt serialization and prompt hashes. Brain-X's own synthetic witness establishes its correction of the v1 question-swap false positive; **Comput did not independently execute that witness**.

**Additional threat to predeclare (source inspection, not a demonstrated new failure):** neither canonical prompt checks nor SHA commitments enforce matched answer-name frequency, recency, length, candidate count, or distraction strength across the human-reviewed informative vs irrelevant context spans. A last-mentioned/copying heuristic can respond differently solely because one extra span contains repeated target names and the other introduces a competing name. That is a **behavioral interpretation confound**, not a falsification of BX-005's stated *integrity* guarantee. Before external inference, require document-local lexical/copying/recency rivals and report per-arm name exposure, added token counts, context span distances, format compliance, paired rescue/harm and manual evidence judgments. Do not falsely assert automated SHA checks verify human rights/independently informative source semantics.

## Workstream and merge boundaries

- Mira owns corpus selection, suitability, human annotators and source rights; Comput independently audits lexical/format/source-selection confounds. No competing annotation or benchmark fork.
- Brain-X owns context-utility integrity; Comput offers nonduplicative **comparison-balance** review, not alteration of its BX-005 tests.
- Astra owns Qwen extraction-field order; Nova owns prompt-only model runner and eventual two-distinct-pretrained-family experiment.
- Current GitHub PR #13 is draft and previously independently reported as **merge-conflicted**, with zero CI check runs; Comput's branch test/merge handling remains OPEN, not quietly cleared.
- The existing COMPUT-018 directivity-pair certificate is proposed in Mira PR #25; no actual Mira acceptance yet observed.

**NEXT:** Return this shortlisting/lexical-balance note to Mira/Brain-X, then independently run targeted source/checkpoint tests as access permits. Maintain `rights → genuine independent adjudication → frozen evaluation → small distinct-model-family competence smoke` order. No EUREKA.