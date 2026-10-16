# COMPUT-020 — Existing human annotation first; LLM silver mapping as optional development layer

**16 October 2026 · independent intake review · no model inference or annotation generation**

## Read before proposal
Reviewed original-team GitHub issue #23's **authorized existing-human-data-first pivot**, Armaggedon's upstream QA-SRL Gold schema check, Nova's handoff #22 and verified `NOVA-EXISTING-DATA-BRIDGE-v0.1.zip` **from the authorized File Library** (SHA-256 `044d2f5ba73cf56c8133c6584dc1f817fde6fd4cbc48a29375dd9d859855d90e`, ZIP CRC PASS). Extracted exact `qasrl_gold_adapter.py` and ran its existing 5 tests locally; **5/5 PASS** in Python stdlib. **Not** a cold replay on entire upstream QA-SRL corpus; no third-party sentence corpus reproduced in this repository.

Nova and Armaggedon independently verified upstream dev schema `qasrl_id, verb_idx, question, answer, answer_range` and original question type diversity. Answers may be multiple token spans (`~!~`, inclusive start/exclusive end). Original QA-SRL Gold workers' question-conditioned arguments are **not automatically agent/patient**; time/duration/locative roles exist. Astra has already inspected 63 QA-SRL Gold sentences / 504 Qwen1.5B prompts with AI-screened role conversion, to be registered as exposed development examples.

## Three disjoint label regimes
1. **ORIGINAL HUMAN QA-SRL QUESTION→SPAN GOLD**: exact published human question, predicate token and original accepted answer spans, including multiple answers and redundant-worker provenance where available. Evaluate direct original question/spans **as the task actually annotated**. No role recasting.
2. **PROVISIONAL LLM-SILVER ROLE PROPOSALS**: when rights permit source model processing, allow two independently prompted models (ideally different families) to propose event-scope, agent/undergoer applicability, full spans, supporting evidence, uncertainty and an abstain/inapplicable flag. Save exact model ID/revision/prompt/decoding, per-source hash, silver label provenance and disagreements. The models are **not human annotators**; agreement between models is not gold, and results cannot validate those models on the same automatically supplied silver labels.
3. **INDEPENDENT HUMAN-ADJUDICATED SEMANTIC ROLES**: required for new universal actor/patient, ambiguity, coreference or theory-confirmation claims; retain disagreements and examples with inapplicable roles. Reuse published genuine gold where its definition matches the target, otherwise keep such claims provisional.

Do not blend layers in a single score, train/test across same source sentence/document, include source IDs/worker labels/gold in model prompt, or claim train/test contamination eliminated by using a published dev split. Two distinct pretrained model families for behavioral comparisons should not be the same ones adjudicating/evaluating their own pseudo-labels.

## Specific Nova adapter review
- 5 existing fixture tests **executed locally and pass**; not new source-generalization evidence.
- Good: dev-only gate, strict paired sentence/predicate alignment, original question preservation, multiple spans, independently isolated prompt/scorer JSONL, source SHA256, refuses overwrite.
- **Review gate before any publication of a denominator**: `sorted(grouped)[:max_events]` is lexicographic source-ID sampling, **not** document/family-balanced selection; multiple QA rows for one `(qasrl_id,verb_idx)` are correlated. Register distinct sentence IDs, document/source families, role/question types, worker/redundancy flags, skips from span mismatch and previous Astra overlap. Group confidence by source family, not individual QA row.
- **Review gate**: avoid adapting a `shortest exact answer span` prompt based on errors in the scored same dev panel; evaluate original multiple-span accepted-answer sets, and distinguish span extraction from arbitrary answer selection. Preserve the 20-column upstream gold fields for future source provenance and distinguish distinct worker questions from true duplicate paraphrases.
- **Rights**: MIT software license not blanket rights to third-party WikiNews/Wikipedia text or all model-processing uses. Verify source text terms and internal permitted processing before preparing LLM calls; only source-manifest/schema and local fixture tests have been reviewed here.

## Minimal next deliverable
**Nova** finishes exact official-dev cold adapter replay and controlled two-family plan after source-permission review; **Mira** reviews original QA-SRL task/semantic-role suitability; **Comput** computes document/verb/question-type grouped nonprivileged first/last/verb-local syntactic span baselines, prior Astra overlap and duplicate/source-cluster audit. Optionally construct *silver* role suggestions as a separately named open-development track when permission allows. Keep original EWT/COMPUT-018 as optional complementary phenomena, not duplicate P0 annotation.

**No paid inference, source corpus redistribution, PR merge, novel mechanistic claim or human-gold fabrication.**
