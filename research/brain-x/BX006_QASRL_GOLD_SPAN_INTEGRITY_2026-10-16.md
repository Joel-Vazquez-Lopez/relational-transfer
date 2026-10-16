# BX-006 — Human QA-SRL Gold span-integrity audit after the existing-data pivot

**Date:** 16 October 2026. **Status:** executed source-pinned public development-set audit, with locally tested stdlib reproduction script. No LLM inference, adapter tuning, annotation, sealed-data certification, or novel mechanism claim.

## Why this changed our Day 2 plan

The lead authorized **Nova** to begin initial authentic evaluation with *existing published human-labeled sources* rather than recruiting original annotators immediately (GitHub issue #23, 16 Oct). Specifically, QA-SRL Gold records original human-written predicate-specific questions with answer spans. **These are NOT automatically binary actor/patient gold labels.** The original QA-SRL Gold repository explicitly documents multiple answer spans and zero-based, start-inclusive/end-exclusive token ranges, sometimes multiple answers per question (separated by `~!~`). Its primary task and official evaluator must be respected. See: https://github.com/plroit/qasrl-gs and https://aclanthology.org/2020.acl-main.626/.

BX-004/005 was intentionally a gold-isolated **binary actor/patient, one-name** evaluator for *newly human-adjudicated contextual evidence*. It is **not** a valid drop-in evaluation for QA-SRL Gold. Naively treating the first accepted span as sole gold, or scoring **any** matched span as full answer-set correctness, would yield avoidable false positives/negatives.

## Exact material inspected

Source repository: https://github.com/plroit/qasrl-gs  
Pinned upstream commit: `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83`

Only these development files form the **analysis**:

- `data/gold/wikinews.dev.gold.csv` and `data/sentences/wikinews.dev.full.csv`
- `data/gold/wikipedia.dev.gold.csv` and `data/sentences/wikipedia.dev.full.csv`

Original CSVs were accessed READ-ONLY with the connected GitHub repository tool, parsed by an independent in-session JavaScript CSV reader, and joined by `qasrl_id`. The **new Python CLI** was tested on 10 *synthetic fixtures*, but has **not yet independently replayed against the full two original development CSVs**; reproduce that separately if the approved source bytes are made locally available. The aggregated outputs are committed in [`bx006_qasrl_dev_audit_summary.json`](bx006_qasrl_dev_audit_summary.json). No original source text, quoted sentences or worker IDs are committed here.

## Actual counts from the pinned development data

| Source split | Gold question rows | Annotated sentence IDs | Sentence reference rows | Rows with >1 accepted answer spans | Total answer spans | Span string ≠ indexed source tokens |
|---|---:|---:|---:|---:|---:|---:|
| WikiNews dev | **3,714** | 500 | 917 | **474** (12.76%) | 4,315 | **3** |
| Wikipedia dev | **3,474** | 500 | 917 | **680** (19.57%) | 4,523 | **0** |
| **Development total** | **7,188** | **1,000** | **1,834** | **1,154** (**16.05%**) | **8,838** | **3** |

All indexed answer span ranges on the two dev sources are syntactically valid and within the source token bounds; no `qasrl_id` join missing. The 3 WikiNews answer-string mismatches are consistent with **accented characters represented by question marks** in the gold answer string while the source sentence token text retains those accented characters. They are **not** automatically an incorrect role or token range; quarantine them for independent reconciliation, never silently modify the upstream gold.

The two dev sets contain **5 duplicate (sentence ID, verb index, question) keys**, potentially from multiple gold annotations. Do not blindly deduplicate or union spans without preserving worker/annotation provenance. Source sentence IDs are *not verified document IDs*, and correlated questions from the same sentence are not independent research examples.

Some multi-answer rows list several **distinct arguments**, not paraphrase synonyms. A system naming one accepted span is not thereby complete on a question requiring several spans. The conservative local `exact_span_match` therefore returns `None` (abstain) for multi-span rows and source-string disagreements; for singleton source-matched spans, it checks strict normalized text only. This is **a diagnostic, not the official task score**; do not convert it into a paper headline. Official QA-SRL evaluation may use richer matching/recall behavior than this diagnostic.

## What was coded and tested

- [Read-only stdlib dev integrity CLI](bx006_qasrl_gold_audit.py), exact tested local Git blob **`39ff2f110c6d090021c19c7856c47898111f4e59`**, source SHA-256 **`aa63bc70caf9969ef24f759f1683e56e7e9ded03a346622187afc3fc522b63e9`**.
- [Ten unit tests](test_bx006.py), exact tested local Git blob **`b98fee9e38945744e968199e605656ead483dae2`**; **10/10 passed**.
- [Read-only data metadata summary](bx006_qasrl_dev_audit_summary.json).

Test coverage includes multiple answers (abstain under single-response completeness), zero-based inclusive/exclusive bounds, bad syntax and out-of-range coordinates, corrupted accented gold (quarantine), duplicate questions, no automatic actor/patient conversion, repeated surface forms (cannot locate exact occurrence from string alone), and rejection of a `.test` split.

Run only on authorized locally obtained **development** copies:

```bash
cd research/brain-x
python -m unittest -v test_bx006
python bx006_qasrl_gold_audit.py \
  --sentences /authorized/source/data/sentences/wikinews.dev.full.csv \
  --gold /authorized/source/data/gold/wikinews.dev.gold.csv \
  --partition wikinews.dev
```

The CLI intentionally rejects names containing `.test`, but this is not a general security boundary: it does not prove a renamed file is non-test. A reviewer must validate original file lineage, SHA and source split externally.

## Research integrity — disclosure about test exposure

During the initial source-schema exploration **before enforcing the dev-only rule**, Brain-X also opened the official WikiNews/Wikipedia **test** CSVs and saw **aggregate metadata and a few examples of their answer text**. I did **not** run any trained model on them, create an evaluation score, adapt a prompt/model to those examples, or use them in the above development counts. Nevertheless, these test files are **not pristine/unseen for this investigator**. This must be recorded and considered by the lead when selecting a prospective confirmatory split. Do not characterize the official test set as sealed from *all* investigators; reserve a separate untouched source or conduct an explicitly labeled non-blind evaluation.

## Ownership, limitations and next scientific test

- **Nova owns** QA-SRL Gold download/adapter, original human question/answer span evaluation, prompt isolation, checkpoint provenance and two different pretrained model families.
- **Brain-X owns** this **independent integrity and metric-confound review** only. It does not rerun Nova or invent actor/patient supervision.
- **Comput owns** semantic/lexical shortcut and question-position heuristic competition; could independently challenge this audit's duplicate-question and span-set treatment.
- **Mira owns** prospective EWT and contextual contrastive authentic semantic adjudication, an option for later confirmation.
- **Astra** field-order and **Brain-O/Compass** causal/composition controls remain separate.

Potential Brain-X follow-up (requires approval/rights and source-appropriate annotation): a human-verified *contrast of independently informative versus irrelevant context* on the **same original QA-SRL question and predicate**, reporting rescue **and** harm against a source-only copying/lexical baseline. Ordinary sentence-local QA-SRL Gold does not itself supply such context-intervention gold; **do not fabricate that intervention from generated paraphrases**.

This audit establishes a concrete metric-validity prerequisite, **not** meaningful natural-LLM relational generalization or a new computation principle. No human local model run or paid inference requested.
