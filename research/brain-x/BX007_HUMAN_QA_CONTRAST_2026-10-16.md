# BX-007 — Paired original human QA-SRL questions: a bounded answer-collapse discriminator

**Date:** 2026-10-16. **Status:** source-checked full development **eligibility census**, executable selector with **14/14 locally passing fictional-fixture tests**, protocol ready for independent cold source replay. **Zero real LLM predictions, no model inference, no new human labels, no publication-level discovery.**

## Team collision check and narrow ownership

Read latest coordinator #22/#23, Nova's official QA-SRL adapter work, Comput-020/021, Mira QA-SRL shortcut diagnostic, Brain-O E051–055, and Sentry's independent **draft PR #29** human-QA pair gate. **Sentry's PR #29 already owns the structurally explicit original human subject-slot vs object-slot pair certificate** (reported 620 pairs / 450 sentence IDs, all same worker). Its task may overlap with BX-007; it is not a second independent replication of BX-007. Avoid claims that the two lists are different without a source-level intersection audit.

**Distinct Brain-X task:** for original **same-wh** questions in the same original sentence and predicate, measure **answer collapse versus full-pair correctness**. Sentry tests explicit grammatical slot differences; BX-007 does not derive actor/patient, semantic directionality or native mechanisms. Nova owns any approved LLM inference/official QA-SRL task evaluation; Comput owns competent syntax/lexical/answer-copy baselines.

## Eligibility and original data

Upstream repository: https://github.com/plroit/qasrl-gs  
Pinned upstream Git commit: `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83`  
Sources: official `wikinews.dev` and `wikipedia.dev` human Gold question CSV joined to official source sentence/token CSV, read-only via connected GitHub.

For each candidate pair, require:

1. Same source `qasrl_id` **and exact `verb_idx`**; same original tokenized sentence and predicate.
2. Two **different original human-written questions**, but same original `wh` category restricted to **who or what**.
3. Each question has **exactly one original accepted answer span**; zero-based half-open coordinates validate against the exact original sentence tokens and match the original `answer` text.
4. Spans are **disjoint**, have **different normalized source-answer strings**, and each answer surface occurs **exactly once** in that source token sequence (so a string-only response is not already ambiguous by occurrence).
5. Any duplicate *normalized question* inside the predicate group is excluded rather than silently selecting a worker response. Multi-span gold remains valid in the full QA-SRL task, but intentionally outside this **small diagnostic subset**.

No original sentence, answer strings, worker IDs, or prompt rows have been committed to this GitHub branch. The eligibility census used a read-only independent JS quoted-CSV parser on actual official **development** files. The committed equivalent Python script was tested on locally created fiction fixtures; **its full original-data cold replay has NOT yet been performed**.

## Actual source-level census (not model predictions)

| Source DEV | Strict eligible question pairs | Same-wh “who” pairs | Same-wh “what” pairs | Distinct source sentence IDs | Same worker + assignment pairs |
|---|---:|---:|---:|---:|---:|
| WikiNews | **299** | 92 | 207 | **223** | **299/299** |
| Wikipedia | **329** | 28 | 301 | **220** | **329/329** |
| **Total** | **628** | **120** | **508** | **443** | **628/628** |

**Crucial interpretive caveat:** all **628/628** pairs were written by the **same QA-SRL worker in the same annotation assignment**, verified directly from original `worker_id` and `assign_id` metadata. There are **zero cross-worker independently corroborated pairs**. The questions are still human-authored, but their pairwise relationship is not independently human-reviewed. Distinct annotated spans/strings also do not prove distinct semantic referents: `She` and `Alice` could co-refer. Accordingly this is an **original question-conditioned span task**, not a verified actor-vs-patient directionality benchmark. Even for a positive two-family LLM result, a competent surface/verb-question syntax strategy may explain the entire effect.

## Small pilot selection — not launched

The committed stdlib selector takes at most **one eligible pair per source sentence**, choosing a SHA-256 rank fixed by source partition, source sentence ID and pair identifier. Proposed **20 pairs per each of four development strata**: WikiNews/Who, WikiNews/What, Wikipedia/Who, Wikipedia/What, for **80 pairs/160 original human questions**, if the approved cold source replay confirms all constraints.

It emits only aggregate counts and a hash commitment to selected private pair IDs; no original source text or scorer-only answer strings. There are 79/161 eligible WikiNews sentence IDs for Who/What and 23/200 Wikipedia sentence IDs for Who/What, so the 20-per-stratum quota is feasible under the available one-per-sentence constraints. **The exact panel-selection hash and source IDs are PENDING** the Python cold replay. This is an exploratory development panel, *not* prospective untouched confirmation; no verified document-level cluster identity exists yet. Before inference, the lead/Comput should separately check overlap with Astra's previously inspected 63 original QA-SRL sentences.

## Falsifiable predictions, statistics, and adverse explanations

For the same original human-written sentence and predicate, run two **original** questions separately, retaining model ID, revision, decoding, one-name/span output validity, prompt hashes, and raw responses for every attempted pair. The model should receive **the sentence + the human question only**, never source IDs, span indices, worker IDs or answer gold. For this matched **singleton** task, the conservative strict diagnostic checks each response against its own original gold span text; official QA-SRL metrics remain the project-facing evaluation.

Key per-pair outcome categories: both correct, exactly one correct, neither correct; whether **two nonempty responses are identical** (answer collapse); error transition by who/what and source; all invalid/empty response rates separately. A **question-blind fixed-answer strategy** necessarily has zero **both-correct** on these pairs under strict distinct-answer gold, even if it gets one question right. That is a necessary weak falsifier, not a strong lexical control and not a novel theorem. A competent **question-conditioned surface/syntax parser** could potentially perform very well with no abstract relational computation. Demand that rival from Comput.

**Overinterpretation to avoid:** answering two different questions with different output strings is not itself a relational mechanism, and collapsing to one name can be induced by response format, superficial question overlap, or systematic positional bias. Need matched competence, raw format validity, two **distinct** pretrained model families and eventual independently source-confirmed data. Question words change between the paired examples by design; unlike BX-005's informative-vs-irrelevant *fixed-question* experiment, BX-007 measures **question sensitivity at fixed source**, not contextual-information utility. Never pool the two outcomes as the same causal intervention.

For uncertainty use source sentence groups at minimum, with a further **document/source-family** audit before population generalization. Raw 628 overlapping pairs must never be treated as 628 independent events.

## Actual implementation, testing and reproduction

- [Executable selector/auditor](bx007_qasrl_contrast.py), byte-identical to locally tested source, **Git blob SHA `04d86eab89a61c868b5f45475cc3c1834fb84ca7`**, SHA256 `c404f80478243bc754414b68ad9a3adf3eafd6d2987238f37c8c61c85ea59d3b`.
- [14 regression tests](test_bx007.py), byte-identical to locally tested tests, Git blob SHA `48e48b1c23c4a31f0393ab898fd2d4ce6d446b2e`, SHA256 `e28a480724b2ad401579e132a993f32da9d4fef45d5aaf908456d3d6473be480`.
- [Machine-readable source-level census](bx007_qasrl_pair_census.json).

Tested locally: `python -m unittest -q test_bx007` → **14/14 PASS**, including source-gold answer uniqueness, span disjointness, original human question and WH pairing, same-worker flag + no ID leakage in aggregates, source test split rejection, one-pair-per-sentence selection, answer-collapse sanity. **No QA-SRL original-data full replay of this Python program is claimed**.

After approved access to pinned **development** CSVs, and *only* for this offline audit:

```bash
cd research/brain-x
python -m unittest -v test_bx007
python bx007_qasrl_contrast.py \
  --wikinews-gold /approved/data/gold/wikinews.dev.gold.csv \
  --wikinews-sentences /approved/data/sentences/wikinews.dev.full.csv \
  --wikipedia-gold /approved/data/gold/wikipedia.dev.gold.csv \
  --wikipedia-sentences /approved/data/sentences/wikipedia.dev.full.csv \
  --quota 20 --output bx007_dev_audit.json
```

Do not rename/insert a test source file and treat the CLI's `.test` guard as a true security or seal boundary. This workstream previously viewed brief official test metadata/examples during BX-006 schema exploration; that exposure was disclosed and **must not be called untouched confirmation**.

## Decision

**Promising low-cost REAL-LANGUAGE evaluation contrast available from existing published human QA gold.** Executed contribution today is the **source census and functioning selector**, **not** an LLM discovery. Sentry's structural gate partially overlaps; do **not** duplicate its analysis. Ask Nova for a bounded model-run feasibility review **after** source rights and cold adapter replay; ask Comput for a question-conditioned syntax baseline; retain Mira's authentically contextual, human-adjudicated case as a separate later hypothesis. No paid inference, external model download, sealed-data inspection, unapproved text circulation, or merge requested.

**Scientific signal: eligible question-conditioned distinction and identified annotation-dependence confound; no verified native mechanism and no EUREKA.**
