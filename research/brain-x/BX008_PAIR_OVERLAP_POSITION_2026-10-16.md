# BX-008 — Independent paired-QA overlap audit and position-balanced development gate

**16 October 2026 · Brain-X independent retrospective analysis.** No new model inference, tuning, synthetic model training, user GPU run, source redistribution, test-split evaluation, third-party annotation, or publication-level mechanism result. **Current GitHub preflight**: [coordinator ledger #31](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31#issuecomment-6211786148), awaiting owner verdict for any future experimental cells. A completed **read-only source analysis** is documented separately from model-run permission.

## Immediate scientific decision: three separate tasks, do not conflate

1. **Sentry PR #29:** human QA-SRL original **subject-slot versus object-slot syntax question** eligibility, including mixed-WH pairs. 620 eligible pairs / 450 sentence IDs; all same-worker; **576/620 (92.90%)** original subject-slot gold spans precede object-slot spans. This latter number is *conditional on privileged gold spans* and **NOT** the measured success of an end-to-end baseline.
2. **Brain-X BX-007:** same original predicate/sentence, same original WHO or WHAT question category, **two distinct original human questions and two exact disjoint singleton answer spans** with unique source occurrence, any QA structural template. 628 overlapping pairs / 443 sentence IDs; all 628 are from **the same original worker AND same annotation assignment**. It is question-specific answer-span **selection**, not original annotated universal actor/patient semantics.
3. **Brain-X BX-005:** same-question source-text informative-versus-irrelevant context utility. **Coordinator reply: NOT TESTABLE ON QA-SRL GOLD AS RELEASED** (no original human-validated matched contextual interventions or dependable neighboring-document links). Must wait for a distinct licensed corpus and independent human contextual adjudication. Do not rename BX-007 as BX-005's intervention.

## Exact source-level overlap — NEW bounded retrospective

Read the original WikiNews and Wikipedia **DEV** gold and sentence CSVs at pinned upstream QA-SRL Gold commit `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83`. Reconstructed the **published eligibility rules** from [Sentry's actual Python script](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/sentry-human-qa-pair-gate-2026-10-16/research/sentry_qasrl_pairs/qa_pair_eligibility.py) and [Brain-X BX-007 source](bx007_qasrl_contrast.py). Intersected **exact source sentence ID, predicate index, and unordered pair of original human question strings paired to the original gold span ranges**. No entity names/source sentences/worker IDs are exported in the result.

| DEV source | Brain-X pairs | Sentry pairs | Exact shared pairs | BX-only | Sentry-only |
|---|---:|---:|---:|---:|---:|
| WikiNews | 299 | 330 | **96** | 203 | 234 |
| Wikipedia | 329 | 290 | **121** | 208 | 169 |
| **Total** | **628** | **620** | **217** | **411** | **403** |

The programs do **not** select identical evidence: ~34.6% of BX-007 pairs are exact matches to Sentry's list and ~35% of Sentry's match BX-007. **But same upstream corpus/annotator and overlapping source worlds mean these are NOT independent scientific replications.** The remaining 411 BX-only pairs supply variation for a complementary **same-WH answer-selection** panel, but eligibility is not semantic gold. A source-level script and dedicated tests are preserved so this distinction is independently checkable.

## Privileged source-span order analysis — NOT a raw-text model baseline

For each BX-007 pair sorted deterministically by normalized **original question string**, compare *original human gold span* start positions:

| DEV source / original WH | First question's gold span EARLIER | First question's gold span LATER |
|---|---:|---:|
| WikiNews WHO | 52 | 40 |
| WikiNews WHAT | 110 | 97 |
| Wikipedia WHO | 13 | 15 |
| Wikipedia WHAT | 173 | 128 |
| **Total** | **348 (55.4%)** | **280 (44.6%)** |

The arbitrary canonical **question-order vs gold-span-position** association is much closer to 50% than Sentry's 92.9% subject-slot-before-object relationship. These are **different conditionals**, not direct estimates of the same heuristic's performance. Gold span positions and selected answer candidates are *privileged scorer-only information*. A sentence-level syntax-aware reader or question-lexical copier may still beat the model or explain all observed errors. Do not use 55.4% as a measured actual raw-text baseline.

## Concrete matched-panel feasibility and code

A separate independent source-only **lexicographic-greedy feasibility check** found **40 candidate pairs per source** with one unique source sentence ID per pair, 10 WHO-EARLY, 10 WHO-LATE, 10 WHAT-EARLY and 10 WHAT-LATE. Across WikiNews and Wikipedia this supports **80 development pairs** at 10 per each of eight source×WH×gold-position cells. Source text, gold answers, worker IDs and per-case identifiers remain unshared.

However that lexical test is **NOT** the exact final hash-selected panel. We now provide [`bx008_nonredundant_pairs.py`](bx008_nonredundant_pairs.py) implementing a deterministic **SHA256-ranked bipartite matching** for 10 EARLY + 10 LATE distinct sentence IDs in each source×WH stratum, reserving scarce WHO before WHAT. It **fails closed** if requested balance or uniqueness is unavailable, rather than silently degrading a quota. It compares original BX and Sentry eligibility and emits only counts and SHA-256 of the chosen *private* pair IDs. The exact **80 selected original pair IDs/hash remain PENDING** the source-pinned full **native Python cold replay**.

The source code only *tests metadata and sampling logic*, not license authenticity, unbiased author/source-document groups, question ambiguity, source-frequency shortcut competence, gold co-reference, independent human adjudication or real-model correctness. One sentence ID does **NOT** prove a unique article/document. An exploratory subset of repeated Wikipedia content may be memorized by pretrained models. User's old Astra 63 inspected QA-SRL events and the investigator's previous exposure to limited official test metadata must remain in the provenance register. All computed development numbers are from read-only upstream source analysis through the connected GitHub read-only JS parser, NOT falsely attributed to running our native Python CLI on original bytes.

## Verified code and tests

- [BX-008 reproducible source](bx008_nonredundant_pairs.py), exact locally tested and committed Git blob **`427e424ab12ece219e0471720637a916b0ec6be2`**.
- [12 additional adversarial fixture tests](test_bx008.py), exact locally tested and committed Git blob **`fde0c44dcc70f7333982206d82d31adaa4794f47`**.
- Previous BX-007 source/test unchanged. **28/28 local unit tests pass** (`16 BX-007 + 12 BX-008`). Source reads and script SHA identities were checked after GitHub write. Test suite runs Python stdlib only, with mock rows containing fictional human questions; no original corpus text or gold is included.
- [Machine-readable source aggregate](bx008_overlap_position_audit.json), no original per-row gold.

Usage after authorized local access to the **exact pinned DEV** CSVs:

```bash
cd research/brain-x
python -m unittest -v test_bx007 test_bx008
python bx008_nonredundant_pairs.py \
  --wikinews-gold /approved/data/gold/wikinews.dev.gold.csv \
  --wikinews-sentences /approved/data/sentences/wikinews.dev.full.csv \
  --wikipedia-gold /approved/data/gold/wikipedia.dev.gold.csv \
  --wikipedia-sentences /approved/data/sentences/wikipedia.dev.full.csv \
  --per-side 10 --output /tmp/bx008_dev_audit.json
```

Do not call any set sealed just because a filename contains `.dev`. Independent source-file Git-blob and checksum verification is required by the recipient. The program's output status `DEVELOPMENT_METADATA_ONLY_NOT_APPROVED_MODEL_RUN` is explicit.

## Scientific endpoint and go/no-go

**Behavioral hypothesis:** a pretrained LLM may produce one or both *correct source answer strings*, yet systematically send them to the wrong original question (swapped-gold routing) or answer both with one repeated string (collapse). These are two **different** error types. A well-paired 80-case **development** set can quantify them, but official question→answer-span scoring remains Nova's primary metric. **Both-correct paired span performance does not prove abstract semantic binding.** We need a competent *question-conditioned* syntax/lexical copier baseline and two distinct pretrained model families after approval.

**Collision decision:** reuse Sentry's structural subset as an optional separate stratification; Brain-X's source-only comparison contributes an exact measured **217-pair intersection**, not another structural gate or duplicated LLM run. Neither Brain-X nor Sentry should independently trigger Qwen/Phi calls: Nova owns model inference. The first accepted *cold* original-data SHA/pair selection and rights audit should decide whether to include an 80-pair optional diagnostic in Nova's existing run. If the syntax copier already explains model behavior or no uncontaminated rights-cleared source sample can be constructed, stop; do not invent a better-looking benchmark.

**Current state:** SOURCE-LEVEL RETROSPECTIVE COMPLETE / LOCAL SOFTWARE TESTS PASS / EXACT OFFICIAL-FILE PYTHON REPLAY OPEN / MODEL INFERENCE BLOCKED / NO MECHANISM OR ACL NOVELTY VERIFIED.
