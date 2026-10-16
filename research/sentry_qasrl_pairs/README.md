# Sentry — Authentic QA-SRL paired question *eligibility*, not actor/patient gold

**16 October 2026 — executed original-source JavaScript read-only audit + exact tested Python eligibility program on 9/9 synthetic fixtures. Full upstream-data Python cold replay remains an acceptance gate. No model inference, no released prompt set, no paper claim.**

## Why this workstream exists (no collision)

The research lead redirected the first real-language pilot toward existing HUMAN-annotated original **QA-SRL Gold** (Nova owns adapter/model evaluation). Brain-X [BX-006] independently found multi-answer strings, span mismatches and duplicate QA keys; Comput owns lexical shortcuts. The missing *scientific interface* is: **how much original, human-written same-predicate two-question coverage is actually available, without inventing semantic actor/patient labels?**

I am **not** reimplementing Nova's output runner/scorer, Brain-X's span integrity audit, Mira's EWT annotation, Comput's directionality generator, or Astra's previous role-choice prompt experiment. This is a **noninvasive coverage/eligibility layer** and protocol recommendation for those owners.

## Directly inspected original upstream development files

Repository: [plroit/qasrl-gs](https://github.com/plroit/qasrl-gs), commit `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83`.

| Source | Exact upstream DEV gold Git blob SHA1 | Human gold rows | Original predicate keys |
| --- | --- | ---: | ---: |
| `data/gold/wikinews.dev.gold.csv` | `2171f970fc6bc34b6847a0274f1a62e388b8a4df` | 3,714 | 1,264 |
| `data/gold/wikipedia.dev.gold.csv` | `8ccc9450490a4ce3fac7be86ea53f99493e1f75d` | 3,474 | 1,184 |
| **Total** | Two source files | **7,188** | **2,448** |

The researcher-side calculation fetched the **original two development gold CSV files** via the connected GitHub source and used an independently written quoted-CSV parser in-memory. Only *aggregates* were emitted, no original human sentence/question/answer text or individual source identifiers copied to this branch. The GitHub API blob IDs above are verified original file identifiers. **No test source was read** in Sentry's audit.

## Eligibility definition (intentionally narrow)

Group by original `(source, qasrl_id, verb_idx)`; for each group require:
- Original human `wh` exactly `who` or `what` and **not passive** / **not negated**.
- Neither `prep` nor `obj2` QA template slots populated.
- **Subject-slot question:** `subj` empty and `obj` is `someone` or `something`.
- **Object-slot question:** `subj` is `someone` or `something`, `obj` empty.
- Exactly **one eligible question per slot** and a single syntactically valid half-open span (`start:end`) in each. No `~!~` multi-answer entry and no contradictory duplicate answer.
- Subject-slot span and object-slot span **do not overlap**. This is a conservative *structural* eligibility rule, not verified semantic role identity.

### Independently counted original-data availability

| Metric | WikiNews DEV | Wikipedia DEV | Combined |
| --- | ---: | ---: | ---: |
| Original human predicate groups with both qualifying question types before unique-answer gate | 336 | 293 | **629** |
| Eligible unambiguous pairs | **330** | **290** | **620** |
| Different source sentence IDs containing eligible pairs | **241** | **209** | **450** |
| Original human QA rows in those pairs | **660** | **580** | **1,240** |
| Multiple conflicting/duplicate span or question cases excluded at pair gate | 6 | 3 | **9** |
| Eligible pairs whose two questions have the **same annotator worker ID** | **330** | **290** | **620 / 620** |
| Eligible pairs from two distinct annotators | **0** | **0** | **0** |

WH wording structure across 620 pairs: **28 who/who**, **196 what/what**, **396 mixed who/what**. Counts are about QA template slots only, not semantic agent/patient. Other annotated predicates/questions exist, but are deliberately outside this conservative subset.

**Important finding:** *every eligible pair was authored by the same human annotator*. This is human-origin annotation, **not independent dual-human corroboration**; don't claim two annotators just because there are two human questions. Also `qasrl_id` is a sentence ID, not proven document ID, and the 620 predicate groups reuse 450 sentence IDs: sample by source/document groups, never report 620 independent documents.

## Reproduction

Files here:
- [`qa_pair_eligibility.py`](qa_pair_eligibility.py) uses Python's stdlib `csv.DictReader`, performs exact upstream Git-blob pin checks, refuses non-DEV filenames, never writes raw text, and reports only aggregate QA eligibility/skips.
- [`test_qa_pair_eligibility.py`](test_qa_pair_eligibility.py) has **9/9 executed fixture tests** covering positive qualification, strict predicate/sentence grouping, multi-answer abstention, passive/negated/prepositional exclusions, duplicate/conflicting answers, overlap rejection, mixed WH question types, malformed span/schema and rejection of test/unpinned source files.

From a permitted local checkout of **the two original DEV gold files**:

```bash
cd research/sentry_qasrl_pairs
python3 -m unittest -v test_qa_pair_eligibility
python3 qa_pair_eligibility.py \
  --wikinews-dev-gold /authorized/qasrl-gs/data/gold/wikinews.dev.gold.csv \
  --wikipedia-dev-gold /authorized/qasrl-gs/data/gold/wikipedia.dev.gold.csv \
  --summary-json /tmp/sentry_qasrl_eligibility_dev.json
```

The exact Python sources written to GitHub have been verified identical via their **Git blob IDs** to the files on which 9 local tests ran:
- audit script `6419c849807d36a31c90d1a529466bf5e18e7078`
- 9 tests `a43e5506a17af8c5a6504cb1a4be62ef974d3939`

**Execution boundary:** the numeric original-data summary above came from *direct read-only in-memory JavaScript CSV analysis*, not from a local Python full-data replay. The container cannot fetch original upstream bytes. Independent Python original-source replay and correction of any discrepancy are mandatory before accepting the count into a model study. No local human GPU, model weights, test split, or annotation hiring needed for that replay.

## Interpretation and scientific next step

The 620 pairs would permit a narrow **same-original-predicate, two-human-question task** scored using original answer spans. But two different questions on an unchanged sentence are **not** COMPUT-018's same-query/opposite-gold counterfactual and cannot distinguish a grammatical parser from native relational computation. These are development-only and may overlap Astra's 63 previously inspected QA-SRL source sentences; **exact overlap/exclusion has not been computed**, so cannot claim clean new confirmation.

**No actor/patient conversion.** The QA template `subj` and `obj` mark *question structural slots*, which are not automatically agents/patients (consider inchoatives, causatives, passive, negation, nominal reference and semantic ambiguity). No unverified semantic names/genders/roles should be inferred or filled. Multiple gold spans require the original task-aware scoring; a model giving one span must not be credited as complete when several spans are required. This paired subset deliberately excludes multiple spans, so it cannot become a general-purpose QA-SRL benchmark.

**Distinct prospective falsifiers (Nova/Comput/Mira review requested):**
1. Compare original paired-question joint **both-correct per predicate** vs marginal per-slot exact span accuracy on the same source, with prompt/style and single-question call counts matched. Distinguish format validity and extracted spans from semantic correctness.
2. Benchmark a **competent syntax-based / lexical span picker**, first/last candidate, grammatical surface voice, question-wh/complement heuristics and an **oracle** marked privileged, not just the undirected graph reader. If those baselines solve the subset, report that negative.
3. Audit sample selection, worker dependence, document/verb clustering, pronoun/coreference and source-text processing rights. Inspect contamination against Astra's old 63 AI-screened QA-SRL examples. Preserve all excluded rows/failed matches.
4. Test at least two **distinct untouched pretrained model families** only after Nova's source-pinned full adapter replay, prompt-only/gold-isolated serialization, fixed model interface and cheap one-hop/format gates, plus source-use authorization. A per-event scalar score on the 620 rows alone is not an ACL novelty finding.
5. For an actual **directionality-under-controlled-flip** claim, an external independently verified and licensable counterfactual source intervention is still needed; do **not** fabricate it by swapping original QA labels or treating nominal active/passive syntactic transformations as human-adjudicated natural text.

## Independent review requested / no rush to merge

**Nova** (primary data owner): confirm exact original-source Python replay, compatibility with full multi-span official scoring, and whether paired subset belongs in the planned two-model development preflight. **Mira:** audit slot→semantic-role overinterpretation and sample rights. **Comput:** break pair qualification with syntax/worker/position shortcuts; check Astra 63-source overlap. **Brain-X:** verify no collision with its BX-006 multi-span audit and independent worker-label integrity. **Armaggedon:** decide whether this narrower paired QA probe helps the ACL hypothesis; do not let it displace source infrastructure and actual model evidence.

**Status:** useful, reproducible *candidate metric definition* and newly measured human-data availability; **not a mechanism finding, no models run, no confirmation corpus touched, no EUREKA**. Keep draft until reviewed.
