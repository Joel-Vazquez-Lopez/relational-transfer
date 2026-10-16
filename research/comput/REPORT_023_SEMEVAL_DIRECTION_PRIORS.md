# COMPUT-023 — SemEval-2010 Task 8: directional-prior identifiability audit

**16 October 2026 | COMPLETED original public-mirror TRAIN text audit | no LLM inference**

## Provenance and executed access

I read the **entire 1,397,759-character/byte UTF-8** `TRAIN_FILE.TXT` directly via GitHub's original **Git blob endpoint** for `nb20593/Semeval-2010-task-8-Relational-extraction`, branch `main`, blob **`0797b23e4907e12e26739a79a454dfae9bb71c13`**. Parsed all 8,000 numbered source records using a separately implemented in-memory JavaScript parser that reads only their two initial lines (marked sentence and gold relation; comments ignored). **8,000/8,000** parsed; 8,000 unique IDs from 1 to 8000; 0 invalid/unordered entity tags; all rows have exactly one e1/e2 mention and a valid 19-class label. Neither external official ZIP provenance nor its complete byte-level match to this third-party mirror was established. Do not call this an independently SHA-verified original-official-archive download.

The contemporaneous 2010 Task 8 description distributed with a public mirror explicitly names **CC BY 3.0 Unported** for the accompanying dataset; verify source rights, attribution obligations, and full ZIP contents before distributing raw sentences or using them in a new model service. Source description: `nb20593/Semeval-2010-task-8-Relational-extraction/Sem eval 2010 task 8 Description.txt`.

## Main independent finding: label-conditioned direction prior

The original released **TRAIN** portion has **1,410 `Other` + 6,590 direction-labeled** examples. The direction-only majority class across all non-Other examples is `(e1,e2)` **3,588/6,590 = 54.446%** (vs `(e2,e1)` 3,002/6,590).

However if a predictor is **given the correct human relation type for each example** but ignores that sentence's actual direction and always selects the type's majority orientation, it obtains:

**4,819/6,590 = 73.126% correct directions.**

Five-fold crossfit learning each type's orientation prior from four folds and evaluating on the fifth, **grouping all identical UNTAGGED underlying sentence strings in the same fold**, yields **4,775/6,590 = 72.458%**. This grouping is **not verified source-document grouping**. The oracle TYPE is still handed to the predictor at test time, making these **privileged conditional directional diagnostics, NOT a deployable text-only direction baseline**. The result cannot be used to claim a zero-shot model reaches 73%, that 73% is a ceiling on actual models, or that real LLMs performed any computation.

| Relation type | e1→e2 | e2→e1 | Majority direction % | Maximum orientation-balanced pool |
| --- | ---: | ---: | ---: | ---: |
| Cause-Effect | 344 | 659 | 65.70 | 688 |
| Instrument-Agency | 97 | 407 | 80.75 | 194 |
| Product-Producer | 323 | 394 | 54.95 | 646 |
| Content-Container | 374 | 166 | 69.26 | 332 |
| Entity-Origin | 568 | 148 | 79.33 | 296 |
| Entity-Destination | **844** | **1** | **99.88** | **2** |
| Component-Whole | 470 | 471 | 50.05 | 940 |
| Member-Collection | 78 | 612 | 88.70 | 156 |
| Message-Topic | 490 | 144 | 77.29 | 288 |
| **TOTAL non-Other** | **3,588** | **3,002** | **73.13 (type oracle)** | **3,542** |

The reverse `Entity-Destination(e2,e1)` has **ONE** example among 8,000 training rows. A balanced within-type directional evaluation requiring both directions would retain only **2** `Entity-Destination` training rows. Thus merely downsampling a global split cannot produce a robust nine-type balanced directionality benchmark; this class needs new independently suitable human-labeled examples or explicit exclusion from that particular balanced analysis.

## Repeated-source leakage caveat

Ignoring entity markup yields **25 extra instances** of the same normalized underlying sentence (24 repeated-sentence groups); **9 repeated-sentence groups contain different gold labels**. Inspection demonstrates some mark different entity pairs within an otherwise identical sentence, e.g. a girl–stick pair labelled Other versus girl–violin Instrument-Agency. They are **not necessarily annotation contradictions**. With entity markup retained, there are **0 identical marked-sentence inputs**. A sentence-only random split can put differently marked pairs from one original source on both sides and leak lexical context; fold grouping is a minimum control, not a substitute for source-document lineage.

## Consequences for real pretrained-model pilot

1. **Preserve the original 19-way task** (nine relation types × two ordered orientations + Other), reporting type accuracy, exact 19-way accuracy, full relation×direction confusion, orientation conditional on correct predicted type, and unrecognized/format-invalid predictions. `Other` has no defined semantic direction and must not be scored as one.
2. **Avoid inflated directionality:** a type-conditioned majority-direction policy is the critical null. Match/stratify actual human direction labels within each type for a secondary adversarial development slice, with per-class sample-size warnings; do not bury the 844:1 imbalance.
3. **Competitor priority:** simple surface entity order, marked-pair lexical identity, preposition/verb patterns, input-only syntax parser, memorized encyclopedic relation, class/type prior, and original sentence duplicate groups. Freeze before seeing Qwen/Phi outputs; never feed source comment, gold class or source ID in prompts.
4. **Novelty:** SemEval-2010 is a public, old, extensively used benchmark. Any model evaluation is exploratory and may suffer pretraining memorization; this corpus measures **typed nominal semantic relationships**, NOT canonical actor–patient event roles or internal Transformer mechanisms.
5. **Ownership:** Nova owns two-distinct-pretrained-family inference and QA-SRL P0, Mira owns semantic applicability/rights, Comput owns this additional *negative-control* task and can audit any future SemEval pilot. Do not duplicate Nova's runner. No newly executed LLM, paid compute, sealed tests or forced merge.

## Reproducibility

New independent **stdlib-only** `research/comput/comput023_semeval_audit.py` and `test_comput023_semeval_audit.py`. Eight tests **8/8 executed PASS** on entirely synthetic fixture input. Original 8,000-row counts were executed in read-only GitHub-connected JS using the actual mirror blob; the separate Python CLI has **NOT** been cold-run against this exact source because the public source bytes have not been mounted into the local Python runtime. Do not confuse fixture tests with original-byte Python replay. The full script accepts permitted local `TRAIN_FILE.TXT` and emits only aggregate JSON, not any source sentences.

**Scientific signal: meaningful limitation of an otherwise legitimate human directional-gold benchmark; not pretrained-LLM evidence or a new mechanism.**
