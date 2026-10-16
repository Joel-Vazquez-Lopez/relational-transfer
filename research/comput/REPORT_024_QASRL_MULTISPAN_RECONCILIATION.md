# COMPUT-024 — Exact QA-SRL Gold 1,154 vs 1,152 multi-span denominator reconciliation

**16 Oct 2026 | independently source-replayed, DEV ONLY; no model inference**

## Identical upstream source lineage

Read all four original QA-SRL Gold **development** CSVs directly from the immutable GitHub Git blob endpoints, as previously pinned by Armaggedon, Nova and Brain-X at upstream commit `f7c64ae9b6fe48ff3910c3e59850a12ec278bf83`:

| Source | Gold blob | Sentence blob |
|---|---|---|
| WikiNews dev | `2171f970fc6bc34b6847a0274f1a62e388b8a4df` | `52fa60f9827d59f90662ade828f358a8cff5d893` |
| Wikipedia dev | `8ccc9450490a4ce3fac7be86ea53f99493e1f75d` | `9845a4e67cdc3588921cbc7c264b611c6bea197f` |

Source data parsed with a **separately implemented quoted-CSV state-machine** (double quotes, doubled escapes, CRLF and embedded quoted delimiters); source gold joined by `qasrl_id` to source `tokens` and raw `answer_range` intervals (start-inclusive, end-exclusive). Multiple answer/range pairs split by `~!~`; exact source-text range checked after whitespace folding and lowercase. No original sentence/answer text copied to report or prompt. No official TEST file accessed.

## Receiver-executed reconciliation

| Measure | WikiNews DEV | Wikipedia DEV | Combined |
|---|---:|---:|---:|
| Reference sentence rows | 917 | 917 | 1,834 |
| Raw human-gold QA rows | 3,714 | 3,474 | 7,188 |
| **Raw** multi-span rows (Brain-X denominator) | **474** | **680** | **1,154** |
| Answer-text/range alignment mismatches | 3 | 0 | 3 |
| Multi-span rows **among mismatches** | **2** | **0** | **2** |
| **Aligned/eligible** multi-span rows (Nova denominator) | **472** | **680** | **1,152** |
| Distinct valid (sentence,predicate,question) keys after source-alignment filter | **3,710** | **3,470** | **7,180** |
| Duplicated valid question keys (extra annotation rows) | 1 | 4 | 5 |

**Reconciliation solved:** Brain-X reports the **1,154 raw multi-span gold questions**. Nova reports **1,152 post-alignment multi-span questions**, because the **two other affected malformed-text rows are multi-span**. These are both correct *with explicit denominator labels*; neither numerical result refutes the other.

The three exact mismatching WikiNews **zero-indexed original raw gold row positions** are 260 (multi), 1489 (multi), 2666 (single). Keep these quarantined without silently repairing Unicode/answer strings, and avoid quoting original source text. Under the original source alignment and valid-duplicate accounting: 7,188 raw rows − 3 quarantined = 7,185 valid raw annotation rows; 7,185 − 5 valid duplicate prompt keys = **7,180 valid unique predicate/question prompts**. Distinct source sentences, distinct predicates, distinct documents and prompts are different denominators; don't call 7,180 independent documents.

## Next gate and caveats

- **Nova's scorer and adapter** must explicitly report `RAW`, `VALID`, `DEDUPLICATED` and `ATTEMPTED` counts, plus exclusions for the three exact source/text mismatches. The union of answers for a duplicated original question is a possible representation but its scoring interpretation needs provenance-aware sensitivity; never count one matched answer as completion of a multi-span set unless the source question permits it.
- **Brain-X** raw integrity audit is correct; label as raw. **Mira** single-span subset naturally quarantines only the one mismatching single-span record, not all three.
- The above **independent full original source replay was in connected GitHub JavaScript**, not the still-unexecuted original-source Python code in Nova's v0.3 ZIP. No new model run, paid inference, test set, rights approval, human adjudication or internal mechanism evidence was involved.

**Status: SOURCE-DENOMINATOR CONFLICT RESOLVED / model evaluation and Python cold replay still open.**
