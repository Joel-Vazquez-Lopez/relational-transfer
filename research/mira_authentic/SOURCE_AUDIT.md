# First Mira EWT pilot: real source-ID mismatch

**Verified against immutable upstream `en_ewt-ud-dev.conllu` at `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`.**

The earlier 20 candidacies (EWTDEV-000 to EWTDEV-019) retain their historical state:
- **20/20** stored sentence IDs exist in the original corpus.
- **19/20** locally recorded sentence texts equal the original `# text =` line **exactly**.
- **EWTDEV-011: MISMATCH.** The recorded text beginning “Arial photos, loose connections…” belongs to sentence ID ending **`-0005`**, not the recorded **`-0007`**, in document `weblog-blogspot.com_thelameduck_20041119192207_ENG_20041119_192207`. The mismatched `-0007` source sentence discusses different events/actors. This is an incorrectly joined text-to-ID record, NOT a harmless punctuation discrepancy.
- EWTDEV-011 was further missing its claimed `make` predicate and `photos`/`evidence` dependency heads when evaluated against its recorded `-0007` ID; the authentic `-0005` source does have those tokens.
- **Do not overwrite or silently fix the historical EWTDEV-011 row.** It is quarantined, unscored and requires versioned correction, upstream reparse and independent human semantic labeling.
- The old 20 are from only **nine distinct source documents and the weblog genre**. Even 19 exact text matches are **not** 19 semantic agent/patient gold labels; no two-human adjudication occurred.

Audit status: source-document reference checked; redistribution rights unresolved; all previous semantic-role labels stay **UNVERIFIED**.
