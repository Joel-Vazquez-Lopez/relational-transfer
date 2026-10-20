# Role and mention-position isolation: Llama native prerequisite failure

20 October 2026. The protocol was frozen at `a608f7989a731d38b3474a3a6e5dafbf31a152a6` before outputs. This study stopped at its declared prerequisite. It is not a failed causal effect: causal interventions were never run.

## Actual result

Llama3.2-3B-Instruct4bit answered **166/288** primary questions correctly, with **21/96** complete three-question source cases. Sham-source questions were **45/72** correct. This fails the fixed native competence gate. The 752 actual forwards include continuations and restart/reference checks. All 631 production checks and 9,300 independent checks passed.

| Name/question set | Active | Passive | Goal-fronted |
|---|---:|---:|---:|
| p01 |46/48|36/48|29/48|
| p02 |19/48|16/48|20/48|

Names and question paraphrases change together between these sets, so their separate contributions are not identified. All inputs are AI authored and independently template parsed. One sending verb, four rotated people, three forms. The unchanged-role questions constrain behavioral interpretation; there is no human-annotated corpus result.

## Exploratory saved-data analysis

After the failure, a query-selective old/new-name contrast was calculated across three separately prompted questions. The contrast is positive for **62/72** role-isolated swaps, and at least one logit for **56/72**. Mean contrast is 5.2694 logits. One hundred synthetic checks confirm its algebraic invariance to specified shared candidate/query biases. No further Llama inference was used.

This is useful evidence that hard answer accuracy and graded relational response can diverge. It is not an independent confirmation, an internal mechanism, proof that each question contains independently sufficient relational information, or a new mathematical statistic. A syntax-aware parser or answer table can produce the same response. The Qwen counterpart freezes a prospective test of this changed question and retains this failure.

## Reproduction and preservation

`NATIVE.jsonl` preserves every full generation and failure; `NATIVE_SUMMARY.json` separates individual answers and complete source cases. `EXPLORATORY_INTERACTIONS.json` retains every contrast. `INDEPENDENT_AUDIT.json` verifies raw vocabulary logits, exact decoding, annotations, tokenization, capture shapes, source-prefix invariance and checksums.

The full raw payloads are preserved in exact-NPZ compressed-member archives described by `RAW_ARCHIVE_MANIFEST.json`. `restore_payloads.py` reconstructs byte-identical original NPZ files into a supplied empty destination. Archival deduplication changes no arrays or scores. No paid API, model download or additional experiment was used.
