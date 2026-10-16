# Mira Wikipedia: prospective event selection × answer interface pre-spec

**DRAFT only — no confirmed semantic labels, source passage or model run.**

**Question:** Given two distinct text events described with the same or related predicate, does a specified contextual cue improve the correct participant assignment more than an equally long irrelevant cue? Is a joint two-field output's advantage primarily suppression of repeated names rather than better relational selection?

## External-validity gates
1. Exact `oldid` revision readback, `pageid`/title/timestamp verification and UTF-8 source SHA-256 pass. Historic revision pointer alone is insufficient.
2. Identify *contiguous* original passages and record precise section/offset and paragraph SHA after filtering third-party quotes/imported text; retain contributor-history and CC BY-SA license links and indication of modifications. No raw source text in this GitHub PR.
3. Two real independent humans label **event identity, predicate scope, semantic role applicability, full referring expressions, negation/modality/coreference**, and whether two mentions denote distinct events. A human adjudicator preserves unresolved cases; syntax tags / model labels cannot replace this.
4. Pre-register **two distinct pretrained families** and one-hop format/competence threshold **before seeing test outputs**. Qwen 1.5B and Qwen 7B are not distinct families. Wiki is common pretraining data, so even excellent accuracy cannot automatically establish fresh OOD generalization.
5. First pilot is **open development**. Independent confirmation must use wholly unexamined page/source-family groups with frozen prompts and approved licensing.

## Planned contrasts (Astra owns field-order inference)
- **Source setting:** single event; two same-predicate mentions with differing argument assignments; two mentions with informative in-document cue; two mentions with an irrelevant-length-matched addition. Added/rewritten context is a controlled derivative, *not* unchanged authentic language.
- **Question:** role-specific original wording vs fully event-specified wording.
- **Response:** independent direct agent and undergoer, joint `Agent;Undergoer`, and joint `Undergoer;Agent` with raw outputs.
- **Baseline:** first/last mentioned names, grammatical subject/object plus passive, verb-key nearest match, question-time competent source parser, and name-complement policy.
- **Metrics:** strict format validity distinct from semantic applicability/role correctness; full-event both correct; reversal, same-name collapse, wrong-event selection, abstention. Group uncertainty by article/document rather than number of generated responses. Small pilots yield only descriptive statistics.

## Falsification criteria
- If informative cues do not improve event-specific answers over irrelevant cues, stronger context-binding proposal fails on this panel.
- If question-time lexical copier or output complement rule explains gains, no native abstract relation mechanism.
- If two human annotators cannot establish two distinct semantically applicable event referents, the **dataset suitability** gate fails (not a model failure).
- If historic text/rights cannot be confirmed, quarantine examples; do not infer on them.

**First concrete candidate:** `INSPECTED_WIKI006.md` pinpoints two discovery references in Marie Curie's historical revision, but both are high-frequency encyclopedic facts, so this is only a pipeline check. All seven page candidates are at discovery/intake status, not a validated evaluation corpus.
