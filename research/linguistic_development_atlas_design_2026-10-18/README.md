# Design delivery: Linguistic Development Atlas

**Status: DRAFT FOR JOINT REVIEW. New model forward passes: 0.**

## Preferred next question

Does the response to **the same name permutation** change when it crosses an argument boundary?

- Within one role: `Ada recommended Boris and Clara to Neil.`
- Across roles: `Ada recommended Boris to Clara and Neil.`
- Swap Boris/Clara in both. Preserve word inventory and name positions.
- Track the difference between those two permutation responses across tokens and layers, with constant-answer and mention-order questions.

The literal marker/position copier in the audit script solves every draft question. This does **not** distinguish an abstract semantic mechanism from a strong syntax-aware copier. It can distinguish restricted additive or answer-label-only explanations and locate contextual interactions. Nonlinearity alone can produce an interaction; a nonzero number is not a discovery.

## Files

| Artifact | Purpose |
|---|---|
| [DESIGN.md](DESIGN.md) | Preferred design, alternatives, competing predictions, limits, resource bound and execution gates |
| [SAVED_DATA_AUDIT.json](SAVED_DATA_AUDIT.json) | Saved results, exact local source hashes and state archive headers |
| [OLD_DESIGN_CERTIFICATE.json](OLD_DESIGN_CERTIFICATE.json) | Exact old role×voice / mention-order / answer alias |
| [OLD_PAIR_COVERAGE.csv](OLD_PAIR_COVERAGE.csv) | Every saved same-answer/different-Agent pair, including wrong answers |
| [DRAFT_STIMULI.jsonl](DRAFT_STIMULI.jsonl) | 64 annotated question rows; **human linguistic review pending** |
| [DRAFT_TOKENIZATION.jsonl](DRAFT_TOKENIZATION.jsonl) | Both models' actual chat strings, token IDs and retained sites; no activations |
| [TOKENIZER_AUDIT.json](TOKENIZER_AUDIT.json) | Matching positions, cue availability, Yes/No aliases and storage estimates |
| [PREDICTION_WITNESSES.json](PREDICTION_WITNESSES.json) | Explicit restricted rival predictions and the 64/64 surface-string toy result |
| [design_audit.py](design_audit.py) | Reproducible saved-data and tokenizer-only calculations; no model runtime import |
| [MANIFEST.json](MANIFEST.json) | File hashes, sources, proposed count and explicit draft status |

## What changed in our knowledge today

The old two-person design has an exact algebraic alias: first identity equals role×voice. Additional names cannot repair it. There are saved same-answer/different-Agent pairs, but they do not simultaneously hold voice, position and question fixed. These are design facts/coverage results, not new model-mechanism findings.

The proposed four-person contrast passes local tokenization checks in both models, uses counterbalanced Yes/No questions, and has a strict zero-interaction causal-prefix control. Proposed maximum: **128 main + 4 validation forwards**, only after review and annotation agreement. Previous competence/invariance/transport failures remain failures.

Existing broad curve/theory exploration remains with [Issue #35](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/35). Codex contributes this design and identifiability audit on the existing research branch. No causal interventions or substantial new inference were run.
