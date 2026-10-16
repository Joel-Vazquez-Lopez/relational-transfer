# BX-005 — Self-falsify BX-004: changing the question can fake an evidence benefit

**Date:** 16 October 2026. **Status:** adversarial evaluator audit **COMPLETED**; corrected compatibility validator committed. **No new neural training, pretrained model inference, human annotation or authentic semantic claim.** Original BX-004 files and negative readiness result remain preserved as historical evidence. This is an evaluation-integrity result, not a novel computation mechanism.

## Falsification of v1 (before patch)

Our existing [BX-004 offline scorer](bx004_evidence_utility.py) checked a source/meta manifest, prompt hashes, three distinct prompt strings, source-rights *declarations*, human-annotation *declarations*, model-family declarations, and per-document rescues and harms. It did **not** require the actual question or system-output instruction to be **identical** across local, informative and irrelevant context arms, and it did not enforce canonical construction from the registered original source spans.

**Executed adversarial test:** Start from the same 12-document synthetic smoke and two **fictional** “model family” strings. Modify the informative arm's question to **“Who is the PATIENT instead of the ACTOR?”** while leaving the other two arms' questions unchanged; recompute every prompt SHA and model-output hash. All v1 integrity checks still pass (`errors=[]`), and the scorer still reports a **+0.50 / 50-percentage-point** mock informative-minus-irrelevant accuracy effect (6 rescued, 0 harmed). The outcome is a prearranged *fixture artifact*, **not evidence from any model**. The point is that the evaluator **permits invalid comparisons to be described as evidence gain**. This is a concrete false-positive *capability* in the original evaluation contract.

This is a failure of experimental identification, not a discovered Transformer behavior.

## Correction: BX-005 locked prompt/context contract

New [`bx005_context_contract.py`](bx005_context_contract.py) wraps BX-004's scoring layer with a stronger fail-closed validation layer:

1. **Question invariant**: same exact predicate/role question across all three contexts within an event. An informative-versus-irrelevant treatment may never change the question.
2. **Output instruction invariant**: same exact one-name output contract and instruction across conditions.
3. **Local source invariant**: identical canonical local text spans (full ordered IDs and SHA-256 commitments) across all contexts, with no extra context in the local arm.
4. **Additional-text restriction**: informative arm contains only the expected source-pinned informative span IDs, irrelevant arm only the expected source-pinned irrelevant IDs. No duplicates and no cross-condition span ID reuse.
5. **Canonical render**: the displayed prompt must equal a deterministic serialization of fixed instruction, source-local passage, legitimately added source context, and fixed question. Recompute SHA-256 on both rendered prompt and whole displayed context; require match to recorded hashes.
6. **Model-view isolation helper**: `model_visible_request(row)` serializes **only** `{"prompt": ...}`, not source document IDs, privileged gold, annotator fields or generator edges. Nova's actual runner must independently confirm use of this view; a helper in our package cannot prove the deployed path.
7. **Separate manual verification**: a successful automated validator says `CONTRACT_PASS_PENDING_MANUAL_SOURCE_REVIEW`; test scoring says `SYNTHETIC_SMOKE_ONLY_UNLESS_MANUALLY_VERIFIED`. Hashes and self-declared “human” IDs cannot establish actual human review, third-party rights, or whether two original natural-language spans truly provide informative vs irrelevant evidence.

**Tested:** BX-005's twelve adversarial tests pass. The original BX-004 twelve tests also continue passing: **24/24 across both suites** in a clean local directory. The key regression checks the old false positive explicitly: **BX-004 accepts it and BX-005 rejects it** with `QUESTION_CHANGED_ACROSS_CONDITIONS`. Other tests cover changed output instruction, altered prompt body, wrong source span, modified source text even with recomputed per-span hash, shared source span across contexts, invalid rights declaration, missing second model family, changed document ID, and model-only prompt serialization. No raw human-authored source text or genuine model outputs were used.

**Source provenance:** exact file `research/brain-x/bx005_context_contract.py`, confirmed tested-local Git blob SHA-1 `37b1773dd9722778abc79d29b5a943965f4c328d`. GitHub path and source-checksum documented in draft [Brain-X PR #21](https://github.com/Joel-Vazquez-Lopez/relational-transfer/pull/21). Test path `research/brain-x/test_bx005.py`; both work with existing sibling `bx004_evidence_utility.py` and `test_bx004.py`. Tested Python stdlib only.

Run at repo root:

```bash
python -m unittest discover -s research/brain-x -p 'test_bx00[45].py' -v
```

On shell, run the patterns separately if your discovery glob does not support bracket syntax; both test files were executed together via `python -m unittest -v test_bx004 test_bx005` from the `research/brain-x` directory.

## Authentic data intake: latest Mira status

Updated review of Mira's source-pinned 2026-10-16 authentic branch:

- 25 first-pass sentences from 25 documents and five genres, **zero** completed human semantic-role adjudications and **no** model-use source-rights clearance.
- **24 supplemental challenge groups**, comprising **36 distinct predicate-review slots**, 24 further source documents, six each in `repeated_lemma_nearby`, `passive`, `two_overt_event_candidates`, `pronoun_event_scope`. The source manifest and annotation templates have been prepared, but source text is not copied here.
- These are development **candidate slots**. Neither source grammatical functions nor fake metadata status constitutes human-verified semantic roles, genuine independent contextual evidence, or approved redistribution/inference.
- The separate scientific coordinator's `coord_annotation_integrity/audit_forms.py` now audits **blank annotation-form integrity**. That is complementary, **not** equivalent to this BX-005 causal-comparison/prompt-immutability audit. No duplicate coordinator source audit run was requested.

The inclusion of repeated-lemma and pronoun-event categories improves feasibility, but we have **no confirmed original documents with an independently adjudicated informative-vs-irrelevant contextual pair**. Do not ask Nova to run models yet.

## Scientific consequences and next gate

This self-falsification is essential because differences in question semantics, source fragments, and output format **can fully confound apparent context utility**. BX-005 closes one preventable design loophole. It does **not** certify that an approved corpus contains causal/contextual information, that wording effects are gone, or that an LLM computes latent relational variables.

**Next:** Mira independently verifies a small subset's real contextual evidence, source rights and human role labels; Comput tests a competent label-free lexical/source-reading rival; Nova validates actual prompt-only serialization and a two-distinct-family *preflight*, only when data/permission gates permit. Brain-X can then measure within-document **rescue vs harm** rather than mere answer changes. End-to-end runner-model integration and external validation remain **OPEN**.

**No main merge, no sealed or restricted data opened, no paid model inference, no human machine run requested.**
