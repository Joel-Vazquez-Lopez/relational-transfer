# BX-004 — Contextual Evidence Utility: offline audit and authentic-data feasibility

**Date:** 16 October 2026. **Branch:** `research/brain-x-theory`. **Status:** offline source-locked evaluator implemented and 12 local tests passed; authentic semantic experiment **NOT READY**. No new LLM inference, new training, sealed data, human annotation, or publishable mechanism result.

## Scientific gap (separate from other workstreams)

- **Astra** owns output-field-order versus input-name position in Qwen. BX-004 uses one fixed, single-name direct answer format and does not vary extraction field order.
- **Brain-O** owns learned event-address/participant-path interventions and matched copiers.
- **Compass** owns composition and auxiliary intermediate-state training.
- **Mira** owns authentic source selection, rights and human adjudication.
- **Nova** owns model runner, lossless prompts, raw outputs and gold isolation.
- **Comput** owns competent source-local lexical/format shortcut baselines and independent audits.

**Brain-X's target:** whether additional *human-verified informative context* produces more *semantically correct answers* than legitimate same-document irrelevant context on the exact same predicate/role question. The outcome must be net utility, not just model answer changes or complementary output formatting.

## Predeclared metric

For document d and event i, let C_inf/C_irr be binary semantic correctness for the same fixed direct role question under two source-context conditions. The primary descriptive diagnostic is

`VCEU = (1 / #documents) * Σ_d mean_events_in_d (C_inf - C_irr)`.

Report exact paired **rescues** (wrong under irrelevant context, correct under informative), **harms** (reverse), joint correctness, strict one-name format validity, and subgroup `local_evidence_status` (sufficient/insufficient/uncertain). Bootstrap documents, never correlated prompt rows. No inference on an individual benchmark item may use scorer-only gold. The model's answer parser is **gold-independent** for format validity; semantic correctness is a separate comparison.

The baseline `local` context provides the starting reference. Added informative and irrelevant context must come from **different unaltered sentence spans in the same original human-authored document**, with the exact local question, decoding and system prompt unchanged. This is not a claim of source-comparable length or identical discourse meaning without separate annotation; human reviewers and Comput must attack confounds. Competent source-only lexical parsing, name copying, extra-token priming and verb-specific questions are all competing explanations. A positive result shows *behavioral utility*, **not** an internal factor graph or cached relation.

## Tested implementation

- [Offline stdlib evaluator](bx004_evidence_utility.py), Git blob SHA **7063ada2c9b5b80012e3b249acba0edb9fa382c2** (same SHA as locally executed source).
- [Independent fixture and gate tests](test_bx004.py), **12/12 passed locally**.
- Four separated JSONL inputs: `manifest` (source, rights, human adjudication, stable event/doc IDs, declared unchanged-span provenance), `gold` (scorer-only, human reviewer IDs and adjudicator), `model_inputs` (only `prompt` transmitted), and `responses` (raw one-name outputs + input hash, model family/revision).
- Fail-closed checks: uncleared rights; incomplete independent human gold or context contrast; overlapping source span conditions; rewritten context; missing/wrong prompt hashes; gold fields in model-view JSON; missing model-family diversity; missing responses; too few document clusters; incorrectly formatted model output.
- 12-document mock with two **fictional** model-family names and synthetic fake approval fields returned the expected fixture metric, rescue 6 / harm 0 / net +0.5. **This proves only the scoring code functions**. It is not model evidence, human consent, human review or a legitimate corpus.
- All intake checks validate **declared metadata**, not actual human identity, licensing rights, true unchanged text or prompt-runner isolation. Independent manual source/runner verification is mandatory.

Usage (after appropriate source permission and annotation):

```bash
python -m unittest discover -s research/brain-x -p 'test_bx004.py' -v
python research/brain-x/bx004_evidence_utility.py --audit-candidates /path/to/candidate_manifest.jsonl
python research/brain-x/bx004_evidence_utility.py --manifest manifest.jsonl --gold gold.jsonl --model-inputs model_inputs.jsonl --responses responses.jsonl --out report.json
```

The candidate audit intentionally exits with NOT_READY and status 2, not a fabricated score. No model inference is performed by either command.

## Actual source-level audit: Mira authentic candidate manifest

Read actual [Mira candidate manifest](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/mira-authentic-pilot-2026-10-16/research/mira_authentic/candidate_manifest.jsonl), Git blob SHA **82ff4543057878a51df8f16a849a23fe3dfd43f8**:

| Metadata criterion | Count |
|---|---:|
| Original sentences, all open development | **25** |
| Independent documents | **25** |
| Genre distribution | 5 each: answers, email, newsgroup, reviews, weblog |
| `human_verified=true` semantic labels | **0** |
| Declared internal-model-use source-rights clearance | **0** |
| Human-adjudicated informative/irrelevant context contrasts | **0** |
| Eligible to run the BX-004 test now | **0** |

The current Mira pilot is *candidate selection*, not a gold benchmark; **these zeroes are expected and not a failure of Mira**. Its verb-root/nsubj/obj selection is grammatical, not semantic evidence, and no same-document independent contextual clue has been authenticated. Underlying-text licensing is explicitly unresolved. No copyrighted EWT source text was copied by Brain-X.

## Decision and next gate

**Do not run real pretrained LLM evaluations from these 25 unannotated candidate rows.** Mira should first decide whether any can be expanded to original source context with independently reviewed actor/patient applicability and eligible source rights. If not, a different rights-cleared human-authored context corpus may be needed. This is a feasibility issue, not a request to manufacture artificial sentences and call them natural data.

When human adjudication, rights and provenance are complete, Nova may freeze an approved two-DISTINCT-family model runner (Qwen 1.5B and Qwen 7B alone are one family), Comput may predeclare source-copying and lexical rival baselines, and Brain-X's evaluator can score the original raw outputs. Confirmation data must be separate untouched documents and predeclared before selection. No human local compute or paid API request today.

## Limits and originality

Contrast-of-context risk differences, rescue/harm and document-cluster bootstrap are standard statistical ideas. Implementing them is an infrastructure/measurement contribution, **not an established new mathematical theorem or neural mechanism**. This separates availability, sensitivity and utility but does not identify their internal representation. The important consequence is that **a seemingly promising authentic pilot currently lacks the evidence needed to support the specific proposed H1 diagnostic**, so early negative/positive LLM results would otherwise be scientifically ungrounded.
