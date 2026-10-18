# Source-access feasibility — results

## Interpretation correction: the competence scorer was inadequate

**This is a scoring-design failure, not a demonstrated linguistic failure.** The frozen endpoint passed 6/8 baselines, but both rejected outputs were the first token `N` of the possible multi-token answer `Nina`. Preflight accepted a single-token leading-space alias while the no-leading-space name requires two tokens. No continuation was observed, so neither full-answer correctness nor full-answer failure is established on those two cases. The machine stop label below is preserved; [POSTMORTEM.md](POSTMORTEM.md) explains the investigator error. **Zero substantive source-access cuts, zero evaluation calls and zero Llama calls were executed.** All 13 frozen input/code/protocol/test artifacts remained byte-identical. No adaptive rescue followed.

Closed bounded execution: **11/400 attempted forwards**. No paid API, training or model download.

## Decisions

| Model | Outcome | Baseline answers | Calls |
|---|---|---:|---:|
| qwen7b | STOP_DEVELOPMENT_COMPETENCE | 6/8 | 11 |
| llama3b | NOT_RUN | 0/0 | 0 |

## What this tests

The intervention closes later access from every question/assistant suffix token to all event-dependent context tokens, while preserving earlier computations. It measures continued-access dependence under that mask. It does not identify when semantic information first exists, establish a reusable role representation, or distinguish a strong syntax-aware copier from semantic composition.

## Complete baseline outcomes

| Model | Family | Correct / total | Entire family correct? |
|---|---|---:|---|
| qwen7b | d1 | 2/4 | False |
| qwen7b | d2 | 4/4 | True |

### qwen7b: validation

```json
{
  "wrapper_max_logit_error": 0.0,
  "native_dense_max_logit_error": 0.0,
  "repeat_max_logit_error": 0.0,
  "repeat_max_state_error": 0.0
}
```

## Failures and controls

ANSWER_LEDGER.csv includes every scored call, including validations and controls. FAILURES.csv includes every incorrect next-token answer, including intentionally source-isolated calls; those are distinct from baseline competence failures. SUMMARY.json retains all decision fields and full control findings. RAW.json includes top-20 outputs and alias logits; full-vocabulary logits and native captures are local.

## Limitations and prior work

Two constructed lexical families per phase are not a population sample. Questions differ in wording/length; passive evaluation changes construction and names together. Answer identities are matched on active development but differ between demands in passive evaluation. Quantized local models and one runtime limit generalization. Attention removal changes normalization and may cause off-distribution damage. Matched irrelevant cuts match edge count but not absolute position or attention mass. A preserved answer does not exclude later rereading; a failed answer does not prove absence of an internal representation. No clean cross-model replication is claimed after a failed gate.

Attention knockout and staged retrieval have close precedents in [Geva et al.](https://aclanthology.org/2023.emnlp-main.751/), [Mixing Mechanisms, appendix F](https://arxiv.org/html/2510.06182v2), and [Lookbacks, appendix K](https://arxiv.org/html/2505.14685v3). [Dai et al.](https://aclanthology.org/2024.emnlp-main.967/) already demonstrates binding-related activation edits. This feasibility run makes no novelty claim.

## Preservation and closure

LOCK.json freezes source, stimuli, installed native model code and local weight/tokenizer/config hashes. FREEZE_RECEIPT.json identifies the pre-inference commit. BUDGET.json journals attempts before dispatch. Tensor/logit files remain local with checksums in RESULT_MANIFEST.json; uploading text artifacts does not imply uploading every tensor. Earlier experiments and their gates were not modified. No adaptive follow-up follows this report.
