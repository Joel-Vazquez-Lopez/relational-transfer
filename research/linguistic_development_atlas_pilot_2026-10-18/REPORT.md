# Linguistic Development Atlas pilot — technical NO-GO

**18 October 2026 · CLOSED · 2 model forward passes · 0 held-out passes.**

## Decision

**NO-GO for this execution because of an implementation failure. The linguistic hypotheses remain untested.** Codex's runner caused the failure. No atlas, cross-model comparison, semantic mechanism or novelty result is available.

The user explicitly accepted **AI-only annotation** with “use ai only annotation.” The response and review-document hash are preserved in `ANNOTATION_DECISION.json`. No independent human linguistic validation is claimed.

## What ran

| Model | Forward calls | Saved scored development questions | Complete 16-question families | Held-out calls | Status |
|---|---:|---:|---:|---:|---|
| Qwen 7B | 2 | 1, recovered offline | 0 | 0 | Aborted during result serialization |
| Llama 3B | 0 | 0 | 0 | 0 | Not started: frozen shared runner has the known defect |

The calls were durably recorded before execution in `qwen7b/FORWARD_JOURNAL.json`:

1. Unwrapped validation on `a01_within_0_agent`. This completed, but its logits were held only in process memory and **not persisted** before the later failure. A wrapped/unwrapped numerical comparison is unavailable.
2. Wrapped main capture on the same prompt. Selected-site tensors and full-vocabulary logits were saved before row serialization failed.

The wrapped repeat did not run. The other 31 Qwen development prompts, all held-out prompts and all Llama prompts remain unexecuted. Paid calls, model downloads and training: **0**. Total inference: **2/132 authorized forward calls**. No further inference or adaptive experiment followed the failure.

## Exact failure and responsibility

Frozen `run.py`, line 167, assembles a result with:

```python
result = dict(**row, **score(logits, row, aliases, tok.decode),
              max_reconstruction_relative_error=recon)
```

Both dictionaries contain `expected`, causing:

```text
TypeError: dict() got multiple values for keyword argument 'expected'
```

The scorer returned successfully; dictionary assembly failed. The preflight tested scoring and analysis separately but missed their integration with real-shaped input rows. That was insufficient verification and was **my implementation error**. `RUN_qwen7b.log` preserves the traceback.

The frozen no-post-output-code-change/no-restart rule was applied. Scripts, stimuli, sites, scoring and predictions were **not changed** after outputs. The run was not resumed. Llama was not run simply to reproduce the same known defect. `OFFLINE_REPAIR_CHECK.json` reproduces the Python error and checks a dictionary-literal repair without inference. `SERIALIZATION_REPAIR_PROPOSAL.diff` is **not applied** to the frozen runner and does not authorize a new run.

## Recoverable output and its limits

> Sentence: Ada recommended Boris and Clara to Neil.
> Question: Did Ada recommend someone?

The unchanged scorer applied offline to saved logits gives:

- Expected **Yes**; full-vocabulary argmax **Yes**.
- Gold-class logsumexp margin over No: **23.906251 nats**.
- This single answer passes the answer criterion. No family competence screen completed; it is not a passing family or a general competence result.
- Residual archive: **29 stages × 14 selected sites × 3,584 dimensions**.
- Attention and MLP archives: each **28 blocks × 14 sites × 3,584 dimensions**.
- Maximum residual reconstruction discrepancy for this one capture: **0.00051468**, below the proposed 0.002 threshold. Wrapper agreement, repeat stability and causal-prefix invariance are untested. The overall technical gate has **not passed**.

`qwen7b/PARTIAL_OUTPUT.json` preserves the input, exact token/site ledger, every original scorer field and array schema. Full selected-site tensors and logits remain local, with hashes in `RESULT_MANIFEST.json`. The missing unwrapped validation output is not imputed.

Only one sentence/question cell exists. The four-cell interaction is unmeasured, not zero. No developmental atlas or inferential plot was produced. The frozen `analyze.py` was not executed on incomplete comparisons.

## Startup failure and operational deviation

The initial sandboxed startup could not import MLX because Metal GPU access was unavailable. This occurred before `mlx_lm.load`, model initialization or creation of a forward journal. Its start marker is preserved byte-for-byte as `qwen7b/STARTUP_BLOCKED_SANDBOX.json`.

A model-free GPU check outside the sandbox confirmed Metal availability. `ZERO_INFERENCE_STARTUP_RECOVERY.json` records the administrative recovery: preserving the failed marker under a different filename allowed the first actual model run with GPU access. This is an **operational deviation from the engineering no-restart marker**, made before any scientific output. Frozen source and scientific settings stayed unchanged. After model outputs existed and serialization failed, no restart or repair was applied.

## Interpretation and prior-work boundary

There is no valid comparison among grouping, lexical permutation, generic schedules, answer preparation or syntax-aware copying. This execution neither supports nor falsifies the Atlas hypothesis. Earlier observations, failed gates and counterexamples remain unchanged.

The scoped prior-work discussion, including [Dai et al. (2024)](https://aclanthology.org/2024.emnlp-main.967/), remains in the [design report](../linguistic_development_atlas_design_2026-10-18/DESIGN.md). No new surviving mechanism or novelty claim is available here.

## Reproducibility and closure

The prospective implementation was frozen at commit `7d90d6563c00a330ceb1255e8cea1436e8dccbf5`; `LOCK.json` identifies input, weight, tokenizer, runtime and source hashes. Frozen source hashes were verified unchanged during offline recovery. Preparation files are historical pre-inference records, not completed native validation.

To reproduce only the offline recovery with existing local files:

```sh
research/stronger_model_development_2026-10-15/.venv/bin/python research/linguistic_development_atlas_pilot_2026-10-18/recover_failure.py
```

This imports the frozen scorer and local tokenizer, not MLX, and performs zero model calls. `EXECUTION_STATUS.json` marks this run closed. The runner must not be silently repaired or rerun as though this failure had not occurred. Large tensor/logit files remain local; remote hashes and ledgers do not imply that every array is hosted on GitHub. No follow-on experiment was started.
