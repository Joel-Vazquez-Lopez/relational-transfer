# Linguistic Development Atlas — frozen pilot implementation

**Current status: implementation ready; annotation gate pending. Model forwards: 0.**

Joel authorized the bounded pilot with “go dor it” on 18 October 2026. The preceding [design](../linguistic_development_atlas_design_2026-10-18/DESIGN.md), Section 2, also required a human annotation pass. General authorization to run does not certify that a human reviewed the 16 gold annotations. A concise annotation question has been presented to Joel, including the option to explicitly waive that prerequisite and accept an AI-only annotation limitation. No confirmation is inferred from silence.

## Work completed

- All 16 sentence variants and 64 question/gold rows inspected by Codex; concerns documented in `ANNOTATION_AUDIT.json`. This is one AI review, not two independent human/agent reviews.
- Native installed Qwen and Llama block sources inspected. Both use sequential residual attention and MLP writes; the capture wrapper delegates to the original operations.
- `run.py`, `instrument.py`, `analyze.py`, exact protocol, annotation review and audit frozen by SHA-256 in `LOCK.json`, before any model output.
- Existing weights/config/tokenizer/chat-template files and installed source/runtime revisions hashed. Local tokenization fixes padding to token 87 (literal x) for both models. Model weights were read for checksums, not loaded for inference.
- Known additive/interactive/rotated toy cases checked; undefined cosine remains undefined.
- Scoring checks cover deduplicated aliases, ties, gold argmax/margin agreement, and an unexpected full-vocabulary argmax that defeats a forced-choice-only score.
- The pending-annotation gate was deliberately tested: it rejects execution **before model load and before STARTED.json creation**. This expected guard rejection is not a model failure or a consumed forward pass.

`PREFLIGHT_CHECKS.json` records these checks. Native numerical validation, model competence and causal-prefix checks have **not** been evaluated yet.

## Frozen bounds

Four sentence/question cells per family × four demands × four families = 64 main prompts/model. Two validation calls/model: one unwrapped first development prompt, one wrapped repeat. **At most 132 total forward calls.** Every attempted native forward is durably journaled before it is called. A started model cannot be rerun/resumed automatically.

At least one of the two development families must pass all 16 answer checks, along with wrapper/repeat and causal-prefix checks, before either held-out name family is evaluated. Component reconstruction failure suppresses component claims. No generated rollouts, patches, downloads, paid APIs, extra prompts or extra models.

The interaction is `I=(across_swap−across_original)−(within_swap−within_original)`. Save full residual/attention/MLP captures and full-vocabulary logits locally. Analysis preserves every exposed family/question/token/layer and separate family competence; it fits no alignment or predictor. The marker/position copier and general nonlinear representation explanations remain competitors.

## Pending annotation decision

Read [ANNOTATION_REVIEW.md](ANNOTATION_REVIEW.md). `ANNOTATION_DECISION.json` currently records pending status. Only a direct human response confirming the readings or explicitly waiving human review may replace that status. Keep the actual response and review-document hash in the decision record. A flagged ambiguity stops this frozen proposal before inference; no automatic rewording or item replacement.

After that gate is met, the existing authorization covers these commands from the workspace root:

```sh
research/stronger_model_development_2026-10-15/.venv/bin/python research/linguistic_development_atlas_pilot_2026-10-18/run.py --model qwen7b
research/stronger_model_development_2026-10-15/.venv/bin/python research/linguistic_development_atlas_pilot_2026-10-18/run.py --model llama3b
research/stronger_model_development_2026-10-15/.venv/bin/python research/linguistic_development_atlas_pilot_2026-10-18/analyze.py
```

Run models sequentially. Do not call `--preflight` again over the existing lock. Do not alter frozen code after outputs. Each model's `DEVELOPMENT_DECISION.json` is saved before held-out inference; all failures and unrun panels must be reported.

Full tensor and logit files will remain local with hashes, with scripts, raw answer ledgers, controls, descriptive maps and reports preserved on the existing research branch. There is no empirical pilot outcome at this preparation checkpoint.
