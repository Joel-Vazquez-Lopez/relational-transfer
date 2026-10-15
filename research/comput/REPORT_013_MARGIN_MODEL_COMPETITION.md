# COMPUT-013 — Complete archived margin-model competition

**Status:** Executed retrospective study on original Qwen2.5-1.5B 384-response file, SHA256 `653a33a0a56f34a84c5c39649dda7bdb100990462daea0390cfcb2fcb8d7ea27`. No inference/training/sealed outcomes. The original source file is not committed.

First-minus-last margin is derived by flipping the sign of archived actor-minus-patient candidate logit scores in passive voice. **This score is reference-role-indexed**, so the regression is a diagnostic and NOT a fully gold-free end-to-end answering model. The answer equals sign of this margin in 384/384 archive records. Predictors use only question wording, voice, and/or verb, with reference roles used in score construction but not as model features.

Four fixed model families: constant; question×voice cell; question×voice cell + additive verb; question×voice×verb cell. Fixed 1e-6 ridge stabilization. Grouped leave-one-name-pair-out and leave-one-verb-out. Q² = 1 – CV MSE / full-panel margin variance (common reference).

| Held-out unit | Model | RMSE | Q² | Sign agreement |
|---|---|---:|---:|---:|
| Pair | Constant | 4.067 | -0.002 | 336/384 |
| Pair | Question×voice | 2.165 | 0.716 | 368/384 |
| Pair | Question×voice + verb | 2.104 | 0.732 | 368/384 |
| Pair | Question×voice×verb | 1.467 | 0.870 | 378/384 |
| Verb | Constant | 4.090 | -0.013 | 336/384 |
| Verb | Question×voice | 2.572 | 0.599 | 368/384 |
| Verb | Question×voice + verb | 2.682 | 0.564 | 368/384 |
| Verb | Question×voice×verb | 4.093 | -0.014 | 336/384 |

**Falsification:** Rich verb-specific cells dramatically improve fitting known predicates but catastrophically fail held-out-verb prediction; the simpler question×voice policy retains predictive sign agreement on all held-out verb folds. This is a clear generalization boundary, not evidence that the model internally uses that policy. Four known verbs and eight known pairs were examined repeatedly; even grouped CV is exploratory, not fresh confirmation.

**Reproduce:** `python3 research/comput/comput013_margin_models.py /path/to/RESULT.json` (requires numpy and pandas). Full locally executed package with six passing source-locked tests and machine-readable results: `COMPUT_013_MARGIN_MODEL_COMPETITION.zip` shared in the conversation. No novel pretrained algorithm identified. Next: freeze the basic positional rule and test new independently reviewed counterfactual sentences/verbs with proper exposure checks and consent for inference.
