# Qwen three-role native development screen

Frozen before new model outputs, 20 October 2026. User authorized the proposed next step. This stage collects the108 development prompts from the reviewed three-role design. **No t03/t04 evaluation prompt, Llama run, intervention, prompt search or calibration is part of this collection.** All old gates and decisions remain unchanged.

## Inputs and output rule

Two AI-authored name triples × six role assignments × active/passive/goal-fronted × three questions =108 prompts. One fixed chat template, the existing candidate-free system instruction and assistant prefix `Answer:`. Source cases are shuffled once with seed20261020, with the three questions adjacent within each source case. Exact token IDs, offsets, complete candidate token paths and annotations are frozen.

Unconstrained greedy decoding at most12 new-token decisions. Stop at `<|im_end|>` or `<|endoftext|>`. Primary correctness requires an observed stop token within the limit and the decoded name, after whitespace stripping only, equal to gold. Punctuation, explanatory text, another name and truncation are retained as failures. No first-token proxy, forced candidate decoding or manual recoding.

Separately compute the full-name-prefix log probability for each of the three names by teacher forcing its leading-space token path. This common continuation convention must agree with tokenizing the actual prompt plus that text. These probabilities exclude EOS and alternative tokenizations/whitespace paths. The gold-minus-best-other margin is descriptive and does not alter native correctness or eligibility. This secondary readout is not a calibrated model posterior or native answer.

## Competence units and decision

Report all108 individual answers; all36 source-level three-question units; all162 six-answer donor/recipient transposition pairs (nine unordered assignment edges × nine ordered form pairs × two lexical families); construction/family/role breakdowns; wrong, off-format, truncated and margin-disagreeing outputs. Pair counts reuse answers and are not independent replicates.

A pair is eligible only if all six native answers for its two source contexts are correct. Full-family perfection is not required by this new bounded claim; this does not relax any earlier study's gate. The minimal **baseline-ready** decision requires, separately in each development family: at least one eligible cross-form pair for each of the three invariant roles, all three source forms represented in eligible cross-form pairs, and every technical gate passed. This is a feasibility screen for drafting a causal protocol, not a positive causal result or a general-competence criterion. All selected and excluded units and unfiltered scores remain visible. No significance test.

## Technical controls

- Local4-bit Qwen2.5-7B-Instruct, unchanged weights/runtime source hashes. No downloads, training or paid calls.
- Batch1, physical length128 for every baseline, continuation and validation forward. Dummy tokens only after the real prompt/continuation; readout at the real end.
- Boundary wrappers delegate unchanged native blocks and masks. First development input: native unwrapped versus wrapped full-vocabulary logits ≤0.05 nat; exact repeated-source hashes and residual-trace max absolute difference ≤1e-6, logits ≤0.05.
- For every scientific continuation, original-prompt-end logits match its own baseline within0.05 nat. Across questions sharing a source, all28 full source-prefix residual hashes must be exactly identical.
- Baseline traces save embeddings plus28 outgoing residuals at the last subtokens of all three mentions (surface order), source period and original assistant end. These are observational captures; attention/MLP writes and identity **patches** are not measured here. A wrapper repeat is not a passed causal-patch control.
- Each worker executes at most32 forwards. Its first call reproduces an exposed earlier native control within0.05 nat, then it executes at most31 new calls. Restarting is preplanned resource management, not retrying failed scientific calls. Each actual pass is journaled before dispatch and its full output vector saved before scoring.

## Bounds and stops

`LOCK.json` derives an exact conservative forward cap from the frozen candidate-prefix trie and12-step decoding bound, plus three new-input validation calls and the maximum required worker-equivalence calls. No beyond-cap retry. Maximum45 minutes collection time;90s per dispatch including worker initialization; disk reserve4GiB; swap used <2GiB; MLX memory guideline6GiB and cache256MiB; active-MLX allocation must remain below6GiB. Workers close after at most32 calls. These are guardrails, not hard OS memory isolation.

Stop on any technical/resource failure, preserving attempts, completed outputs and traceback. Do not retry, substitute prompts or loosen tolerances after outputs. A successful baseline-ready result permits preparation of the separately frozen development intervention test under the user's conditional authorization. Its site-selection rule, controls, budget and forward instrumentation must be concrete and validated first. Evaluation families remain outcome-unseen until a candidate and prospective predictions qualify. A failed baseline gate stops causal continuation.
