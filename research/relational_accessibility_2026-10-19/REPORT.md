# Saved-state accessibility: completed exploratory diagnostic

19 October2026. Frozen local protocol/code before all probe fits, with source hashes in LOCK.json.3,708 small deterministic probe fits, including all12 training-label permutations per Atlas boundary/site/fold. **Zero Transformer forwards, no hyperparameter or layer search; zero convergence warnings.** Probabilities/losses for1,468 records replayed independently;96 graded baseline payload hashes verified. Plots visually inspected.

## Result

**No robust portable linguistic decoder is established.** The saved Atlas period states often support positive cross-family log-score gain for complete unqueried Theme/Goal membership, while late answer-state transfer is poor and overconfident. These are two lexical families with four sentences each. Final-boundary period accuracy is2/4 in both Qwen transfer directions and4/4 versus1/4 in the two Llama directions. Final-boundary answer-state gain is negative in all four checkpoint/fold combinations. A single favorable fold is not a replication or a significant population finding.

|Checkpoint|Site|Train→test|Final log-score gain (nats)|Correct/4|
|---|---|---|---:|---:|
|Qwen|period|a01→a02|.0253|2|
|Qwen|period|a02→a01|.2295|2|
|Qwen|answer|a01→a02|−1.1190|1|
|Qwen|answer|a02→a01|−1.3671|2|
|Llama|period|a01→a02|.2153|4|
|Llama|period|a02→a01|−.0359|1|
|Llama|answer|a01→a02|−.5763|2|
|Llama|answer|a02→a01|−.5323|2|

Gain is improvement over the training prior (.5,.25,.25), not estimated internal uncertainty or exact mutual information. The prior has50% accuracy and1.03972nats cross-entropy on these balanced-structure test cells. The target is the complete membership class, preserving within-group permutation; it is not the native Yes/No answer.

Both questions and correct/native Agent answers are fixed within a lexical family, but differ between families. Thus cross-family transfer also crosses a question/answer/context shift. The late failure cannot be interpreted as disappearance of the relationship, semantic erasure or evidence that late layers merely prepare output. Coordinate changes, confidence, decoder misspecification and the tiny training set all remain viable explanations. The surface syntax parser solves the synthetic labels perfectly, so a positive probe would not uniquely imply semantic computation either.

The separate six-family binary panel gives weak, unstable fixed-linear R decoding and little cross-voice portability; all29 stages, sites and transfers remain in RESULTS.json. Its role label is deterministically recoverable from correct answer plus question, so it cannot separate a relational representation from prepared answers. It does not rescue the old trajectory mechanism claim.

## What is useful

A sentence-period representation and a later answer representation can differ greatly in accessibility to a fixed trained decoder even when the underlying prompt still contains the same relationship. Their geometric growth or shrinkage is not a functional interpretation. This motivates explicit causal message tests, not a broader claim that our probes discovered processing stages.

All shared-decoder before/after-attention/MLP probabilities and separately fitted boundary decoders are retained, with negative changes and errors. Component accounting is not causal use. Twelve label-permutation controls are descriptive sensitivity comparisons; their ranks are not population p values. No layer, successful fold or family was selected as a new positive endpoint. Failure of these specific probes does not establish absence of relational information.

The next investigation is a separately frozen Qwen period K/V message interchange. Its rationale is the pre-existing F-versus-closed route ambiguity, not a selected favorable probe outcome. No significance-driven adaptive experiment follows this diagnostic.

## Reproduce

Existing Python environment: `research/stronger_model_development_2026-10-15/.venv/bin/python`. `run.py` reads existing arrays only; never loads an LLM. It refuses to overwrite RESULTS.json. `analyze.py` independently replays losses and exports complete curves. LOCK.json identifies source hashes. All probe probabilities, per-layer errors, controls and scripts are saved; source tensors remain in their original archives. Old failed gates are unchanged. The scientific target and score/scaling choices were fixed before fitting; the report is post-result interpretation.
