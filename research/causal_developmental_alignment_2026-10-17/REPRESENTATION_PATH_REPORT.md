# What the role-swap paths show

Date: 2026-10-17. Exploratory continuation of the [four-case causal pilot](ROLE_PATH_PILOT_REPORT.md). The same sentence pairs, prompts, models, and baseline answers were reused. No new model or paid service was used.

## What was recorded

For each of four actor/object swaps, I saved both models' hidden vectors at every stage: the embedding and all 28 block outputs. I tracked four aligned sites in each prompt: subject, verb, object, and the final prompt token. Qwen 7B has 3,584 hidden coordinates; Llama 3B has 3,072. All eight prompts per model had correct greedy actor answers in the earlier screen. Exact source token spans were one token each in both models, and each swapped pair kept the same total token length.

The figure has two views of the *same saved vectors*. The left panels show mean relative distance between original and swapped vectors, `||h_original − h_swapped|| / mean(||h_original||, ||h_swapped||)`, across four sentence pairs. The right panels trace the final-token **difference vector** through a fixed two-dimensional PCA basis fitted separately within each model. All 29 stages are connected in order. The two PCA axes explain 46% of Qwen and 52% of Llama final-token contrast variance; the omitted dimensions remain in the raw arrays.

![Role-swap representation trajectories](ROLE_PATH_TRAJECTORIES.png)

## Observations

1. At the embedding, the subject and object positions are already far apart across swaps because they contain different name tokens. The verb and final prompt token start with zero separation because those tokens are unchanged. This is an input-identity fact, not evidence that the model has bound semantic roles.
2. The verb and final-token states acquire nonzero role-swap differences as layers accumulate. Mean final-token relative separation at stages 7, 14, 20, 24, and 28 is **0.015, 0.028, 0.079, 0.348, 0.632** for Qwen 7B and **0.044, 0.141, 0.195, 0.268, 0.251** for Llama 3B. Under this normalization, Qwen's final-token contrast grows sharply late, while Llama's emerges earlier and changes less at the end. This is descriptive; the scales are not guaranteed equivalent between models.
3. The PCA paths *fan out by sentence pair* rather than collapsing onto a single common line. At the final stage, the six pairwise cosines between four final-token role-swap difference vectors average **0.03** in Qwen and **0.07** in Llama. The lines visibly encode substantial example-specific content. The first two PCs capture only about half the variance, so their turns and endpoint distances are an illustration of the saved paths, not a complete geometry.
4. The earlier final-token causal patching pilot showed late transfer in Qwen and some mid-layer sensitivity in the two Llama cases with valid first-token scoring. That is directionally compatible with the geometry above, but does **not** establish that the same hidden dimensions implement role binding. Two Llama cases had a tokenization-related answer-readout failure in that causal pilot; the hidden-vector capture itself is unaffected.

## What follows scientifically

We **can** follow a multidimensional contrast through layers and see when it spreads from changed name positions to unchanged verb and answer positions. We **cannot** read a model's algorithm directly from the line. Lexical differences, residual-stream scaling, prompt format, layer normalization, and PCA can all shape it. A rising separation says the two prompts differ internally; it does not by itself say which difference causes the answer.

The strongest hypothesis suggested by this exploratory view is a *timing difference*: role-swap information reaches the final prompt site differently in Qwen and Llama. The strongest simple alternative is model-specific scaling and answer-readout geometry. A future confirmatory study would need new contrasts, a valid full-answer score, selective interventions at the participant and verb sites, and a frozen prediction about an unseen transformation. No claim of a conserved computational path or new mechanism is supported by this figure.

## Artifacts and reproducibility

- [Capture code](representation_paths.py) and [plot code](plot_representation_paths.py).
- [Qwen metrics](PATH_METRICS_qwen7b.json), [Llama metrics](PATH_METRICS_llama3b.json), and [figure summary](ROLE_PATH_FIGURE_DATA.json).
- Full selected-site vector arrays: [Qwen `.npz`](PATH_STATES_qwen7b.npz) and [Llama `.npz`](PATH_STATES_llama3b.npz), each shaped `[4 cases, 2 variants, 29 stages, 4 sites, hidden width]` in float16. These are selected token sites, not every prompt token.
- [Standalone figure](ROLE_PATH_TRAJECTORIES.png).
- SHA-256: Qwen states `0776be0f0f1b2d5e0de3f85e100c81b64065f5f2e2ea5ac56f8ac087871232de`; Llama states `85de38e3da7a491bdf16b4be597aa7a21fdc70951a53f08ae9faf55b4005d3b8`; figure `e1890fd95a41c2229a5bb98fb7a377877b0e164c3cdafd358f33f9e4e61b00bb`.
