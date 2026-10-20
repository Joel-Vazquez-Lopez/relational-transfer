# Cross-model role-path pilot: feasible intervention, inconclusive comparison

Date: 2026-10-17. [Frozen protocol](ROLE_PATH_PILOT_PROTOCOL.json); [script](role_path_pilot.py); raw outputs [Qwen 7B](ROLE_PATH_qwen7b.json) and [Llama 3B](ROLE_PATH_llama3b.json).

## Question and design

Can we trace how a role reversal becomes causally available to the final answer across the layers of two existing local LLMs? Four synthetic sentences used two verbs and four disjoint name pairs. Each sentence was tested with its subject and object swapped, using the same question. At nine preselected block outputs, the reversed sentence's final-prompt-token residual state replaced the base sentence's state. We recorded the first-token logit margin between the two names and repeated each patch using the base sentence's own state as a technical sham. Both models were local, 4-bit MLX; no model download or paid API.

The first causal question is narrow: **when is the reversed prompt's final-token state sufficient to redirect the base prompt's next-token name preference?** It is not a direct measurement of when a model first represents an actor or of a full linguistic computation.

## Competence and technical validity

Both models greedily generated the correct actor name for **8/8 prompts** (four original, four reversed), and all four pairs had equal token length within each model. All self-patch sham margins matched the unpatched base exactly at the reported precision.

The frozen first-token margin had a **readout failure in two Llama cases**. For `Clara` and `Ellen`, Llama generated the correct full name by first emitting `Cl` or `Ell`, while the preselected whole-name candidate tokens had lower logits than the rival name. Thus the reported margins for `help_1` and `praise_2` do not faithfully score the answer choice. These rows remain in the raw output; they are **excluded from interpretive cross-model comparisons**, a post-result measurement limitation, not a replacement of the frozen score. The pilot has only two comparable cases (`praise_1`, `help_2`).

The first Llama run crashed before saving output because its model loop reads a `use_sliding` property absent from the wrapper. A documented wrapper-only fix was then applied. Qwen output used the original script; Llama output used the amended script. Sentences, layer sites, patches and margin code were not changed.

## Results

For Qwen 7B, all four answer margins remained on the base actor's side through block 20 of 28; the reversed donor flipped all four margins at block 24 and reproduced the reversed margin at block 27. For the two comparable cases, Llama 3B showed earlier sensitivity: at block 14, the reversed donor changed `praise_1` margin from **+0.484 to −0.250**, and `help_2` from **+0.125 to −0.840**. The `praise_1` effect was nonmonotonic: the margin returned positive at blocks 17 and 20, then flipped again at block 24. Both models reached the reversed margin at the final patched block, as expected from replacing the final residual state.

The scale-free transfer fraction below is `(base margin − patched margin)/(base margin − reversed margin)`. Zero means base-like and one means reversed-like. It is reported only for the two cases where the first-token margin reflects the model's emitted name.

| Model / case | Block 0 | Block 10 | Block 14 | Block 17 | Block 20 | Block 24 | Block 27 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 7B, Alice/Beth | 0.00 | 0.00 | −0.01 | −0.01 | −0.01 | 0.72 | 1.00 |
| Qwen 7B, Grace/Henry | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.69 | 1.00 |
| Llama 3B, Alice/Beth | −0.01 | 0.01 | 0.44 | 0.15 | 0.28 | 0.73 | 1.00 |
| Llama 3B, Grace/Henry | 0.00 | 0.00 | 0.14 | 0.24 | 0.37 | 0.68 | 1.00 |

The other two Qwen rows also showed late block-24 transfer fractions of **0.68** and **0.68**. The invalid Llama first-token rows are preserved verbatim in the raw JSON.

## Interpretation

**What happened:** We can technically measure a layerwise, within-model causal response to a controlled role reversal on both models. The two interpretable Llama profiles are more sensitive around the middle blocks than the corresponding Qwen profiles under this final-token patch. The sham passes.

**What this does not show:** It does not establish shared or different algorithms. Final-token residual replacement can transfer generic answer state, not an isolated actor computation. The models differ in architecture, size, chat template and tokenization. The Llama readout failure reduces the comparable set to two cases, there are no unrelated-donor or neighboring-token controls, and no held-out intervention prediction. Early or late patch sufficiency cannot be equated with the time at which a linguistic operation occurred. The predeclared gate of at least two cases was met numerically, but the measurement problem makes this a feasibility pilot, not an interpretable cross-model trajectory test.

**Decision:** The basic intervention is feasible. A serious next design must fix the answer readout *before* new results (for example, score the full name sequence under the patched model), add source/content controls, and predeclare a prediction about an unseen linguistic change. Do not infer a conserved process from these four curves.

## Provenance

- `ROLE_PATH_qwen7b.json` SHA-256: `1b2634cc0b752478255ca8d7f650fbfb56049d57bf085190893bc82c9cd1ae88`.
- `ROLE_PATH_llama3b.json` SHA-256: `c15136a44b1d1469bb739b604c4d0078a55de8151ef797fade032e8a7cdb2aeb`.
- Initial script SHA-256: `8a1979660d20af3fc5a42584e62f6b6cced72a93388e5da2e0640588b1347516`; wrapper-fix SHA-256: `28eb8afcffa94903bb1f994f8595e1f12db20b68346da3797d4816cd1f0fef44`.
