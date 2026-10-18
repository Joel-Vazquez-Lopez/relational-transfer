# User-authorized scoring verification and rerun

Authorization: **can you verify it and run it again?** This explicitly supersedes the previous attempt's no-rerun closure for a separate, prospective amended attempt. The original attempt remains unchanged at commit f35e167716a4b59be71fb5cf7a4185462f8a5f90.

## What changes

The old next-token whole-name alias endpoint was invalid for natural multi-token names. Replace it with the probability of every token in the full natural name sequence, summing distinct leading-space/no-leading-space paths without length normalization or EOS probability. This follows the full-name teacher-forcing principle already used in the verified factorial gate. It measures name-prefix probability, not terminated-answer probability.

Separately observe greedy generation until EOS or a fixed four-token maximum. Correctness requires observed EOS, an exact stripped natural name equal to gold, and a positive full-name margin. A partial first token cannot count as either a correct complete name or an established full-name failure without a completion. If generation truncates, report it as truncated/format failure under this endpoint, not proof of linguistic incapacity.

Each required continuation uses batch one and the same fixed total length 128. Every generated or teacher-forced token is subject to the original all-suffix source cut. Exact shared continuation prefixes are evaluated once per condition and reused. Audit that changing only future continuation tokens preserves the original prompt-end logits within .05 nat. Every actual model forward, including candidates, generation and validation, is journaled before dispatch and its full logits are saved immediately.

## Bound

There are still at most **195 primary conditions/validation records per model**, not new linguistic tests. Full-sequence and terminated-generation checks require extra forwards. The revised cap is **2,000 attempted forwards total, 1,000 per model**; worst-case 192 scored records times five distinct forward prefixes plus three validation passes is 963 per model. Expected use is lower, and all original early stop gates remain. The old 11 calls remain recorded separately, never deleted from the history. No paid API, model download, training or additional model is allowed.

## What stays fixed

All 32 model-specific input rows retain exactly the original prompt text, token IDs, names, annotations, splits, positions, instruction, questions and source/irrelevant spans. Development outcomes are already exposed. Original evaluation model outcomes remain unobserved at freezing. The native mask wrapper is byte-identical. Cuts, matched controls, restored windows, numerical tolerances, family competence requirement, matched-effect formula, .5-nat prediction criterion, selection rule and Qwen-to-Llama gate remain fixed. No successful-family substitution or new site search.

## Verification before rerun

Inference-free adversarial fixtures check a correct multi-token completion, a wrong partial completion, missing EOS, duplicate path removal and exact teacher-forced probability summation. A fixture uses the previous saved first-token logits with **explicitly hypothetical** continuations: it is a software test, not recovery of the missing answer. The actual production controller is rehearsed with synthetic engines through both models, both phases, all condition types, continuation journaling and persistence, matched controls, prediction selection and reporting gates. A negative competence fixture must stop before any source cut; repeated attempts must be rejected. Static mask tests close suffix relay paths. All fake outputs are labeled and temporary.

The amended code, input manifest, test results and this document are committed before any new target-model output. Real-model wrapper/dense/repeat and continuation-prefix checks remain necessary; synthetic tests cannot establish their values.
