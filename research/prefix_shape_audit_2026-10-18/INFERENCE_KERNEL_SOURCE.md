# Source-level dispatch explanation

The installed runtime reports MLX 0.32.4. Its [tagged Metal quantized-matmul source](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/quantized.cpp#L1094) chooses split-K partitions using the output matrix tile count, then falls back to ordinary matrix multiplication when only one partition is selected.

For the first Q projection, Qwen has output width 3,584 and Llama 3,072. Applying that source calculation gives:

| Model | Output tiles | Sequence rows | Row tiles | Initial split count |
|---|---:|---:|---:|---:|
| Qwen | 112 | 63–64 | 2 | 2 |
| Qwen | 112 | 65–96 | 3 | 1 |
| Llama | 96 | 63–64 | 2 | 2 |
| Llama | 96 | 65–96 | 3 | 1 |

The predicted switch matches the measured 64-to-65 discontinuity. This is a source-level inference, not a recorded GPU dispatch trace. Additional operations may contribute downstream. Tagged source correspondence does not verify the installed binary's build commit.

The broad phenomenon of shape-dependent reduction order is established [prior work](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).
