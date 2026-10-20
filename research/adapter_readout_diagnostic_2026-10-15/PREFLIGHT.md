# Technical smoke preflight

**Status:** completed on 2026-10-15. These four new sentences are format/load checks only and are excluded from the proposed full diagnostic.

The smoke used two new events, each rendered in active and passive voice. It evaluated untouched Qwen2.5-0.5B-Instruct and the fresh1 / seed 830211 ordinary, ordered-role, and participant-only adapters. The archived direct QA prompt asked “Who performed the action?” and scored the first generated token. The explicit extraction prompt requested `Actor: <name>; Patient: <name>`. [SMOKE.json](SMOKE.json) stores exact prompts, raw tokens, parsed fields, checkpoints, hashes, and timing.

| Arm | Load | Four direct QA prompts | Four extraction prompts |
|---|---:|---:|---:|
| Untouched | 1.07 s | 0.74 s | 1.37 s |
| Ordinary | 0.35 s | 0.33 s | 1.34 s |
| Ordered-role | 0.33 s | 0.38 s | 1.56 s |
| Participant-only | 0.34 s | 0.33 s | 1.43 s |

All base, source, graph, and protocol hashes matched their frozen references. Each trained checkpoint supplied exactly the 96 expected LoRA tensors; the two auxiliary arms supplied exactly four expected head tensors, and ordinary supplied none. All checkpoints loaded locally with `weights_only=True` and CPU `float32` inference. No training or network access occurred.

**Format blocker for a full run:** the fixed strict extraction parser accepted only 7/16 outputs. Nine responses contained a final period after the patient name and were marked unparsed, despite otherwise matching the requested fields. The full protocol must freeze a parser rule before any full inference. The original raw output and exact-format validity must remain separate from semantic correctness. One untouched passive extraction also named the participants in the wrong roles, but this tiny technical sample has no evidential weight and was not used to revise the prompt.

At batch size four, the observed inference time was approximately 0.33–0.74 seconds for direct QA and 1.34–1.56 seconds for extraction per four-sentence batch. The proposed 2,016 sentence-model comparisons would require 504 batches for each interface, suggesting roughly 15–25 minutes of CPU inference plus startup, checkpoint loading, output writing, and any longer generation. This is a planning estimate, not a benchmark of the full panel. The host is an Apple M2 with 16 GB RAM; this runtime exposes CPU only.
