# Compass Investigation 04 — Timing is not structure
**Date:** 2026-10-15  
**Branch:** `research/compass-reuse`  
**Status:** Exploratory negative result and mechanism-calibration experiment; **not** a pretrained-LLM discovery.

## Research question
Can pre-question relation caching be experimentally distinguished from question-time reconstruction, and does pre-question computation improve compositionality on its own?

## A. Learned causal-attention controls
Two architecturally scheduled causal-attention networks solved a four-entity, two-ordered-swap task. There are six primitive swaps, 36 operation-pair contexts and four questions per context (144 QA cases). Seeds 19, 29 and 41.

- **Cache:** 59,588 trainable parameters; prefix is processed independently of the question; a question-independent whole-world readout is selected by the later question.
- **Reconstruction:** 53,236 parameters; question-conditioned causal-attention layers process the raw operation tokens after the question.
- Both reached **144/144 on the same finite table on which they were trained** in all three seeds, and both matched donor context outputs after coarse whole-context interchange (**144/144** in each seed).
- A staged cache API computed 36 unique prefix worlds for 144 answers. Reconstructing independently computed 144 query-time worlds. This schedule and reuse were enforced by the APIs, **not discovered in a natural Transformer**.

## B. Genuine withheld-composition negative result
A fixed development split (seed 555) withheld 9/36 *entire ordered operation pairs* (36 questions); 27 contexts/108 questions were used for training. Each individual primitive was still present in training.

At 200 training steps, all six networks fit training **108/108**. Held-out accuracy:

| Seed | Cached | Reconstructed |
| --- | --- | --- |
| 19 | 9/36 | 16/36 |
| 29 | 9/36 | 11/36 |
| 41 | 10/36 | 13/36 |

Both architectures mostly failed to generalize. Pooled descriptive totals across seeds are 28/108 cached and 40/108 reconstructed; these are **repeated evaluations of the same 36 cases**, not independent samples. Architecture and parameter counts differ; these data **cannot show a causal OOD advantage for reconstruction**.

**Interpretation:** forcing the pre-question computation of a state is not sufficient to learn a reusable *compositional operation*. Computation schedule and learned algebraic structure are distinct design axes.

## C. Identical learned weights, different computation schedules
A separate 14,096-parameter causal-attention Transformer was trained on six *individual* swap primitives (24 matrix-row labels) and achieved 24/24 at 250 steps for seeds 19,29,41.

We then supplied matrix multiplication externally to compose the learned primitive matrices, without any training on the 36 two-swap compositions. Across all seeds:

- **144/144** questions correct on two-swap combinations.
- Changing the order of overlapping swaps changed 72/96 answers; disjoint swaps changed 0/24.
- The same trained weights supported **36 pre-question** relation compositions or **144 question-time** compositions for the same 144 answers.

**Major qualification:** matrix composition was **hard-coded by the experimenter**. The model did not independently discover the composition operator, and neither schedule was spontaneously chosen. This is a matched-weights execution-schedule calibration, not an explanation of natural computation in Qwen, Phi, or Gemma.

## Literature boundary
Li, Guo & Andreas, [(How) Do Language Models Track State?](https://proceedings.mlr.press/v267/li25r.html), ICML 2025, already analyzed learned permutation-composition algorithms and their generalization. We do not claim novelty for the swap task or for known circuit reuse.

## Scientific decision
Neither pure behavioral accuracy nor whole-context donor patching identifies when a relational computation is executed. **Compositional operator competence must be separately controlled.** The next meaningful test is a *single common architecture with both pre-question and question-time information routes available*, where a predeclared intervention-based measurement must infer which route the network naturally uses; we must validate that measurement against controlled known-mechanism models before involving pretrained LLMs.

## Reproducibility / integrity
The complete local package `COMPASS_INVESTIGATION_04_TIMING_VS_STRUCTURE_2026-10-15.zip` was generated and SHA-256 verified in this conversation. It contains four Python scripts, PyTorch CPU checkpoints, full raw metric JSONs, training logs, evaluated composition cases, an explanatory report and file hashes. **The binary package is not stored in this GitHub repository**; retrieve it from the Compass conversation artifact. No sealed data were opened, no historical artifacts changed, no substantial external compute used, no merges made, and no pretrained Transformer mechanism claimed.
