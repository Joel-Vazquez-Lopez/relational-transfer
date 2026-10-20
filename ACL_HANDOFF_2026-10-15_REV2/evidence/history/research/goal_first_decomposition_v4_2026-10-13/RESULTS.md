# Corrected clause-decomposition pilot (13 October 2026)

## Status

This is an exploratory, controlled pilot on generated two-hop questions. It does **not** establish a new general reasoning method, natural-language transfer, or an LLM training repair. The earlier pilot and v2 results had an incorrect padding mask and are invalid for scientific inference; the partial v3 run was stopped. This v4 run uses the actual `<pad>` token ID (2) and a fresh data bank. The [frozen protocol](PROTOCOL.json), [audit script](analyze.py), and [machine-readable analysis](ANALYSIS.json) preserve the design and all five paired runs.

## Question and comparison

The task is to identify two operation directions, such as actor-to-patient followed by patient-to-actor, from a nested question. The model is **not** asked to extract names, identify verbs, use the facts, or calculate the final answer. Training includes only same-direction pairs, FF and RR; testing includes the unseen mixed pairs, FR and RF. Both arms receive the same 6,000 training questions, two direction labels per question, architecture, paired initializations, optimizer, update count, and sequence length. The control receives a repeated full-question view. The treatment also receives a view of each relevant clause with the other clause replaced by a constant marker. Clause boundaries for these extra views are supplied by the generator.

The primary test asks whether the treatment predicts **both** directions correctly from the intact question. The separate diagnostic supplies the same normalized clause views during inference to both models. There are 400 items in each test set and five paired seeds; the seeds share items and should not be described as independent datasets.

## Results

| Seed | Full question, control | Full question, treatment | Supplied clause views, control | Supplied clause views, treatment |
|---:|---:|---:|---:|---:|
| 11 | 0/400 | 0/400 | 207/400 | 400/400 |
| 23 | 0/400 | 0/400 | 200/400 | 400/400 |
| 37 | 0/400 | 17/400 | 0/400 | 400/400 |
| 41 | 0/400 | 0/400 | 200/400 | 400/400 |
| 53 | 0/400 | 0/400 | 352/400 | 400/400 |

Both models scored 400/400 on the in-domain *full-question* test in every seed. All ten trained models used 470 updates and 240,000 training labels each, including the auxiliary views. The audit verified protocol and checkpoint hashes, matching item order and labels within each pair, and recomputed every accuracy from the saved predictions.

The frozen **full-question improvement gate failed**: no paired seed gained the required 40/400 mixed items. The narrower **supplied-normalization gate passed**: the treatment scored 400/400 on mixed items in all five starts, with a mean advantage of 208.2/400 over the control under the same diagnostic views.

## Interpretation and limits

This shows that clause-level supervision can teach the operation choices **when the relevant clause is supplied separately**. It does not show that the model learns to find that clause in the intact question: in this design, it generally did not. That gap is the useful result. It separates a local operation learner from a system that constructs and uses its own decomposition.

The grammar is narrow, the operation labels are binary, and questions can reuse surface forms even though no complete sentence/question pair is duplicated across banks. The observed gain also relies on a supplied representation at inference. These are exploratory results after development on earlier bug-affected runs; they are neither a confirmatory finding nor evidence of novelty against prior methods. The next decisive test is to learn a clause selector from the intact question, evaluate the selector and final operation choice separately on fresh mixed combinations, and then vary syntax so the selector cannot pass merely by memorizing this template.
