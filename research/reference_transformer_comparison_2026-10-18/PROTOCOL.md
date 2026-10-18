# Reference Transformer comparison — prospective, bounded protocol

2026-10-18. Freeze this directory before compiling/evaluating the main N=4 models. One main run, no changing programs, sampler, cases, sites, doses, scores or hypotheses after outcomes. Prior MLP failure remains failed. Small CPU reference comparison; no training, paid APIs, pretrained model downloads or private artifact access.

## Why the original comparison cannot be called a reproduction

The released Congruity testbed uses a modified InterpBench checkout c1841ad7dd892e8b6b116d74d2b675d7519e1d74 and a bank of trained variants/correspondences. Its download script uses an institutional SCP source. The inspected public Hugging Face revision provides only tasks63 and124 of the nine component task IDs needed by the released six chains; zero complete chains are available. No GitHub release asset or model in the checked author/name searches supplied that bank. This is a bounded availability finding, not proof that artifacts exist nowhere.

This experiment is therefore explicitly **reference-only**, not a reproduction of the published trained-model results. It uses public algorithmic routines and the released practical comparison functions with a new, narrow implementation sampler. No full-method superiority claim is licensed.

## Models and prior sources

Two four-symbol permutation tests over sequences of length four, vocabulary {0,1,2,3}:

A sorting: InterpBench case59's stable unique-key ranking/aggregation sort, then check each sorted value equals its index and aggregate all matches.
B counting: InterpBench case108's equal-token counting/duplicate detection, then negate. Fixed length equals vocabulary size, so absence of duplicates is equivalent to permutation membership.

These programs have different known causal graphs but the same finite-domain normal behavior. Label this algorithm distinction at the stated RASP abstraction, not every possible abstraction. Compile with official Tracr commit9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd, mlp_exactness default100, bidirectional softmax attention and ReLU MLPs, no LayerNorm. No learned student models are trained. Compiled reference Transformers are legitimate attention/MLP models, but not evidence about pretrained linguistic processing.

Tracr dependencies are isolated in the sibling feasibility directory; shared environment untouched. The Torch float64 execution adapter is checked against upstream JAX layer states. The N=3 infrastructure preflight is outside the main domain. Adapter API issues were repaired before freeze; N4 outcomes were not inspected. Both native role-source predictions are checked on fixed N3 sanity inputs, not used to tune an N4 result.

## Implementation sampler and limitations

GetImpl is a finite bank of the native circuit plus a parallel, output-irrelevant attention/MLP branch of width8. The two branches never communicate; readout ignores the auxiliary branch. Random auxiliary embeddings/weights vary, and the combined residual coordinates are permuted. The same input is supplied to both branches. Thus arbitrary auxiliary changes preserve the original mechanism under a known native-circuit projection, including the declared interventions transported by the permutation.

This implements the general idea of changing causally irrelevant components. It is not the paper's trained SIIT variant distribution, not a sampler over all implementations, and not an architecture-diverse learned-model replication. Independent branches can be represented using block-separated attention/MLP computation; the code exposes the branches rather than materializing one merged parameter tensor. Residual permutation is an architecture-compatible linear reindexing, not the previous arbitrary nonlinear wrapper.

Development implementation seeds101,103,107,109; held seeds113,127,131,137. Nuisance width/amplitude and all seeds are fixed. The known circuit projection is provided to the intervention assay, so same-algorithm positive intervention transfer is deliberately favorable and not learned correspondence.

## Cases and competence gates

Enumerate all256 inputs for untreated competence. Both native models must classify all256 correctly, and match upstream intermediate states to1e-8 on fixed [0,1,2,3] and [0,0,2,3] sanity cases. No baseline case exclusion or fallback model selection.

Evaluation partition: enumerate lexicographic positives/negatives; shuffle each with NumPy RNG4819. Dev uses first12 positive and first36 negative cases; held uses next12 positive and next36 negative cases. Shuffle each subset with the same RNG stream. Dev/held are disjoint; all24 true permutations are used across them. The distribution is25% positive, not class-balanced. All per-class outcomes must remain available; no binomial significance claim from repeated variants.

Before comparison, each augmented model must preserve normal outputs, native intervention outputs and dose-zero sham to1e-8. A failed numerical/sampler gate invalidates subsequent comparison, with raw failures saved. No rerun after a scientific gate failure.

## Causal intervention and distinct predictions

Common manipulated variable: original non-BOS token-value coordinates in the residual stream. A hard intervention replaces them at all content positions with the one-hot representation of token0; other variables, including computed summaries and BOS, remain fixed. Dose interpolation is linear in residual coordinates. The permutation mapping transports this intervention exactly in each augmented model; this mapping is known rather than learned.

Record every embedding/post-attention/post-MLP state. Source-availability profiles evaluate all stages. Separately, an oracle semantic diagnostic uses a frozen static stage rule: the final operation writing the intermediate summary coordinates (`sorted_position` for A, `duplicate_counts` for B). The rule uses nonzero compiled parameter writes, not intervention outcomes or response peaks.

Prespecified mechanistic predictions on true permutation inputs:

- A: replacing original values after sorted positions exist, before sorted values are aggregated, changes the final result to non-permutation (negative margin).
- B: replacing original values after duplicate counts exist preserves the positive result.

These predictions differ and were verified on N3 preflight. Failure on N4 is reported as a failed semantic gate; no alternate site is chosen. Semantic landmarks differ between programs and exploit ground truth. The diagnostic is NOT a blind correspondence classifier or a fair superiority comparison against representation-only methods.

Dev doses[0,.25,.75,1]; held[0,.125,.375,.625,.875,1]. Save all raw scalar margins. Effect=(clean margin−patched margin)/2, with no clipping. Native categorical reference margins are approximately±1; normal deviations are reported. Keep negative-input results and label them separately.

## Developmental-profile method

Use source seed101 for each program. Average development responses across the48 inputs at doses .25,.75,1, producing stage×dose profiles. Match stages with monotone DTW, mean squared response cost, extra .01 per horizontal/vertical move, fixed diagonal-first tie ordering, both endpoints covered. Distance is RMS unpenalized profile discrepancy along the path.

A coarse source-availability schedule may be shared despite different algorithms. Do not presume DTW will distinguish the programs. This is an explicit competing explanation. If maximum same-program development distance+1e-7 < minimum cross-program distance, freeze the midpoint threshold. Otherwise declare the profile nonseparating; no changed features or tuning. Test all16 ordered source-program/target-program/held-implementation pairs, noting repeated identical task circuits.

Using the frozen mapping and the source's measured held intervention effects, predict individual target held effects at four new interior doses. Require maximum same-program RMSE≤.1. Source held interventions are measured predictor inputs, not predictions from normal data alone. Positive exactness follows from known projection and exact same circuit; no cross-seed learning claim. Negative algorithm predictions may differ or collapse; report all cases.

## Existing-method baseline: exact released practical functions

Source: alansun17904/interp-equiv commit0bdc4f6ff45c9e8319d2450bcd63b082b4e167e9, `left/encoding.py`, Git blob abd3db7367317c51974a265026efb1649a4ec76a. Verify source bytes. Extract `encoder_fit` and `point_test` AST function definitions without modifying their bodies, avoiding unrelated CLI/import dependencies.

`encoder_fit`: released RidgeCV, fit_intercept=True, alpha_per_target=False, train_test_split test_size=.1. `point_test`: released one-sided comparison of within-group vs across-group R²; n20, crit1.96, same default replacement sampling. A logging wrapper records all encoder scores without altering random draws or fitted values. Record library versions.

Representations follow the released extraction concept: average over sequence positions (including BOS), concatenate embedding and every accumulated residual state. Inputs to the baseline are full combined task+nuisance representations. Four fixed bootstrap repetitions for each ordered family pair (A,A),(A,B),(B,A),(B,B), separately on dev and held banks/input sets. Seeds7301+1000×held+100×source_index+10×target_index+repetition. A returned True rejects same-group equivalence; within-program pairs should not reject, cross-program pairs should reject. Keep every decision, regression score and warning. No further tuning or post-result selection.

This reproduces the TWO PRACTICAL RELEASED FUNCTIONS on a restricted new bank, not the paper's full Figure2 setup, not theoretical Algorithm1 with the constrained bidirectional norm of DefinitionG.1, and not a representative test of the whole Congruity framework. Four bootstrap repetitions and four implementations per bank are small and correlated. No formal power or superiority claim is permitted. If the baseline fails calibration, report its scoped failure and do not claim a refutation of Congruity.

The causal method receives native input-coordinate projection while the released baseline uses observed representations; information access differs. Scores are reported side by side, not as an equal-information competition. Neither baseline classification nor the oracle diagnostic predicts all linguistic computations.

## Audit, decision and stopping

Independent NumPy replay of every native counterfactual and the baseline decision formula from recorded R² pairs. Preserve full native/auxiliary weights, all hidden states, raw margins, source inventories, input manifests, paths, individual predictions and logs. Record the exact lack of the published trained-model bank separately from results.

GO/NARROW/NO-GO applies only within this declared setting. NARROW is the default ceiling for a successful assay: reference controls and a comparison under a custom sampler, not trained-model or linguistic generalization. Any baseline or profile failure stays visible. Never claim novel mechanism or publication readiness from a known-reference calibration.

After this one run, stop. No reconstruction of the large trained bank, new tasks, extra training, paid compute or Qwen/Llama inference follows adaptively.
