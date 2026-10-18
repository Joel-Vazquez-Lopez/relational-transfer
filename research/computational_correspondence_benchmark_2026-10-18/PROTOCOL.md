# Computational correspondence: known-program identification ladder

Status: prospective protocol. Execute only after this source and protocol are committed.
Date: 2026-10-18. Local CPU only; no trained-model inference or paid APIs.
Starting research branch: research/causal-developmental-relations-2026-10-17, f9edba11bb26d138fc478be055d997165f16ea79.

## Why this experiment

Previous single-path geometry was dominated by common schedules; the copying counterexample and layer-write calibration already showed why geometric transitions need not identify relational processing. The verified source-access rerun passed numerical controls but stopped on a fresh baseline failure before evaluation interventions. Those failed gates remain unchanged.

We validate measurement on programs whose dependence structure is known. This does not test a shared Transformer mechanism or establish a new linguistic law. Identical observations cannot identify different counterfactual dependencies: that limitation is established, not a novelty claim.

## Fixed programs and data

Input factors are O (surface order), V (voice code), R=OV (role), Q (question demand), L (answer identity mapping), and two nuisance inputs. Outputs Y=RQL are continuous answer coordinates. These codes are a mathematical toy inspired by linguistic contrasts, not annotated natural-language sentences.

All programs compute z=OV before Q arrives. Cached uses zQL; recompute uses OVQL, leaving z unused; threshold uses sign(z)QL. Natural z is ±1, making complete natural trajectories and outputs exactly identical between these programs in the base coordinates. An additional context program uses cached when |C1|>.75, otherwise recompute. Sign(0)=+1; tested doses avoid zero.

Five recorded stages: initial, cache, query, answer, readout. Six cached variants: orthogonal rotation, isometric width expansion 10→14, bounded nonorthogonal mixing (condition number four), inserted noops, nuisance clock amplitude 20→200, and invertible nonlinear shear u_z=z+.5OV followed by rotation. Noop stages never recompute or overwrite registers. All states are fully observed, not selected tokens or only answer axes.

Each split has 64 cases: full O,V,Q,L factorial × four nuisance sign combinations. Development C1=±1; held C1 magnitudes .37 and 1.31 determined by the second nuisance sign. Held intervention outcomes are Patient Q=-1; development alignment uses Agent Q=+1. Normal Patient states are used to fit the observational linear encoders/decoders: Patient interventions, not all Patient data, are held out. Seeds 11,23 calibrate thresholds; 37,41,53,67,79,97 evaluate new encodings. This is six repeated parameterizations, not six independent linguistic datasets.

## Measurements and predictions frozen before execution

1. Normal output and trajectory matching cannot distinguish the base cached, recompute, threshold or development-context programs. Final/trajectory geometry therefore has an information limitation, not an implementation failure.
2. Whole-state swaps from an O-reversed donor change all routes identically. A coherent donor state changes both raw inputs and the cached role, leaving the competing routes indistinguishable.
3. Targeted z swaps after cache/before answer change cached and threshold, not recompute. Targeted O swaps at the same stages change recompute, not cached/threshold. Initial O swaps affect all programs, since cache has not yet been constructed; post-answer targeted swaps affect none.
4. Graded z interventions distinguish smooth cached (effect equals dose) from threshold (effect zero at .25 and one at .75), although full z swaps cannot. Recompute is unaffected. This is a continuous response distinction: a discrete answer alone would not distinguish every dose.
5. Fitted causal profiles should survive declared affine coordinate transformations, width expansion, idle steps and clock amplitude. This follows from the construction; success is a calibration check, not discovery of natural invariance.
6. Nonlinear shear is a deliberate out-of-affine-domain stress. The fitted role direction includes .5OV, so dose .75 produces native effect 1.125 after cache, instead of .75. This challenges universal coordinate invariance. We will not alter the intervention to repair it after observing outcomes.
7. Context routing looks cached in development; half of fresh nuisance contexts recompute. Normal observations cannot reveal this change, but intervention predictions should fail in those contexts. Finite calibration does not license global algorithm equivalence.

## How interventions are fitted

Standard OLS on balanced normal observations: design [1,O,V,R,Q,L,Y,C1,C2]. Encoder contrast coefficients and linear decoders are fitted independently per stage; no native decoder matrix or program graph is passed to this fit. A patch adds dose × (flipped target value − decoded current value) × fitted encoder direction. Directions with decoding residual ≥1e-8 are marked unavailable and left unchanged; availability is saved and does not demonstrate semantic absence. Role-before-cache is unavailable in base programs. Decoder/encoder interventions are axis/factor dependent and can be off the natural manifold.

Raw effects are (patched Y − clean Y)/(-2 clean Y), without clipping. Save every individual case, stage and channel: whole, role full, order full, role .25, role .75, dose-zero sham, nuisance full. A nuisance patch flips C1's sign, not magnitude, so it does not cross the context gate. Baseline outputs and controls are reported per program instance separately from per-answer totals.

## Stage correspondence and locked evaluation

Methods: normalized-depth DTW; cross-stage linear CKA DTW; whole-state causal profile; whole+role-full+order-full profile; those plus role-.25. Cost for profiles is mean squared component difference. Monotone DTW covers both endpoints, allows diagonal/horizontal/vertical moves, with .01 extra penalty for each nondiagonal move; score is RMS unpenalized cost along the selected path. No independent unconstrained permutation is allowed.

Alignment is fitted on development Agent responses. Using that fixed path, predict held Patient role-.75 effects from source cached-base held responses, averaging mapped source stages. Report stagewise truth/prediction, RMS error, and all stress failures. Averaging across cases is used only to fit/score the profile, and raw individual responses remain saved.

Per method, a separation threshold is halfway between the largest development same-program distance and smallest development different-program distance. If the gap is ≤1e-7, method is declared nonseparating; no threshold tuning or replacement. Core positives: rotation, width, affine, idle, nuisance. Core negatives: recompute, threshold. Nonlinear and context are stress tests, excluded from threshold fitting and primary classification. All outcomes are reported whether or not a method separates.

## Existing methods and scope

An operational DSA building-block baseline uses unchanged upstream DMD plus angular SimilarityTransformDist, n_delays=2, rank=8, learning rate .01, 200 iterations, CPU. Exactly 64 benchmark comparisons (8 seeds × 8 targets excluding context) plus one independent preflight self-comparison. No post-outcome tuning. DSA models autonomous dynamics, so finite feedforward programs and arbitrary noops stretch its intended assumptions; this is descriptive application, not a full reproduction or refutation. Scores are distance-like (self zero). Errors are saved, not silently dropped.

Upstream code: https://github.com/mitchellostrow/DSA . Vendor modules retain exact Git blob hashes in SOURCE_MANIFEST.json; MIT license included. The minimal namespace loader is ours. POT is absent: an import guard raises if any transport function is called; only the angular path is used. No optional OT results are fabricated. Preflight checks unrelated encoding roundtrips, design rank and DSA API only.

Congruity (Sun & Toneva, https://arxiv.org/abs/2603.30002) already compares mechanistic interpretations using causally sampled output-preserving implementations. We do not reproduce its GetImpl sampler and do not categorize it as an observational method disproved here. InputDSA (https://arxiv.org/abs/2510.25943) addresses driven dynamics; not implemented here without a suitable controlled-system specification. Tensor similarity (https://arxiv.org/abs/2605.15183) has particular multilinear/model-class assumptions, including a bilinear attention variant; not evaluated on these arbitrary conditional programs. Existing causal abstraction/interchange (https://arxiv.org/abs/2106.02997), OLS, DTW and DSA (https://arxiv.org/abs/2306.10168) are prior tools, not inventions of this study. Dai et al. (2024), https://aclanthology.org/2024.emnlp-main.967/ , is relevant to binding interventions; this synthetic benchmark does not establish a more novel binding mechanism.

## Decision and stopping

A useful result is a bounded identification ladder with accurate held predictions and explicit counterexamples to excessive invariance. If controls or independent oracles fail, report invalid/implementation failure; do not adapt the study and rerun heldout outcomes. Even a complete core pass licenses only this declared affine/intervention domain, not unrestricted algorithm identity or cross-LLM equivalence. Preserve raw NPZ, all metrics, code, report and source provenance; publish text/SVG artifacts with local binary hashes if binary upload is unavailable. Stop after this benchmark and report. No Qwen/Llama inference, retraining or subsequent adaptive experiments.
