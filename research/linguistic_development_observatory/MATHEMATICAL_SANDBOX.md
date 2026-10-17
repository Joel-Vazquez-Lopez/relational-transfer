# Mathematics sandbox — compare development, not just points

**Status:** candidate analyses and thought experiments, not novel validated estimators. Prefer the least complex model that explains the observations. The goal is to *discover* useful mathematical objects, not bless Path-Cosine.

## Inputs to any future analysis

For model m, stimulus x, layer/block index l, token site s, record hidden state h[m,x,l,s] with exact tensor site semantics. Linguistic transformation T yields a contrast
  Delta_T[m,x,l,s] = h[m,T(x),l,s] − h[m,x,l,s]
only when sites have an explicitly defensible correspondence. Record where that is impossible.

Plotting ||Delta|| says how large a change is but discards its direction. Within-model step changes
  V_T[l,s] = Delta_T[l+1,s] − Delta_T[l,s]
and their normalized Gram K_T[i,j,s] = cos(V_T[i,s],V_T[j,s])
preserve some directional relations independently of absolute translation/orthogonal rotation. They are **not** semantic invariants.

## Candidate A — multivariate shape / shared developmental rhythm

Given N sites (subject, verb, object, question), build an N×L family of curves (response magnitude, derivative, signed quantities when meaningfully defined). Ask whether relative transition times, site ordering and co-change patterns recur across models **after controls**. Normalize at most with transformations justified before evaluation; do not use flexible unconstrained warping to manufacture agreement.

## Candidate B — within-trajectory relational geometry

Compare K_T matrices or multiscale directional summaries, including off-diagonal long-range step pairs and uncertainty under lexical-family resampling. Existing exposed controls show that generic depth can dominate: unrelated contrasts exceed role swaps. Treat Path-Cosine as a FAILED specific detector on these exposed cases unless a truly new controlled question is defined.

## Candidate C — inter-perturbation relational atlas

From multiple matched transformations T1,...,Tk build labeled within-model relationships across sites/layers among Delta_T, V_T and interactions, not just individual curves. Active/passive R×V four-cell finite difference:
  I_RV = h(RV) − h(R) − h(V) + h(base).
Its presence/absence does **not** identify algorithmic dependence: coordinate changes and downstream nonlinearities can create or mask representation interactions. Distinguish semantic, syntax, name-position and output identity through crossed stimuli.

## Candidate D — inside-layer contribution process

For each architecture correctly log residual input, normalized inputs, attention sublayer output, head contributions (including output projections), MLP/gate contributions, residual output, and any additional sublayers. Verify a numerical **reconstruction identity** with tolerances appropriate to quantized MLX execution before interpreting diagrams. Layer norms and large attention weights alone are not explanations. Token interactions require values/outputs as well as attention weights.

Potential descriptors: a token-site × layer flow profile; differences under role/voice perturbations; head/MLP contribution orientation and opposition; interactions and cross-site change timing. These can be explored *without assuming* one-to-one head matching across Qwen/Llama.

## Candidate E — causal functional relationships

Only after descriptive calibration, predeclare a patch/ablation or naturally matched causal effect. Predict held-out selective effect sign/rank, separating semantically correct counterfactual acquisition from arbitrary output change. Compare against cheap depth, position, answer name, step-norm and generic response-profile baselines. Interventions may be off-manifold. A late patch flip is not the onset of role representation.

## Known mathematical pitfalls

1. **Different dimensions:** within-model Gram structures can be same shape across hidden widths, but that does not establish shared computation.
2. **Reflections:** vector paths v and -v have the same within-path cosine Gram; they may or may not have corresponding functions.
3. **Operation order:** X→Y and Y→X can share endpoints, speed and full within-path directional Gram; a single unlabeled path may not identify the labeled sequence.
4. **Different maps by depth:** h[l+1]=F_l(h[l]); F_l need not equal F_(l+1), so ordinary autonomous dynamical claims need justification.
5. **Nonorthogonal coordinates:** Gram invariance does not extend to arbitrary invertible nonlinear changes of basis.
6. **Answer leakage:** even projecting out one output axis leaves distributed answer information.
7. **Cross-model alignment:** a learned depth warp, site alignment or projection can overfit exposed families and make false similarity.
8. **Norm/scale:** quantization, layer norms, residual growth and token position influence magnitudes and turning.

## Tiny analytical standard before a metric claim

Construct (i) exactly equivalent transformations under rotations/reflections/scale changes and modest time distortions; (ii) different algorithms with similar final outputs and path descriptors; (iii) output-code/copier examples with apparent linguistic alignment; (iv) a perturbation-factorial toy with known semantics and token order. Report what the candidate cannot distinguish. Calibration uses no new pretrained-model inference.

This file is a sandbox, not a demand that the final method use any of A–E.
