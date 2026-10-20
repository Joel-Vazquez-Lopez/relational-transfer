# The paths tell us what a single-path metric cannot identify

**17 October 2026 — exploratory analysis, no new model inference.** I reanalysed the saved full-dimensional `[4 cases, 2 variants, 29 stages, 4 sites, hidden width]` arrays with [reproducible code](explore_path_structure.py). All four examples were already exposed and used for method development. These results are neither held-out validation nor a paper claim.

## Scientific reading

The first striking observation is real but deceptive. At the final prompt token, the four role-swap contrasts end in nearly unrelated directions: mean pairwise endpoint cosine was **0.03** in Qwen 7B and **0.07** in Llama 3B. Yet the full-dimensional *within-path* matrices of cosines between every pair of adjacent-layer updates have similar patterns across the four examples: mean pairwise matrix correlations **0.82** in Qwen and **0.79** in Llama. Their layerwise movement-size profiles correlate about **0.98** within each model. Looking only at these numbers, one could be tempted to say that different-looking paths share a computational shape.

The controls reject that inference. When I subtract the hidden trajectory for one base sentence from a *different* base sentence, rather than comparing a sentence with its own role reversal, the within-model turning-pattern correlations are **0.92** in Qwen and **0.90** in Llama—higher than the role-swap values. Across models, the mean final-token turning-pattern correlation is **0.575** for role swaps versus **0.586** for unrelated-sentence contrasts. Matching the exact same sentence pair across models provides almost no advantage: **0.581** matched versus **0.573** mismatched. Movement-size profiles are similarly generic. Resampling the paths at 2-, 4-, and 7-block scales raises both similarities; the unrelated control remains at least as similar as the role contrast.

![Full-dimensional path controls](PATH_STRUCTURE_CONTROLS.png)

There is a strong model-specific *depth schedule*. For Qwen, the largest five updates account for an average **85%** of final-token role-contrast travel; for Llama, **53%**. The last block is the largest step for **all four role contrasts, all four unrelated-sentence contrasts, and all four raw base trajectories in each model**. Thus the earlier “late Qwen, earlier Llama” observation is useful phenomenology, but the final-block burst is not role-specific. The raw final-token paths themselves have within-model Gram correlations **0.99** and **0.96**, showing how much common architecture/prompt processing can drive shape similarity.

The paths are genuinely irregular. Between adjacent updates of the role-contrast vector, the step-direction cosine is negative in **20–24 of 27** pairs for Qwen and **23–26 of 27** for Llama. Unrelated contrasts are similarly jagged. I retained these reversals rather than smoothing them into a monotone phase story. At the token sites, the subject and object contrasts are almost antiparallel early (stage 3 mean cosines **−0.97** Qwen, **−0.91** Llama), then less so late (**−0.20**, **−0.53**). Since swapping the two name tokens produces opposite lexical difference vectors by construction, even this orderly evolution cannot be called a role-binding mechanism without an independent control.

**Conclusion from the actual four trajectories:** there is reproducible within-model structure in how contrast paths turn and grow, and models differ in timing and scale. A single-path similarity score does **not** identify a linguistic computation here. The strongest explanation still alive is a generic layer schedule plus example-specific lexical and answer information. Whether a task-specific developmental relation exists remains open.

## Mathematical reason the first Path-Cosine idea can fail

Let `δ_l = h_l(x_changed) − h_l(x_base)` and `v_l = δ_l − δ_{l−1}`. Path-Cosine uses adjacent `cos(v_l,v_{l+1})`; an expanded version uses the full step Gram matrix `G_ij = cos(v_i,v_j)`. These are invariant to a shared orthogonal change of hidden coordinates, which is desirable. They also discard which *linguistic operation* a direction represents.

The [small mathematical control in the code](PATH_STRUCTURE_ANALYSIS.json) makes the loss exact. Path A performs step X then Y: `(0,0)→(1,0)→(1,1)`. Path B performs Y then X: `(0,0)→(0,1)→(1,1)`. They have identical endpoints, step lengths, adjacent cosine and full step Gram. A rotation/reflection exchanges their axes, so any single-path descriptor invariant to all orthogonal coordinate changes must treat them as equivalent, although the labeled operation order differs. A signed second-order path area flips sign, but its sign also flips under coordinate reflection; taking an invariant magnitude again makes the paths equal. Path signatures can preserve order in a *fixed* coordinate system, but do not by themselves resolve this cross-model semantic-axis ambiguity.

Add an independently labeled X-anchor trajectory and compare **cross-path** step relations. X then Y has cross-anchor dot pattern `[[1,1],[0,0]]`; Y then X has `[[0,0],[1,1]]`. Those patterns survive a shared orthogonal transformation of both paths while preserving the difference in labeled order. This is a positive mathematical control for studying *interactions among perturbation trajectories*, not proof that our LLM paths have an X or Y axis.

## Best candidate: a labeled perturbation-relation atlas

The next object should be a small family of controlled, paired trajectories, not one scalar per path. For model `m`, transformation `p`, token site `s`, and layer `l`, define a contrast `δ[m,p,s,l]` and its update `v[m,p,s,l]`. Then record the **labeled relation tensor**

`R_m[(p,s,l),(q,t,k)] = cos(v[m,p,s,l], v[m,q,t,k])`,

alongside separate channels for relative update size and the model's *selective causal answer effect* under a patch at `(p,s,l)`. This is a Gram/kernel construction, not a claim to have invented a new mathematical primitive. The potentially useful contribution would be a carefully controlled linguistic perturbation design and evidence that relationships in `R_m` predict unseen causal responses better than single-path geometry, CKA, or a generic depth schedule.

The tensor preserves relative direction, interaction between labeled transformations and token sites, and update order. It is invariant to one shared orthogonal change of coordinates within a model and tolerates different hidden widths across models because each model's tensor lives in the same *label space*. It loses absolute neural directions and can miss computations encoded by nonlinear or nonorthogonal coordinate changes. Unit directions remove per-step magnitude; therefore amplitude must be reported separately. It is **not** invariant to arbitrary layer-specific rotations, unconstrained time warping, or a change in the perturbation set. These restrictions should be stated rather than hidden with a flexible learned alignment.

The four existing examples offer only one transformation family, so they cannot validate this atlas. Their raw-versus-unrelated control is exactly the warning: the generic layer schedule must be estimated and beaten. A label-specific residual is an *exploratory diagnostic*; subtracting a common schedule on these four examples would risk manufacturing a role signal. I have not done that.

## What prior methods already cover

| Method | Established contribution | Boundary for our question |
|---|---|---|
| [CKA](https://proceedings.mlr.press/v97/kornblith19a.html) | Compares sample-representation geometry across independently trained networks without matching neuron coordinates. | Does not alone test ordered, selective causal dependencies; our eight prompts are too few for a credible CKA benchmark. |
| [DSA](https://papers.neurips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html) | Compares temporal dynamics beyond static geometry in recurrent systems. | Treating 28 different Transformer blocks as repeated applications of one stationary operator would require an argument we do not have. |
| [InputDSA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a437d77270d5588a84d63a851a3fdeb-Abstract-Conference.html) | Separates intrinsic and externally driven dynamics in recurrent systems. | Highly relevant precedent for disentangling input effects; not an off-the-shelf estimator of linguistic causality from four feedforward paths. |
| [Path signatures](https://proceedings.mlr.press/v130/lemercier21a/lemercier21a.pdf) | Summarize ordered paths with useful time-reparameterization invariance. | Standard signatures use coordinates; an orthogonally invariant summary can erase the semantic order distinction in our toy control. |
| [Transformer trajectory geometry](https://arxiv.org/abs/2606.09287) | Already studies length, curvature, convergence and stability across layers. | Our current observations of turns and timing are not novel on their own; causal/task specificity is the missing test. |

This is a targeted prior-work boundary, not a comprehensive novelty claim. Existing representational kernels and causal abstraction may subsume important pieces of the proposed atlas.

## A falsifiable next experiment

Build a **small actor-identity × voice factorial** with both active and passive sentences. Each lexical family has four cells: actor A or B, expressed actively or passively, with the same event and unambiguous question. This yields labeled role-change and syntax-change trajectories with shared lexical ingredients. Add a carefully controlled negation/status contrast only after checking that both models can answer its baseline question. Use fresh names and verbs; split lexical families before any method choice. Keep the existing four families as development examples only.

On development families, compare the labeled relation tensor with single-path Gram/Path-Cosine, normalized depth, an answer-probability trajectory, and CKA where sample count is sufficient. Include identity-only, word-order-only, and wrong-question controls. Freeze a simple layer mapping (initially normalized depth; no unconstrained warping) and the answer scorer **before** held-out results. Score full name strings in Llama, fixing the first-token failure in the previous pilot.

The first held-out test asks whether the same *role* transformation across models is more related than role-to-syntax or role-to-negation after accounting for the generic schedule and lexical controls. The stronger test withholds one target-model perturbation family and predicts its selective patch-effect profile from source-model relations to the common anchors. Site-specific patches and wrong-site shams must distinguish role information from generic answer-state transfer. If the atlas cannot beat the unrelated/syntax controls or predict held-out causal effects, reject the shared-process interpretation rather than tune the metric on the test set.

## Provenance and limits

[Machine-readable full results](PATH_STRUCTURE_ANALYSIS.json), [analysis/figure code](explore_path_structure.py), [control figure](PATH_STRUCTURE_CONTROLS.png), and the earlier [vector arrays](REPRESENTATION_PATH_REPORT.md) are saved in this workstream. The unrelated control is deliberately broad: it changes names and sometimes verbs, so it tests **genericity**, not a perfectly matched linguistic rival. The tiny sample, common chat prompts, both models' 28-block depth, float16 saved vectors and selected token sites limit generalization. No new held-out model behavior, causal intervention, model training, or paid inference was performed in this analysis.
