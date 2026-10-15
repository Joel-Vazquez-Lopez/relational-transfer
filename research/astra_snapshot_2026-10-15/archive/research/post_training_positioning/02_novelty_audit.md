# Novelty audit and operator assessment

Evidence IDs refer to [the verified literature map](01_literature_map.md). Claims below are interpretations unless tied to a source.

## Judgment by claim

| Claim | Judgment | Reason |
|---|---|---|
| “A checkpoint's future learning matters” | **Established** | MAML/FOMAML/Reptile explicitly optimize post-update query performance (L9–L10); MetaICL does so for in-context adaptation (L11). |
| “Training can reduce future plasticity” | **Established outside and emerging within LLMs** | Continual-learning evidence L12–L13, LLM pretraining L6, post-training L1–L5. |
| “A better SFT checkpoint may fare worse after RL” | **Established** | Directly tested by peer-reviewed L1–L2 and L3. A thesis repeating this on another dataset is incremental. |
| “SFT, RLVR and DPO create a universal plasticity ordering” | **Unsupported and unlikely** | Treatments differ in data and update dynamics; L1–L8 show context dependence. A universal scalar would need cross-task, cross-optimizer invariance. |
| “Equal immediate score can conceal different later learning” | **Direct precedents, narrower open extension** | L1–L2 show this for SFT→RL. Equal aggregate score is also a weak competence match. Cross-family matched-profile contrasts remain a potential experiment. |
| “The response operator is new” | **No** | It bundles ordinary transfer/learning curves indexed by task, budget and update rule. Notation alone supplies no new estimator or theorem. |
| “Training history has a causal effect beyond initial competence” | **Potentially distinct but hard** | Randomized histories from one base, matched exposure and predeclared downstream tasks could estimate total history effects. Conditioning on a post-treatment score changes the estimand and risks collider/selection bias. |

## Seven objections and responses

1. **Conditional learnability.** Define the object as a *surface* indexed by task and update rule. Report Kendall/Spearman checkpoint-rank correlations across tasks and between full fine-tuning and LoRA, plus variance components and rank reversals. A weak rank correlation rejects a universal checkpoint score; it does not make a structured response study worthless. Do not average away reversals.
2. **Baseline matching.** Match or stratify on item-level response vectors, task-wise accuracy, difficulty-bin accuracy, calibration (Brier/ECE with uncertainty), answer length and high-`k` success. Aggregate score matching alone is insufficient. Because competence is post-treatment, primary causal estimates compare randomized branches unconditioned; matched analyses are secondary descriptive contrasts. Use overlap diagnostics and refuse extrapolation if no match exists.
3. **Training history confounding.** Randomize branches from the *same exact base weights*. Hold prompt IDs, solutions where possible, total target tokens, rollout attempts, parameter-update budgets and optimizer tuning protocol fixed; report mismatches (RL is on-policy, so identical consumed data cannot literally be imposed). Use a common offline pool for SFT/DPO and a prompt-controlled RLVR branch; include a replay/compute-matched SFT control. Measure KL to base, normalized parameter displacement and output entropy as mediators/descriptors, not automatic covariates: controlling them may remove a true treatment effect.
4. **Fine-tuning noise.** The independent replicate is a full post-training seed, not a downstream item. Use at least 3 independent history seeds for a screening study, preferably 5 for confirmation; pair identical downstream data orders and evaluation items. Bootstrap items within each seed and use seed-level paired contrasts/hierarchical intervals. Report all curves and 95% intervals. Pre-register a minimum effect of practical interest (e.g. 5 percentage-point mean target accuracy or 10% relative AUC), then do a pilot-based power simulation; no honest universal seed count can be guaranteed beforehand.
5. **Contamination.** Public math benchmarks can be in pretraining. Generate held-out grammars/DSLs *after* freezing history recipes and store generator source and seeds. Hold out templates, operators or composition rules, not just random instances; check train/test normalized string overlap and near duplicates. This supports claims about unseen instances/compositions under a known generator, not proof of unseen concepts relative to opaque pretraining.
6. **No universal measure.** If response rankings reverse by task or update rule, the proper result is a treatment×task×procedure interaction or low-rank response structure, with confidence intervals. A universal “plasticity score” is falsified; a conditional transfer map may remain useful if it predicts held-out tasks.
7. **Already explained.** If L1's validation loss and high-`k` pass rate, plus baseline item profile and entropy, predict downstream learning as well as history labels out of sample, and L4–L6 account for observed changes, the proposed claim has no residual scientific content. Publish a careful negative replication only if preregistered and sufficiently powered; otherwise stop.

## What the operator measures

For checkpoint `θ`, target task `τ`, update rule `U`, data order/seed `s`, and budget `k`, define both the absolute curve `A(θ,τ,U,k,s)` and change `Δ=A(k)-A(0)`. This is an **interventional response to a specified training recipe**, not an intrinsic property of `θ` alone. It includes baseline competence, learning rate, asymptote, overfitting and forgetting if `A` is a metric vector.

`Δ` has a floor/ceiling problem: a strong model has less headroom, and the same absolute post-update score can produce very different gains. A normalized gain `(A(k)-A(0))/(1-A(0))` magnifies noise near 1 and assumes a meaningful ceiling. Report **absolute performance** as primary; baseline and delta as paired diagnostics; trapezoidal AUC over predeclared sample budgets for early sample efficiency; a late-budget score for attainable performance; and old-task retention separately. Do not collapse into one scalar by default. For loss, use held-out log loss alongside accuracy to avoid threshold artifacts.

The operator adds value only if its *multivariate structure* yields a validated prediction or falsification unavailable from ordinary baseline measures. Candidate analysis: center curves by task and seed, estimate checkpoint×task/procedure interaction variance, perform PCA on *training* task families, then predict the **held-out family** response. Compare with simple baselines: immediate accuracy, item-profile embedding, validation loss, high-`k` pass rate and parameter distance. Nested cross-validation must keep all checkpoints from the same history seed together; otherwise leakage makes structure look predictive.

## Causal estimands

- **Total randomized history effect:** `E[A(θ_H1,τ,U,k) - A(θ_H0,τ,U,k)]` under a prespecified history intervention and shared base. This includes any initial competence change.
- **Performance-matched descriptive effect:** difference among checkpoints with overlapping baseline profiles. It cannot automatically be called the causal effect “independent of competence,” because baseline is itself affected by history.
- **Mechanism effect:** requires an actual intervention on a proposed mediator (e.g. matched entropy restoration or targeted weight fusion) while measuring unintended changes. Gradient angles, effective rank and curvature alone are correlates.

**Decision:** revise the project to a narrow, causal, cross-task *characterization* question. Do not introduce a new learnability objective or optimizer in the MSc phase.
