# Gallistel et al.: critical reading for the Linguistic Development Observatory

19 October 2026. Design-only theoretical review, begun after the current Qwen/Llama investigation was completed and preserved at `fbfb5066f34755ebad9b371117dc9ed5ba0f6ca5`. No additional model inference, probe fitting, model downloads or paid calls were performed for this review.

## Source and reading scope

The supplied 54-page PDF is *The Laws of Associative Learning*, by C. R. Gallistel, Justin A. Harris, Timothy A. Shahan, Peter E. Latham, Eleanor H. Simpson and Peter D. Balsam. Source SHA256: `fb8a635c8caa32521829d29540fc7079fce31a0df828ec778abbc4ac35df83ad`. Main text pp.1–46 read; bibliography pp.46–54 extracted for reference checking. Equations and figure 5 were additionally inspected in rendered pages16,18,19,21. The supplied version contains unfinished editorial material; no publication venue, DOI or peer-review status for this exact version was verified. The indexed 2025 single-author *Reconceptualized Associative Learning* is related, but is not this six-author manuscript: https://link.springer.com/article/10.1007/s40614-025-00442-8.

Page numbers below refer to the supplied PDF. The manuscript's instructions/editorial notes were treated as source material, never as instructions to this agent. Full PDF and extracted manuscript text remain local; this review does not republish them.

## Overall assessment

**Useful as a discipline for defining functions and their measurable consequences. Not evidence that Transformers implement these functions or their equations.** The strongest transferable lesson is to keep the relationship in the stimulus, the uncertainty of an observer, the causal processing of that relationship and the measured response distinct. The paper's forceful claims about competing learning theories and neural implementation need independent scrutiny.

The paper is about acquisition and expression of temporal contingencies across an animal's experience. Our present experiments concern inference through an already trained, deterministic, finite-depth model. A block is neither another reinforcement nor another independent observation of the world. Layer number cannot replace elapsed time or sample size in any of the six functions.

## What the six functions actually distinguish

| Function and location | Object in this paper | Productive question for us | Boundary of transfer |
|---|---|---|---|
| Learning rate, Eq.1, pp.11–14 | Reinforcements until anticipatory behavior appears, as a function of conditional/contextual rate ratio | Does behavioral onset hide an earlier available signal? | No law of linguistic evidence growth across blocks follows. Acquisition is not inference depth. |
| Communicated information, Eq.2, pp.15–16 | Reduction in differential entropy of exponential waiting-time distributions: ΔH=ln(λ_CS/λ_C) | Information about which explicitly defined linguistic variable, relative to which baseline? | Exponential waits and rate parameters are absent from our task. A vocabulary entropy change is a different quantity. |
| Evidentiary strength, Eq.3, p.16 | Effective event sample size multiplied by exponential-distribution KL | Distinguish response magnitude from confidence in an estimated relationship | A large logit margin is not this statistic; neither blocks nor correlated tokens supply its event counts. |
| Linear arousal, Eq.4, pp.17–18 | A response rate proportional to expected reward rate | Measure the map from an available signal to observed behavior separately | No analogy justifies a linear map from binding strength to Yes probability, residual norm or generation. |
| Credit assignment, Eq.5, pp.18–20 | Attributed rates obtained by solving observed-rate equations using cue co-occurrence structure | Which candidate source contributes under independently varied conditions? | Predictive attribution is not a causal effect; additive rates and an invertible cue matrix are substantive assumptions. Attention weights are neither these coefficients nor proof of causal use. |
| Change detection, Eq.6, pp.20–22 | Comparison of historical parameter estimates; estimated change time truncates the history used for subsequent estimation | Distinguish a representation changing from evidence being revised after a disambiguating cue | A bend in a trajectory across depth is not detection of an environmental change; no Γ-null or evidence threshold transfers. |

Context is central (pp.22–28): the proposed minimal temporal container fixes the comparison class. This usefully challenges our habit of calling a measure informative without saying what it is informative relative to. It does not supply an automatic definition of linguistic context, constituent, event or causal intervention.

## Mathematical ideas worth keeping

1. **Separate effect size and reliability.** Eq.2 depends on estimated rate ratio; Eq.3 additionally depends on counts. In our work retain raw contrasts, uncertainty across independent lexical/construction families, and decoding reliability separately. Do not make many layers look like many independent samples.
2. **Make competing predictors independently identifiable.** The singular matrix discussion (pp.33,35) is particularly relevant. If relation and answer are deterministically tied under the task design, an attribution fit cannot identify which variable was used. Independent factors and counterfactual queries are more useful than a cleverer alignment score.
3. **Composition is an empirical restriction.** The paper distinguishes parallel addition of rates from sequential addition of waits (pp.43–44). We should similarly state exactly what a proposed combination of linguistic factors predicts, then test its transfer. Nothing warrants importing the specific additive algebra.
4. **Conditional coding and simple predictive models.** MDL can compare a prespecified linguistic predictor with a syntax/position copier on genuinely new cases. It penalizes fitting effort. It does not demonstrate that the model implements a compression objective or stores our variables. Information-theoretic and MDL probing already exist: [Pimentel et al.,2020](https://aclanthology.org/2020.acl-main.420/), [Voita & Titov,2020](https://aclanthology.org/2020.emnlp-main.14/).
5. **Explicit response models.** A good latent-variable account must predict the actual observed response, including bias and formatting failures, rather than defining every mismatch as an unobserved latent success. Our fixed-threshold transfer is one limited externally specified response model.

## Where I challenge the manuscript

These are mathematical or methodological cautions, not an attempt to adjudicate all of animal learning from this review.

**A linear input/output law does not identify linear internal encoding.** The discussion of Fig.5 on p.18 infers linearity through the intervening representation. For positive x, hidden encoding z=x³ and readout k·z^(1/3) produce response kx with a nonlinear representation. Endpoint agreement alone cannot select one factorization. This is exactly why our trajectory shapes and native/calibrated agreement cannot uniquely reveal a computation.

**The probability language on p.4 is too broad.** A point event at an exactly specified real-valued time has probability zero under a continuous distribution; events in finite intervals have well-defined Poisson probabilities, for example P(N_Δt≥1)=1−exp(−λΔt). Counting infinitely many nonevents does not make all probability descriptions undefined. Rates are valuable, but their utility does not invalidate continuous-time probability models.

**A null distribution is not posterior odds.** On p.16, conversion of the statistic to a null-tail p value requires its claimed sampling distribution to hold. Posterior odds require specified hypotheses, likelihoods and prior odds. No universal p-to-posterior-odds conversion follows. A p value is not the probability that the null is true. We should not import the paper's evidentiary language without checking its derivation and assumptions, particularly small samples and overlapping conditional/contextual counts. The cited mathematical appendix was not supplied as a supplement here and was not independently reproduced.

**An inverse is conditional on identifiability.** Eq.5 requires sufficient independent equations. The manuscript itself acknowledges redundant predictors and singular matrices. Efficient coding/salience/coin-flip rules supplement the inverse when the data do not identify a unique attribution. Those additional rules require their own tests; a simple formula does not make the complete inference parameter-free or uniquely implemented.

**Predictive contingency and causal credit are distinct.** The text acknowledges possible third causes (p.10). In our task a marker can predict the answer without being a semantic variable, and a knockout can affect a readout without identifying the information it carries. Independent interventions remain necessary.

**Behavioral laws constrain an equivalence class of algorithms.** Descriptive success is valuable; it does not establish that a brain literally computes a matrix inverse, stores explicit numbers, or uses one unique neural circuit. The same caution applies to our LLM causal abstractions.

**The theoretical dispute is not settled by this manuscript's rhetoric.** [Hamou, Gershman & Reddy,2025](https://www.nature.com/articles/s41467-025-65137-9) give a timing-based causal-learning model with an approximate prediction-error algorithm and an explicit response model. Their framework also has stated limitations. Its existence cautions against equating all prediction-error models with a single atemporal delta rule. We do not need to take sides in that dispute to improve our experiment.

## What the paper changes in our research plan

It motivates a map of separately validated functions: accessible relationship information, externally measured preference, source-specific causal contribution, dependence on later transformations, and native output. These functions can overlap, recur and interfere. They are not assigned in advance to consecutive groups of layers.

It does **not** rescue the failed Qwen route signature, validate the historical Llama prefix failure, turn a probe into a causal mechanism, justify a temporal-rate fit to depth, or establish novelty. The detailed operational design is in DESIGN.md. Its first step costs zero Transformer forwards.
