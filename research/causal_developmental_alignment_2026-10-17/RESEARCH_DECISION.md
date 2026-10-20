# Causal developmental alignment: a new research bet

Date: 2026-10-17. Status: design and literature audit, **no new experiment or positive result**.

**Exploratory update:** The subsequent [full-dimensional analysis](PATH_STRUCTURE_INTERPRETATION.md) found that single-path Gram/turning similarity is at least as strong for unrelated-sentence contrasts as for role swaps. A toy control shows a formal inability of an unlabeled orthogonally invariant single-path descriptor to identify operation order. The strongest current candidate is a labeled *multi-perturbation* relation tensor with held-out causal-effect prediction; this supersedes any suggestion here that a single path cosine would be sufficient. All numerical findings remain exploratory.

## Question

When two language models answer the same linguistic question correctly, do they use the same *causal dependencies* at corresponding processing stages, or do they arrive at the same answer by different routes? A second, conditional question asks whether those dependencies emerge in the same order during training once models are aligned by independently measured competence, rather than raw update count.

The two axes must stay separate. Transformer layers within a trained model are processing stages; training checkpoints are learning stages. A similar-looking curve on those axes is not evidence that they are the same process.

## The methodological gap: comparing paths

The object of comparison should be an **ordered path of causal response profiles**, not a set of final-layer vectors. Let `K_m(s, q, a)` denote the output effect in model `m` at stage `s` when intervention `q` is made and answer contrast `a` is scored, after subtracting a matched sham. An alignment between models is a monotone map `phi` between stages that minimizes the difference between their `K` profiles on *development interventions only*. It must preserve order, allow unequal numbers of stages, and penalize stretches that match many stages to one stage. On held-out interventions, compare the predicted effect signs, ranks, sites, and transition order. No rotation between hidden spaces is required.

This answers the user's sharper question: can we compare **how a computation changes**, rather than comparing a photograph of two final states? The method is not automatically novel. Geometric layer trajectories and training trajectories already exist. Its possible added value is that the path's coordinates are selective, tested causal effects, and that an alignment predicts unseen interventions. If a simpler normalized-depth or output-probability path predicts equally well, the method has added nothing.

## Stronger target: tested process equivalence, not one similarity score

The third pasted note makes an important correction to the design above. A cosine-like path score can assist discovery, but a high score is neither necessary nor sufficient for the models to compute the same thing. The primary object should be a **bounded counterfactual specification**: a set of tested operations, their selective causal dependencies, and the predictions those dependencies make for new combinations of inputs. Two models count as equivalent only *relative to that declared set of interventions and outputs*. The method cannot establish unrestricted algorithmic identity.

For example, a model that binds participants before applying negation and one that routes negation first may agree on all simple sentences. To distinguish them, construct counterfactual cases in which role and negation information can be separately changed; determine whether patching one dependency changes the downstream use of the other. A claimed edge requires a selective effect and a control for generic disruption. A claimed ordering requires more than observing an early probe or attention pattern: it needs a frozen intervention prediction that fails if the order is reversed.

Before applying this to language models, test it on small systems whose algorithms are known by construction. Include (a) the same algorithm with permuted/rotated hidden coordinates, (b) the same algorithm spread over different numbers of layers, (c) different algorithms with identical answers on the evaluation set, and (d) shared intermediate geometry but different causal dependencies. These are *identification tests* for the method. If it cannot separate case (c) from (a), stop. A successful toy benchmark would validate the method's discrimination, not establish that pretrained LLMs share a linguistic algorithm.

## The central measurable object

For each model, input pair, layer, and controlled transformation, measure a **selective causal response**: how much a matched activation intervention changes the logit margin for the intended answer, compared with a matched sham and a wrong-site intervention. The transformation set should include event swap, actor/patient reversal, negation, and an irrelevant named participant. For each transformation, record the whole *answer distribution change*, not only whether top-1 changed. Keep role and event contrasts balanced across lexical items and token positions.

This gives an intervention-effect tensor indexed by transformation, answer contrast, token site, and layer. It is comparable across hidden widths because its entries are output effects. It is **not** a cross-model activation transplant and does not require learning a flexible map between vector spaces. A predeclared layer alignment may map only monotonically ordered depth bins, and must be learned without access to held-out transformation effects.

## A falsifiable prediction

Train an alignment on role reversal and event swap in development sentences. On untouched lexical families and sentence frames, ask whether that alignment predicts the **relative layer and site** of selective effects for negation and irrelevant-participant interventions, and predicts which wrong answer source appears. Compare against (1) output-behavior matching alone, (2) normalized-depth alignment, (3) CKA/RSA-based layer matching, (4) a position-and-token-copy heuristic, and (5) a shuffled alignment. A meaningful positive result requires out-of-sample predictive gain over all five, comparable baseline task competence, and selective interventions whose shams do not reproduce the effect.

The strongest rival is mundane: similar output heads, common syntax, or token position could generate similar intervention-effect profiles without shared internal algorithms. Even successful held-out prediction would support *shared causal organization at the resolution tested*, not identity of circuits or a universal developmental law. Failure to beat the position/output baselines would end the claim.

## A small first gate, not a checkpoint sweep

1. Build a competence-screened, balanced development set and untouched evaluation set. Use paired counterfactual sentences with answer labels fixed before model inference. Start with the existing local Qwen 7B and Llama 3B; Qwen 1.5B enters only where it passes the same competence gate. The earlier named-observer panel is exposed development material, not a fresh evaluation set.
2. Run one within-model intervention feasibility pilot on development items: clean/corrupt pairs, answer-logit margins, sham patches, nearby-token patches, and matched-token-length controls. Verify that the intervention measures a selective role/event effect rather than generic answer disruption. Freeze the site-selection and alignment recipe before new items are evaluated.
3. Only if the first two gates pass, run held-out prediction across models. Do not choose a new layer, intervention, or transformation after seeing held-out outcomes.
4. Add training time only when independent local checkpoints from comparable models exist. Currently the cache contains one Pythia-160M snapshot, not a local Pythia/PolyPythias training series. The project's synthetic listener checkpoints offer a separate *toy* acquisition testbed, but their D82/D86 results and input-grammar confound prevent using them as pretrained-language confirmation.

## Existing evidence that constrains this bet

The local [Track A report](../../track_a_development_2026-10-03/TRACK_A_FULL_REPORT.md) already found that feature distances can increase while unsupported relational behavior worsens. Its first prospective symmetric-world test missed its threshold and was later found to have a padding/identity alias; the repaired prospective test showed a bounded relative interaction but missed its full predeclared criterion. That is motivation for measuring causal *use* rather than geometric presence, not evidence for shared development across models.

The recent [named-observer report](../paper_pivot_2026-10-17/NAMED_OBSERVER_REPLICATION_REPORT.md) found a narrow Qwen 7B preposed, second-event error pattern; its broad frozen prediction failed and Llama 3B showed a different error profile. This gives an instructive disagreement for the new framework, but the exposed sentences cannot become held-out confirmation.

## Prior work and novelty boundary

- Kornblith et al., [CKA](https://proceedings.mlr.press/v97/kornblith19a.html), already align representations across initializations. Geometric alignment alone is not the proposed contribution.
- [Pythia](https://arxiv.org/abs/2304.01373) and [PolyPythias](https://arxiv.org/abs/2503.09543) already provide training trajectories and multi-seed comparisons.
- Geiger et al., [causal abstractions](https://arxiv.org/abs/2106.02997), already use interchange interventions to test aligned causal variables. We cannot claim invention of causal alignment.
- Shao et al., [Differentiable Faithfulness Alignment](https://arxiv.org/abs/2604.24302), already test cross-model circuit transfer; the present bet must show a distinct *held-out predictive question* and beat strong baselines.
- [A Negative Result on Cross-Model Activation Transfer](https://arxiv.org/abs/2606.03280) reports high mapped-state cosine similarity without functional transfer, reinforcing the need to score native causal effects.
- Sutter et al., [The Non-Linear Representation Dilemma](https://arxiv.org/abs/2507.08802), show that overly flexible alignment can make causal abstraction vacuous. Hence use a simple, predeclared monotone layer mapping and strict held-out tests.
- [Trajectory Geometry of Transformer Representations Across Layers](https://arxiv.org/abs/2606.09287) already studies geometric processing paths, while [The training process of many deep networks explores the same low-dimensional manifold](https://pmc.ncbi.nlm.nih.gov/articles/PMC10962999/) compares trajectories in prediction space. Neither fact licenses saying that no trajectory comparison exists; the narrower gap is causal, selective, out-of-sample comparison of linguistic computations.
- Ostrow et al., [Dynamical Similarity Analysis](https://papers.neurips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html), already compare computational dynamics beyond geometry, especially in recurrent systems. [Neural Networks as Paths through the Space of Representations](https://arxiv.org/abs/2206.10999) already compares layerwise paths. [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://www.jmlr.org/beta/papers/v26/23-0058.html) already formalizes intervention-based equivalence. A credible contribution must outperform these relevant baselines on an explicitly new identification/prediction problem, not merely combine their vocabulary.

## Decision

This is a plausible **new independent research direction**, not yet a paper result. The distinctive candidate contribution is a validated test of whether *selective intervention effects* transfer across models and development stages better than geometric and behavioral matching. If the small feasibility gate fails, keep the negative result and stop. If it passes, preregister one held-out comparison; do not expand into a broad adaptive search.
