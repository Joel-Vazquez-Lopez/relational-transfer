# Decision memo: relational acquisition project

**Decision date:** 2026-09-28. **Verdict: B — STRONG BOUNDED FINDING.** This is a viable 15-credit project if its central claim is restricted to controlled choice-set supervision and off-support relational generalization in the synthetic oracle-listener system. It is not yet a general law of transformer-native relational acquisition. All D01–D51 post-P3.2 results are adaptive discovery evidence; P3.2-E remains the canonical endpoint. No protected seeds 1704–1706 or protected confirmation material were used in this decision run.

## 1. The claim to commit to

**Candidate bounded finding:** In this six-identity transformer listener, changing which *observed* relational alternatives compete in training choice sets can select a computation that distinguishes and correctly handles *untrained* endpoint combinations, even when the support graph, queried relation set, training budget, and familiar-task performance are essentially unchanged. Increasing the same identities' marginal exposure without changing those comparisons does not reproduce the effect; the learned distinction is distributed across message and candidate computations and depends on training history.

The first sentence is the course-project core. The second names tested controls and a measured computation, not a universal mechanism. The broader idea that a numerical “contextual distinguishability” measure predicts acquisition across tasks, architectures, or language is **not established**. Nor is there evidence for a unique transformer-native grammar or operator.

The overlooked D51 check changes the scope of the claim. F introduced direct 1-versus-2 foils, but it improved **all three** withheld H–H pairs: accuracy 0–1 = 0.961, 0–2 = 0.979, 1–2 = 0.982, versus A's 0.580/0.587/0.579 and exposure-control H's 0.559/0.567/0.573. Mean source-query contrasts on common observed third-identity contexts were F 0.821/0.867/1.370, compared with A 0.044/0.035/0.030; target contrasts likewise grew (`d51_cross_pair_scope.csv`). A pair-local training-collision explanation is therefore false. F seems to select a more general H-endpoint computation or to break a wider H-class symmetry, but the present evidence does not distinguish those accounts.

## 2. Prediction versus retrospective fit

Before a new run, the bounded account predicts that an audited task with within-H same-role competitor foils, but no queried H–H relation, will produce higher *all-H–H* withheld accuracy and larger source/target query contrast than a task with the same topology, identity exposure, budget, and familiar fit but without those competitors. It predicts a group-level contrast, not a precise accuracy for every seed or a learning-time threshold.

This has **not** yet been tested prospectively. F was designed after A/E evidence; G/H controls were adaptive; D51 was discovered after F training. The present evidence supports a strong retrospective prediction and a viable preregistered test, not a confirmed general rule. The E route is less predictive: asymmetrical contexts can select separation, but tied E solves training almost as well, so no current statistic says when such a soft preference will win.

## 3. Acquisition → computation → behavior bridge

| Arrow | Assessment | Basis and limit |
|---|---|---|
| Experience → changed learning dynamics | **Strong discovery-level intervention** | A/F share graph, initialization seed, indexed schedule and training budget but differ in foils; F contrasts grow, G/H contrasts contract. E-versus-A continuation from exactly the same tied E state diverges after 2,000 updates. Novel conditions still have mostly one seed. |
| Learning dynamics → changed internal computation | **Observed longitudinally, causation incompletely localized** | Saved checkpoints show opposite trajectories in message-query and candidate-vector contrast; task switches and long F→A withdrawal show path dependence. Token patching identifies distributed routing in trained E. No necessary/sufficient parameter pathway or optimizer-level law has been demonstrated. |
| Changed computation → off-support behavior | **Partial causal support** | D5's preregistered support relocation moves both final-query compression and redirection. Tying E identity inputs causes a large withheld-pair decline despite similar familiar fit; releasing the tie under E recovers contrast and behavior. Identity merger and patching affect relevant computations, but query/candidate alignment is also a score-algebra description, and the entire transfer circuit is unproved. |

Thus there is more than an accuracy correlation, but less than a mechanistic explanation of circuit formation. The synthetic system provides controlled causal training and input interventions plus developmental readouts; it does not establish a reusable transformer algorithm.

## 4. Strongest alternative and decisive separation

The strongest live null is **generic task difficulty or hard-negative pressure**: F may force a broadly better identity representation because its foils are harder, regardless of their specific H–H comparative structure. G/H match exposure, not the full per-example hardness or gradient covariance. D51's all-H–H rescue makes this alternative more serious, not less. A second limit is architecture/optimizer specificity; no independent architecture comparison has been performed.

A prospective difficulty-matched control can separate the primary account from this null. It should introduce the same number of modified rows, preserve the true queried meanings and role-wise identity counts, and match the F arm's **frozen pretraining difficulty distribution** (e.g. initial-model correct-versus-hardest-foil margin histogram), while preventing two H identities from competing as the same-role alternative in one choice set. If that control improves all H–H pairs as much as the comparative arm, the comparison-specific explanation fails. The bounded observation that choice-set design matters may remain, but the proposed *relational* interpretation would have to be narrowed.

## 5. Compression and unexplained facts

One restricted story covers several outcomes: training comparisons can make an identity distinction useful to observed decisions (F), ordinary A/G/H schedules can fit training while suppressing it, and a learned separated solution can persist under later A training. This explains the broad F-versus-exposure-control contrast, developmental direction, and hysteresis without assigning a new explanation to every seed.

It does **not** explain why a single 1/2 foil intervention lifts 0/1 and 0/2 so strongly; why E usually separates despite an almost equally good tied solution; why E continuation from the tied state splits while A continuation does not; or why a one-step antisymmetric gradient at that state is larger for A than E (`d47_tied_gradient.csv`). These gaps block a general acquisition law. The current account compresses a useful subset of the evidence but not the whole programme.

## 6. Literature collision and novelty

| Established result | Collision with this project |
|---|---|
| [D'Amour et al., underspecification](https://arxiv.org/abs/2011.03395) | Similar in-domain fit can coexist with divergent out-of-domain behavior. That general phenomenon is not novel here. |
| [Csordás et al., transformer systematic generalization](https://aclanthology.org/2021.emnlp-main.49/) and [Bogin et al., unobserved local structures](https://aclanthology.org/2022.emnlp-main.175/) | Training choices and local support structure already affect systematic generalization, often invisibly to IID performance. |
| [Zhou et al., data factors](https://aclanthology.org/2023.emnlp-main.898/) and [Singh et al., scene-graph contrastive negatives](https://aclanthology.org/2023.emnlp-main.56/) | Example difficulty, data complexity and relation-aware negatives have already been used to improve compositional/relational generalization. “Harder relational foils help” alone is not a paper-level novel principle. |
| [Xue et al., feature suppression](https://proceedings.mlr.press/v202/xue23d.html) | Optimization can collapse or suppress task-relevant distinctions; a general implicit-bias interpretation is already established in related settings. |
| [Wang et al., grokked transformer reasoning](https://arxiv.org/abs/2405.15071) and [Tang et al., compositional circuit](https://arxiv.org/abs/2502.15801) | Developmental circuit formation and causal circuits for synthetic compositional tasks are active prior work. Our current patching and checkpoints are not comparably complete circuit identification. |
| [Kempf et al., CLIP compositional generalization](https://proceedings.mlr.press/v267/kempf25a.html) | Controlled training distributions plus mechanistic analyses of shared representations are not unique to this project. |

The **potentially novel observation** is the tightly controlled six-identity result: changing only observed-relation foil composition, while keeping the queried support and training schedule fixed, appears to switch an entire withheld endpoint class from near-chance to high generalization; exposure controls and tie-release history constrain the interpretation. It is a distinctive synthetic case study, not yet shown to be unprecedented or general. The **potentially novel explanatory principle** would be a predictive rule for when comparison structure selects a reusable relational computation. No such rule has yet survived prospective or cross-architecture testing.

## 7. Single prospective experiment to freeze next

Run **one preregistered, paired three-arm experiment** on untouched newly initialized models (for example, ten fresh seeds in a designated range other than 1704–1706). Use previously untrained foil conditions: target the H pair 0/1 for half the seeds and 0/2 for the other half, assigned before training. These are new conditions relative to F's targeted 1/2 pair.

1. **Relational-comparison arm:** observed-relation H/H same-role foils, using the frozen F-style rule applied to the assigned new pair; no H–H relation is ever queried as an answer.
2. **Exposure arm:** A-style foils with role-wise H marginal exposure matched to arm 1, following the H-control logic.
3. **Difficulty arm:** non-H/H same-role foils selected by a frozen algorithm to match arm 1's pretraining loss/margin and exposure distributions as closely as the generator permits. Feasibility and match tolerances must be audited *before* any new model trains; if adequate matching is impossible, the arm is invalid rather than silently redefined.

Use the same initialization and finite schedule within each seed across arms, identical update count and architecture, and sealed evaluation scenes. Primary endpoint: per-seed mean accuracy across all three **never-queried H–H pairs**. Secondary: mean source/target query contrast on common observed third-identity contexts, plus candidate contrast, score redirection, familiar accuracy, and exact input audit. Freeze the analysis and thresholds before training; do not tune using new outcomes.

**Directional prediction:** arm 1 exceeds both controls on all-H–H accuracy and query contrast; arm 2 stays close to the A/H discovery range. A conservative success gate would require arm 1 minus each control to exceed 0.20 accuracy and 0.40 query contrast on paired seed means, with familiar S/R accuracy above 0.95 in each arm. The pairwise contrasts and multiplicity rule must be preregistered. **Falsification:** arm 1 fails to exceed the exposure arm, or the valid difficulty arm performs comparably to arm 1. Failure of only the all-H–H transfer prediction would specifically falsify the proposed broad scope, even if targeted 1/2 behavior improves. This test confirms a bounded synthetic claim; it cannot by itself prove an architecture-independent acquisition principle.

## 8. Feasibility and decision

**15-credit feasibility: yes, if bounded.** The code, finite schedules, audits, and evaluation readouts already exist. One paired confirmatory cohort, analysis of both encoders at saved milestones, a careful literature comparison, and writing form a finite project. The course thesis should ask *how candidate alternatives in training affect unsupported endpoint behavior and learned computation in this controlled listener*. It should report the surviving difficulty/optimizer alternatives rather than promising a transformer-native grammar. If the prospectively matched difficulty control erases the treatment contrast, the project can still present a negative result on the limits of the comparison hypothesis, but the central positive claim would not be confirmed.

**Paper-level potential: low at present.** The general phenomena overlap substantial prior work, all new F/G/H training is adaptive and mostly single-seed, and the mechanism is not isolated. A paper would likely require a genuinely predictive comparison statistic, independent topologies or architectures, and a causal developmental account beyond margin geometry.

| Judgment | Confidence (subjective) | Reason |
|---|---:|---|
| Real synthetic phenomenon | **0.92** | Large controlled effects, D5 causal anchor, multiple discovery controls; independent F replication remains absent. |
| Viable bounded 15-credit project | **0.78** | Clear finite confirmatory design and existing apparatus; difficulty matching may be hard. |
| Current explanatory account | **0.42** | Comparison pressure explains F/H but not E's soft-selection path or F's global transfer. |
| Paper-level potential now | **0.20** | Literature collision, single-system scope, adaptive selection, incomplete circuit. |

**Final choice: B — STRONG BOUNDED FINDING.** Commit to a preregistered test of the bounded choice-set effect; stop open-ended discovery. Do not claim a general acquisition law unless that later evidence supports one.
