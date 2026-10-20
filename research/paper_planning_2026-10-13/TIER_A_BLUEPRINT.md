# Blueprint for a selective main-conference paper

**Status: research strategy, not a claim of acceptance or completed experiments.** “Tier A” here means a serious ACL/EMNLP main-track attempt or comparable selective venue. The course project remains viable if only the controlled core succeeds. No new model training was run for this blueprint.

## The paper worth trying to earn

### Working title

**When Does Role Supervision Make Generalization Less Dependent on Training Relations?**

The title is a question because the stronger cross-setting answer is not yet known. The current controlled result already supports a bounded answer for one rewiring package; the paper must establish the relevant structural variable and scope.

### Central scientific claim, conditional

With the same entities, examples, and training budget, the organization of indirect role evidence can change transfer to an unseen pair. A role-aware learning objective may change the model's **sensitivity curve** to that organization, rather than merely raising its mean accuracy. This is the real candidate contribution. It becomes a stronger main-conference claim only if the interaction survives graph-property isolation, performance-matched controls, and an independent task or architecture.

The strongest presently supported version is narrower: a degree-preserving rewiring package lowered focal role accuracy by 26.33 points across eight graph families; on eight fresh families, ordered-role supervision attenuated a similar package's penalty by 22.59 points relative to unordered participant supervision. The measured scene-state intervention has a selective effect in the same architecture. These results are in a generated world, and the exact responsible graph property is unresolved.

## Why this could stand out, and what is already taken

The possible originality is a **causal interaction** between training-example relational structure and a learning objective on a fixed withheld relation, followed by a test of what in the model's computation changes. It is not the existence of compositional failures, local-structure difficulty, graph-based text reasoning, or paraphrase consistency. Those themes are established by [COGS](https://aclanthology.org/2020.emnlp-main.731/), [Bogin et al.](https://aclanthology.org/2022.emnlp-main.175/), [CLUTRR](https://aclanthology.org/D19-1458/), [Fu and Frank](https://aclanthology.org/2024.findings-emnlp.492/), and [ParaRel](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00410/107384/Measuring-and-Improving-Consistency-in-Pretrained). A manuscript must compare experiments and claims directly with these papers, not just cite them.

The new [graph-path audit](GRAPH_PATH_AUDIT.md) sharpens the challenge: every historical high-overlap rewire also removes three-edge paths. No strict path-and-triangle match exists within two edge switches of the eight fixed old bases. A statement that “shared partners cause failure” is therefore currently unsupported. A paper that resolves this confound would be more valuable than another large run of the original contrast.

## Three experiments that form one argument

### 1. Identify the causal evidence variable

Construct fresh graph families *without model outcomes*. Seek pairs that vary shared partners while matching three-edge paths, and pairs that vary paths at fixed shared-partner count. Match node degree, pair count, event exposure, held-out pair, total examples, answer positions, and as many triangle/edit properties as feasible. If strict matching is too rare, declare a 2×2 or stratified design before training and treat remaining focal-triangle differences as rival explanations. The current [feasibility search](GRAPH_PATH_AUDIT.md) found a strict matched cell but not a complete four-cell design in the sampled 10-node space; matching only total triangle count permits factorial cells. Do not announce a single-feature manipulation until graph-only enumeration and validity checks confirm it.

Primary outcome: withheld-pair actor/patient accuracy, averaged within initialization and graph family. Analyze the structural contrast **within family** and across at least eight independent families. Report all family effects and uncertainty. Familiar/event accuracy must clear predeclared guardrails. A rule-based identifiability audit must show the target is learnable from intended evidence, rather than a contradictory or shortcut-laden dataset.

**Interpretation matrix:** If shared partners matter when paths are held fixed, the overlap claim gains support. If paths matter at fixed overlap, revise the story toward distinct indirect routes. If neither isolated contrast reproduces the old effect, the causal unit is likely the broader rewiring package or an interaction; say so. A mixed result can still be a useful paper if designed prospectively and explained.

### 2. Show that the objective changes structural dependence

Use the same fresh graph conditions, paired starts, and data for ordinary training, unordered-participant auxiliary, ordered actor/patient auxiliary, and the proposal's *score-swap consistency objective*. Add a token/compute-matched sham auxiliary and, crucially, a **performance-matched control** tuned on development graphs to reach similar average accuracy without ordered role labels. Otherwise, any method approaching 100% will mechanically appear less graph-sensitive. Choose a harder, predeclared operating regime if the ordered model is at ceiling; maintain familiar accuracy and avoid selecting test graphs by how dramatic the interaction looks.

Primary estimand: the **graph-condition × training-objective interaction** on fresh families. Show absolute accuracy, error rate and a bounded-scale or log-odds sensitivity analysis; do not hide a regression in familiar/event performance. The most convincing result is not merely “ordered wins,” but “at comparable overall performance, ordered or swap-consistency training has a flatter structural risk curve on previously unseen graph families.” The direct ordered-versus-score-swap comparison answers the course proposal's RQ2 more faithfully than relabeling one as the other.

The score-swap rule should be tested as a genuinely different training signal: swapping actor/patient in the message and each scene permutes the four scene scores accordingly. Holding the examples fixed tests a constraint on predictions rather than extra reversed examples. It may succeed, fail, or complement ordered labels; each result distinguishes a learning principle.

### 3. Establish a boundary beyond the coded scene task

**Minimum main-track external validity:** replicate the prospectively declared *interaction* in a second architecture or a linguistically meaningful rendering that separates role from surface position. A second architecture alone tests robustness to implementation; active/passive/relative-clause renderings test linguistic variation. Neither alone proves authentic-language LLM transfer.

**Ambitious LLM bridge:** test whether an open-weight pretrained model is sensitive to the *structure of additional relational training evidence*. Pretraining cannot be rewired, so this is a **fine-tuning sensitivity** claim, not a claim about how pretraining acquired roles. A development-only ceiling check must come first: ordinary English active sentences may already make the withheld-pair question trivial, in which case a positive or null graph contrast is uninformative. If a nontrivial learnable setting exists, freeze fresh entities, held-out pairs, events and constructions, and compare structurally matched fine-tuning sets and objectives with candidate-free exact role extraction. Include meaning-preserving active/passive changes, role-reversing swaps, and multiple question wordings. Score all variants against gold roles, not agreement alone; a model that repeats one name is consistently wrong on a role reversal. Evaluate with independently reviewed natural sentences as a separate validity check. Use two model families if resources permit; if only one is possible, scope the conclusion to that family.

The previous candidate-supported typed-cue and Qwen wording results justify testing this bridge, but do not demonstrate it. The former collapses without candidates and the latter has prompt-specific ambiguity. API-only frontier models can supply a **read-only external evaluation**, not evidence that custom role training works in those weights.

## What a selective reviewer must see

1. **A clean estimand:** one predeclared graph property or a transparent factorial package, with the independent graph family as the statistical unit.
2. **A real interaction:** structural sensitivity differs between objectives on new graph families, not only a higher average score or a ceiling artifact.
3. **Matched rivals:** participant-only, token/compute-matched, performance-matched, and explicit consistency training.
4. **Mechanism restraint:** state patches and probes show functional contribution under controls; they do not identify a unique semantic-role circuit. Identity, event, random-direction and sham interventions are needed.
5. **External validity:** at least a second model/task; an LLM claim needs a task where pretraining does not trivially solve held-out pairs, fresh candidate-free tests, reviewed labels for authentic sentences, and source-shift analysis.
6. **Reproducibility:** generator, all graph manifests, exact prompts, all starts, raw predictions, seeds, compute, exclusions, and exploratory/confirmatory chronology. Negative results belong in the paper.

## Work order and stop/go gates

| Stage | Smallest next action | Continue when | If it fails |
|---|---|---|---|
| **G0: graph feasibility** | Finish graph-only enumeration/matching and solvability audit. | Fresh independent families exist for a defensible contrast. | Broaden graph size or declare a multi-property rewire; do not call overlap the cause. |
| **G1: controlled explanation** | Freeze and run the isolated/factorial contrast with ordinary training. | The effect is reproducible at graph-family level without shortcut or identifiability failure. | Publish the bounded historical rewire finding; stop the single-feature explanation. |
| **G2: objective interaction** | Compare ordered, unordered, score-swap and matched controls on the same fresh graphs. | Role-aware benefit survives performance matching and non-ceiling evaluation. | Interpret prior attenuation as improvement/optimization, not changed structural dependence. |
| **G3: generalization** | One second architecture or linguistic task, then LLM development ceiling check. | A comparable interaction survives a new setting. | Keep the controlled paper scoped; split RQ3 into a separate LLM evaluation paper. |
| **G4: LLM claim** | Freeze candidate-free and human-reviewed tests, compare open-weight model families. | Correct-answer gains survive syntax, question and source shifts without candidate leakage. | Report an informative boundary; do not manufacture a unified mechanism. |

The course paper needs G0–G2 and a bounded RQ3 check. A convincing main-track attempt likely needs G3 and, for a paper explicitly claiming LLM training improvement, G4. The precise threshold depends on the quality and explanatory sharpness of the results; no venue outcome can be guaranteed.
