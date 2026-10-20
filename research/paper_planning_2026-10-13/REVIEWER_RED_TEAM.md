# Skeptical reviewer assessment

## Provisional verdict

**Promising controlled finding, not yet a safe main-conference claim.** The matched rewire, fresh-family ordered-role comparison, and functional state perturbation are a coherent contribution. The major unresolved issue is whether the rewire isolates a meaningful relational property rather than a bundle of graph changes in one coded task. The LLM experiments do not currently validate the controlled mechanism in natural language. I would review a manuscript favorably only if its title, abstract, figures, and limitations respect those boundaries, and if the new graph and supervision controls survive.

| Objection a strong reviewer may make | Why it matters now | Decisive response or honest limitation |
|---|---|---|
| **“The local-overlap story is confounded.”** | The high-overlap rewire also changes other paths and possibly support conflicts. The earlier eight-graph clustering prediction did not hold. | Prospectively match or factorially vary the relevant graph properties; report full covariate vectors. If isolation is infeasible, claim only a matched *multi-property* rewire effect. Do not retrofit one graph statistic as the cause. |
| **“Equal exposure does not imply equal information.”** | A graph can have the same examples but differ in whether the target relation is identifiable. | Enumerate admissible labelings and provide a rule-based solvability audit. Compare graphs on intended informational support; exclude ambiguous targets by a frozen rule. [Jabbar et al.](https://aclanthology.org/2025.findings-emnlp.1133/) makes this concern especially salient. |
| **“This is a peculiar code-and-scene game, not language.”** | The controlled messages and four-choice scenes are not authentic sentences. | Defend causal control as the first contribution, then add a second architecture or position-separating linguistic rendering. Keep authentic-language and LLM claims conditional on separate evidence. Avoid phrases such as “LLMs do not understand semantic roles” based on this task. |
| **“Your graph families, seeds, and paraphrases are pseudoreplicates.”** | Reusing semantic structure can make item-level or seed-level p-values falsely precise. | Treat graph family as the independent unit; show per-family effects and exact/clustered uncertainty. Renderings and starts are repeated measures. Predeclare the primary contrast and correct or clearly label secondary tests. |
| **“The numbers were selected from many exploratory runs.”** | The project tried several graph measures and interventions, and some failed. | Date and hash the protocol for new confirmation. Show the negative early overlap/clustering tests and all exclusions. Distinguish historical discovery, fresh confirmation, and post-hoc analysis in every figure caption. |
| **“Ordered-role auxiliary just adds supervision or compute.”** | The role order target may be longer, easier, or more informative than the comparator. | Keep participant-only, token/compute-matched sham, and ordinary arms; align examples, updates, weights, and parameter counts. Report both absolute and interaction effects and any event/familiar regressions. One prior auxiliary study already showed participant-only can outperform ordered on a cleft endpoint, so do not claim universal superiority. |
| **“The role probe/patch is an off-manifold artifact.”** | A state swap can perturb any brittle decision boundary. Norm-matched random controls alone may not fully isolate semantic content. | Add same-layer, norm, identity, event, sham, and bidirectional patch controls; report margins and flips. Say the chosen direction functionally influences this decision. Do not call it the model's unique role circuit. |
| **“Position and identity shortcuts were not eliminated.”** | Role tasks can be solved by answer index, first name, scene layout, or token frequencies. The Qwen and natural diagnostics found strong presentation effects. | Balance candidate position, actor order, names, verbs, and filler symbols; run shortcut-only baselines and counterbalanced renderings. If a shortcut accounts for transfer, make that the finding and narrow the mechanism claim. |
| **“The external baselines are weak.”** | Comparing only ordinary training and one auxiliary leaves obvious alternatives. | Test participant-only and token-matched auxiliaries; consider an explicit swap/consistency objective. Explain why direct transplantation of SCAN architectures may be infeasible, and do not imply a reproduced literature benchmark when only citing related work. |
| **“One architecture and one generated distribution are not a conference paper.”** | All central positive results may share the same inductive bias. | The minimum stronger version needs fresh graph-property isolation plus a second architecture or a meaningful linguistic rendering. If these fail, a bounded Findings/workshop empirical result remains possible but main-track breadth is weak. |
| **“The LLM section is a narrative jump.”** | Qwen prompt sensitivity and typed candidate-supported improvements do not establish the same graph mechanism. QA-SRL labels are automatically mapped and unreviewed; candidate-free extraction collapsed. | Make LLM results a separate limitation or second paper. For an LLM claim, obtain independent natural annotations, fresh candidate-free tests, balanced prompts, and more than one model family. A list of prompts is not a causal bridge. |
| **“The authentic-language training result was presented as a repair.”** | The synthetic slot-first gain did not transfer to the natural Gold evaluation; natural-role results still have label and active-position concerns. | Preserve the negative result in the audit and discussion. Do not use automatically mapped QA-SRL labels as verified ground truth. Do not claim that externally routing a correct field proves the model learned routing. |
| **“The novelty claim ignores adjacent literature.”** | COGS, local-structure studies, data-factor work, semantic-role circuits, role-conditioned QA, and backward planning already cover much of the general framing. | Define novelty as the **specific prospective matched relation rewire + ordered-versus-unordered repair + bounded functional state test**, only if replicated. The [related-work matrix](RELATED_WORK_MATRIX.md) records overlaps. “Inverted CoT,” “goal-first,” and “structured roles” alone are not novel. |
| **“A single aggregate gain masks trade-offs.”** | One controlled report included event regression; the natural construction study improved passives while harming actives. | Report each graph, role, construction, and familiar/event endpoint. Predefine a harm tolerance. A method that merely shifts errors is not a general repair. |
| **“The result is practically unimportant.”** | A 4-way artificial scene task may seem remote from deployed LLM evaluation. | Show exactly what common accuracy fails to detect: familiar performance stays high while a held-out relation changes after equal-volume rewire. Treat this as an evaluation-validity principle, not evidence of real-world harm or a ready-made LLM training recipe. |

## Questions the paper must answer on its first two pages

1. What is the independent experimental unit, and how many are there in discovery versus confirmation?
2. What exactly stays fixed in the rewire, and which local graph properties still change?
3. Why is the withheld target identifiable from intended training evidence?
4. What does the ordered auxiliary add beyond predicting both participants or spending more tokens?
5. Which primary result was frozen before inspection, and which figures are exploratory?
6. What does the state patch show behaviorally, and what internal claim does it leave open?
7. Why is the LLM material present if it does not reproduce the controlled causal effect?

## Explicit rejection triggers

- Title or abstract claims a generally learned semantic-role mechanism in LLMs from the controlled task.
- Uses individual sentences, renderings, or seeds as independent replicates of graph effects.
- Calls the graph statistic causal without matching the co-varying paths.
- Calls generated role fields an internal representation, candidate-supported selection open-span extraction, or deterministic external routing an internally learned behavior.
- Uses unreviewed automatic QA-SRL mappings to claim authentic-language improvement.
- Drops the negative clustering, synthetic-to-natural, candidate-free, or construction-trade-off results without disclosure.
- Presents goal-first decomposition or Inverted CoT as new merely because it is named differently from prior backward-chaining/planning work.

## How to strengthen the manuscript with the least work

**First:** regenerate an auditable result table from per-item outputs and freeze new graph contrasts. **Second:** run the local-property isolation and ordered/participant/token-matched comparison on fresh families. **Third:** add a second model or linguistic rendering only if the primary result survives. Keep the LLM repair line as a separate paper program until the natural-language measurement is independently valid. This order prioritizes finding out whether the central claim is true.

## Academic and reproducibility note

These are research-planning notes, not text to submit as a student-authored paper. The student's own course rules govern AI assistance and attribution. A conference submission would need a reproducibility statement covering generator code, graph splits, checkpoints, all seeds, compute, annotation protocol, and the distinction between exploratory and sealed tests.
