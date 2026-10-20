# Recommended near-term paper

## Decision

Build the first conference-oriented manuscript around the **controlled Transformer relation-topology result**. It has two prospective eight-family interventions, matched controls, a hierarchy of outcomes and a functional state test. Its claim can be bounded to the experimental world without inventing a general LLM mechanism. It is the strongest paper *we can presently substantiate*, although a main-conference submission still needs the decisive additions below. Treat the LLM extraction/routing work as a separate, ambitious paper direction. Its current candidate-supported typed-cue result is useful but fails candidate-free extraction.

### Provisional title

**When Indirect Relations Mislead Role Transfer: Matched Rewiring and Ordered Supervision in Transformers**

The title should change if the next local-structure isolation shows that shared-partner overlap itself is not the central explanatory variable. The current title does not assert that it is.

### Main question and provisional claim

**Question:** With entity exposure and training volume held fixed, can the arrangement of observed actor–patient relations change interpretation of one unseen pair, and can explicit role-order supervision attenuate that effect?

**Supported now, within the controlled task:** matched local rewiring changed withheld-pair role accuracy by −26.33 points across eight graph families; on eight fresh families, ordered-role auxiliary training reduced an analogous penalty by +22.59 points versus an unordered-participant auxiliary. A role-direction scene-state swap caused more decision-margin loss in the ordered models than norm-matched random swaps. These are **three linked findings in one generated world and architecture**, not three independent language domains.

**Still unproven:** shared partners alone cause the loss; ordered training teaches a portable semantic-role algorithm; the scene-state effect identifies a unique circuit; the same mechanism occurs in pretrained LLMs. A [new graph-only check](GRAPH_PATH_AUDIT.md) shows that three-edge paths also decline in all 16 relevant rewires. The old two-switch neighborhood has no graph satisfying the strict path-and-triangle match for any of its eight fixed bases, so the next experiment needs new graph selection or a predeclared relaxed/factorial design.

## Provisional abstract for research planning

Models can identify every familiar actor–patient relation yet fail when familiar entities appear together for the first time. We study whether the *arrangement* of indirect training relations contributes to this gap. In a controlled two-tower Transformer task, we hold entity exposure and training volume fixed while rewiring the relations seen during learning. A matched local rewire lowers role-direction accuracy for the same unseen pair by 26.33 percentage points across eight graph families, despite high familiar-example accuracy. On eight fresh families, auxiliary prediction of **ordered** actor and patient identities attenuates the rewiring penalty by 22.59 points relative to a parameter-matched objective that predicts participant identities without their roles. A state intervention based on role-related scene directions changes withheld-pair decisions more than equally sized random-direction interventions. The results show that equal amounts of evidence need not support equal relational transfer and that role-order supervision can reduce this controlled vulnerability. They do not establish a unique graph cause or a general mechanism in natural language. We close by stating the additional linguistic tests needed to determine whether the effect extends to pretrained language models.

This abstract is **provisional planning text**, not a claim that the full manuscript or independent linguistic validation is complete. The student's own final writing and course AI-use rules govern any submission.

## Three precise contributions

1. **Controlled exposure intervention.** A degree-preserving, sham-matched graph rewire of the same unseen pair produces a large, reproducible role-transfer change while familiar accuracy stays high. This advances beyond a correlation between training data and instance difficulty; it does not isolate one local graph statistic.
2. **Matched repair test.** Ordered actor/patient auxiliary supervision attenuates the rewire penalty on new graph families compared with an auxiliary that identifies the same participants without role order, at matched architecture and data schedule. This identifies a useful training signal in the controlled architecture, not yet an LLM recipe.
3. **Functional state check with limits.** A role-related scene-state perturbation affects decisions more than same-norm random perturbations, and the ordered-versus-unordered contrast repeats across graph families. This is evidence that the measured component contributes to decisions, not evidence of a unique role circuit.

## Proposed figure and table structure

1. **Figure 1 — task and intervention.** One concrete actor/patient example, four scene choices, observed graph and fixed withheld pair; show base, sham, and high-overlap rewires with changed edges marked. Include what counts as a semantic trial versus repeated rendering.
2. **Table 1 — audit and controls.** Graph count, degree, observed pairs, withheld pairs, training examples/updates, balancing of order/event/answer position, matched and unmatched local graph properties, solvability baseline, and familiar-accuracy guardrail.
3. **Figure 2 — primary causal contrast.** One line per graph family for high-minus-sham role accuracy, with eight-family mean/interval; show second rendering as a reliability panel, not a second independent sample.
4. **Figure 3 — repair interaction.** Unordered versus ordered accuracy in sham/high conditions, paired family effects, familiar/event guardrails including the one 5.76-point event regression.
5. **Figure 4 — state intervention.** Role-swap versus energy-matched random-swap margin losses by graph; alongside modest readout accuracy to prevent overinterpretation.
6. **Table 2 — negative and alternative explanations.** Historical failed shared-neighbor sign, failed early predictor, failed low-rank adapter, non-isolated local paths, plus any new graph-property ablation. This keeps the paper honest and narrows the contribution.
7. **Optional bounded LLM table.** Show unmodified Qwen wording sensitivity and the failed synthetic-to-natural transfer/typed candidate-free boundary. Label as a separate motivation/limitation, **not** a replication of graph mechanism. If page budget is tight, move it to discussion or an appendix.

## ACL/EMNLP-style outline

1. **Introduction:** why a correct familiar answer does not establish unseen role transfer; one concrete failure; exact research question and bounded contributions.
2. **Related work:** compositional benchmark fairness, local structures/data factors, role supervision and semantic roles, causal state analysis; distinguish the graph rewire from Bogin et al. and circuit claim from Kumon/Aljaafari.
3. **Task and validity:** world, graph, model, scene choice, withheld pair, balancing and rule-based solvability; limitations of movement cues and generated language.
4. **Experiment 1 — relation topology:** graph selection, base/sham/high rewiring, endpoints, graph-family inference, results and alternative paths.
5. **Experiment 2 — ordered supervision:** matched unordered control, frozen fresh families, primary interaction, event/familiar trade-offs, individual failures.
6. **Experiment 3 — functional state test:** probe training only on observed pairs, unseen test, role-direction patch and norm-matched random controls, interpretation limits.
7. **Further controls / boundary:** new local-structure isolation and second architecture or rendering; optional clearly separate pretrained-LM evaluation boundary.
8. **Discussion and conclusion:** what the graph study establishes; what it cannot imply about human language/LLMs; negative results; next tests.
9. **Reproducibility and ethics appendix:** exact graphs, seeds, generators, training budgets, hashes, all runs, statistic units, known generator artifacts and rerun instructions.

## Publication threshold and split decision

For **Findings or a specialist workshop**, the current controlled record is a serious starting point, but I would still require the graph-property ablation and a clean reproducibility package. For **ACL/EMNLP main**, add a second architecture or a linguistically meaningful task in which role and position are separable, with a prospectively frozen transfer test. No acceptance probability is defensible now.

Do **not** merge Inverted CoT into this paper. Its v4/v5 result is a narrow operation-selector feasibility test and its broader claim overlaps extensively with existing backward-planning literature. The LLM extraction/routing paper is potentially more impactful for the course's LLM theme, but it requires candidate-free, independently reviewed natural data and cross-family repair. If those pass, it deserves a second manuscript with its own question rather than being used to retroactively explain the controlled state patch.

## How this fits the submitted course proposal

The current [proposal](../../proposal_working_draft/project-proposal.tex) asks three related questions, but the completed experiments do not match every promised method exactly:

| Proposal question | What is already supported | What remains |
|---|---|---|
| **RQ1, training-relation pattern and an unseen pair** | The matched rewire and fresh-family replication directly address this. | Identify whether shared partners, lost three-edge paths, or another aspect of the rewire drives the effect; test another presentation or architecture. |
| **RQ2, role-swap consistency training** | Ordered actor/patient auxiliary supervision demonstrably attenuates the rewire penalty versus unordered participant supervision. | The proposal's *score-swap consistency rule* is a **different objective and has not yet been tested**. Compare it directly with ordered roles and matched auxiliary controls. Do not silently describe the ordered result as a test of that rule. |
| **RQ3, pretrained-model stability across sentence and question forms** | The Qwen2.5-1.5B-Instruct constructed-item study found strong wording sensitivity on one actor question. | Independently validated natural sentences, more than one model family, and candidate-free tests are needed for a broader LLM contribution. Current 0.5B repair studies do not establish general natural-language transfer. |

### Ambitious hypothesis worth testing, not yet a result

The paper could grow beyond “a graph pattern affects accuracy” by asking **which learning signal makes actor/patient binding stable** when either the *available evidence* or the *linguistic form* changes. Two candidate signals are explicit ordered-role labels and a transformation-consistency constraint. For a role reversal, the correct answer should swap; for a meaning-preserving active/passive transformation, it should stay fixed. This provides a clean comparison of correct answers, wrong-but-consistent answers, and brittleness to presentation. It would be a substantial contribution only if a prospectively frozen experiment shows transfer beyond the training templates, against direct-answer, participant-only, and matched consistency baselines. The idea of invariance/equivariance is not itself novel.

**Practical sequence:** (1) resolve the graph-path confound and keep the controlled result defensible; (2) test the proposal's exact swap-consistency objective against ordered-role supervision in that task; (3) run a small development-only ceiling check for a pretrained-language bridge; (4) if there is room for improvement, freeze a candidate-free, human-reviewed, construction-balanced LLM test and then compare objectives. If pretrained models already solve the bridge or the intervention fails, publish the controlled result within its scope and keep RQ3 as a bounded evaluation finding. This avoids making the course project depend on an unproven LLM repair.
