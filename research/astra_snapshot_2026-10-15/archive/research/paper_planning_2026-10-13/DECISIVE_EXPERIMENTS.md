# Minimum decisive experiment package

## Decision target

The near-term paper asks whether rearranging *equal amounts* of observed actor–patient evidence changes transfer to a withheld pair, and whether ordered role supervision reduces that vulnerability. The existing rewire and repair studies provide the starting result, but local graph properties still move together. The package below is ordered by **uncertainty removed**, not by apparent sophistication. Do not start a large training run until the data, graph, and analysis audits are frozen.

**Primary hypothesis H1:** a prospectively specified relational-evidence contrast changes withheld-pair role accuracy while entity exposure, pair counts, event frequencies, answer position, and training budget are held fixed. **H2:** ordered actor/patient supervision reduces that contrast more than participant-only supervision under matched training resources. **H3, narrower:** a measured scene-state direction contributes to decisions in this architecture, without implying a unique internal role representation.

**Falsifiable alternatives:** the apparent topology effect is caused by another changed path property or by generator artifacts; the ordered objective works because it adds labels, tokens, or easier optimization rather than role order; the patch works because any state disturbance moves a narrow decision margin; the result is peculiar to one architecture or coded presentation. Predefine what each experiment can and cannot distinguish.

## 0. Freeze the record and validity audit — before training

1. Preserve all prior exploratory/fresh outputs and their status. Publish a machine-readable manifest of graph IDs, seeds, checkpoints, generator versions, hyperparameters, hashes, and every run, including failures. Keep sealed sets sealed; do not inspect them to choose contrasts or prompts. The `ARTIFACT_INVENTORY.tsv` is an index, not an independent reanalysis of every raw score.
2. Recompute every headline number directly from per-item predictions where available. Confirm denominator, chance level, interval construction, and whether each value is accuracy, percentage-point difference, or continuous margin. Resolve any mismatch before drafting figures.
3. Audit label identifiability. Enumerate rule-consistent interpretations of each training graph, including entity aliases and any positional or filler cues. Exclude or separately report graphs where the held-out target is not determined by the intended evidence. Run trivial identity/position/event-only baselines; publish their results even if strong.
4. Freeze graph-family as the inferential unit. Renderings and random starts are repeated measurements within a family, not independent evidence. Save a preregistered analysis script and a short protocol before fresh confirmations.

**Stop rule:** if the task can be solved by a shortcut or the target cannot be identified from intended evidence, repair the generator and create entirely new confirmation graphs; do not relabel old outcomes as confirmatory.

## 1. Isolate the relational contrast — highest-value new result

**Design.** Before training, search the space of valid rewires for pairs of graphs that match degree sequence, observed-pair count, per-entity and per-event exposure, held-out pair, message/scene renderings, and all feasible local path properties except one declared statistic. Candidate statistics include shared partners, directed two-step support, and conflicting role paths. Generate the graph pairs *without looking at model performance*. Report the complete covariate vector and number of acceptable pairs. A pure single-statistic contrast may be impossible under these constraints; check feasibility first rather than asserting isolation.

The [metadata-only path audit](GRAPH_PATH_AUDIT.md) makes one alternative especially urgent: every existing high-overlap rewire removes some three-edge paths between the focal entities. An exhaustive graph-only check found no strict path-and-triangle match within two switches of any of the eight old RQ1 base graphs, so simply rerunning the old generator cannot isolate overlap. Search new bases or allow more switches, or vary path count at fixed overlap. If a one-statistic match exists, choose a prospective contrast and fresh graph families. If it does not, use a small, predeclared factorial set or matched strata that separately vary the most important path properties. In that case the paper's claim remains **multi-property relational arrangement**, not “shared-neighbor overlap causes transfer.” Include sham rewires and unchanged layouts as controls.

**Minimal confirmation.** At least eight *new* independently generated graph families, with three paired initialization seeds per family and the same training schedule across arms. More families help, but increasing seeds on the same family does less for generalization. Freeze graph generator, inclusion rule, primary endpoint, confidence interval, and sign test before opening results. Report family-wise effects and uncertainty, not only the pooled item score. A success is a repeated directional role-accuracy difference on the declared contrast with familiar/event performance maintained; a heterogeneous or vanishing difference sharply limits the paper.

**Presentation controls.** Counterbalance actor position, participant frequency, answer index among four scenes, entity aliases, verbs, and filler symbols. Use at least one independent rendering of the same graph contrast as a robustness check, analyzed as repeated measurement. Evaluate whether shortcuts trained on identities, answer positions, local text patterns, or event presence recover the label. Count semantic trials separately from surface variants.

## 2. Ordered-role repair against the strongest controls

Use the same fresh graph split, architecture, initial weights, training examples, and number of updates for all arms. Compare at least:

| Arm | Why it is needed |
|---|---|
| Ordinary four-scene objective | Direct task baseline. |
| Participant-only auxiliary | Controls extra identity supervision without role order; already the crucial comparator. |
| Ordered actor/patient auxiliary | Candidate repair. |
| Token/compute-matched sham auxiliary | Checks whether added targets, decoding length, or optimization time explain gains. |
| Explicit score-swap consistency objective from the [course proposal](../../proposal_working_draft/project-proposal.tex) | Tests the proposal's actual RQ2 intervention and whether transformation consistency, rather than ordered labels, suffices. |

Choose auxiliary weight only on development graphs and freeze it before confirmation. Match effective optimizer steps and input exposure; record target tokens, gradient contributions, wall time, and parameter counts. If exact token matching changes the task, report both update-matched and token-matched analyses rather than hiding the trade-off. The primary estimate is the **difference in rewire penalty** between ordered and participant-only arms, paired within graph family and seed. The score-swap arm is a separate, prespecified comparison: it could match ordered supervision, fail, or improve on it, each with a different theoretical interpretation. Secondary endpoints: absolute withheld accuracy, familiar accuracy, event accuracy, and variance across graphs. Report regressions as well as gains. A repair claim fails if the apparent benefit is limited to one selected graph, disappears against the token-matched auxiliary, or sacrifices ordinary performance beyond a predeclared tolerance.

**Literature comparator.** The closest conceptual controls are compositional architectures/training that separate alignment or roles, not an invented “state-of-the-art” baseline. Use [Russin et al. (2020)](https://aclanthology.org/2020.acl-srw.42/) as design context; if a compatible implementation is unavailable, state that rather than suggesting a reproduced comparison. Data-factor and local-structure work by [Zhou et al. (2023)](https://aclanthology.org/2023.emnlp-main.898/) and [Bogin et al. (2022)](https://aclanthology.org/2022.emnlp-main.175/) motivate the graph controls. Confirm exact bibliographic details and implementation compatibility before a manuscript comparison.

## 3. Test whether the state effect is functional, without overclaiming

Replicate the role-direction scene-state patch on fresh models using an intervention specified from development data. Compare with same-layer, same-norm random directions; a frequency/identity direction; a role-irrelevant event direction; and a matched no-op/sham patch. Report both decision flips and continuous margin changes, with familiar and withheld pairs. Where possible, patch both directions and check predicted sign reversal. Random controls should be sampled before inspecting effects. Use graph family, not individual patched examples, for uncertainty. If the result is selective, say the measured state **contributes to this decision**. It does not identify a unique circuit, prove a human-like role concept, or establish causation in pretrained LLMs.

## 4. One carefully chosen external-validity test

For a plausible main-conference case, test the same *prediction* beyond the original coded scene task. The smallest useful option is a second architecture with the **same controlled graphs and semantic labels**, or a natural-language rendering where actor/patient and surface position separate (active/passive or relative clauses). The latter is more valuable for the course's LLM theme but requires new confound control. Predefine its primary contrast and use untouched test combinations. A second architecture alone establishes model robustness, not linguistic generality; a template rendering establishes some linguistic variation, not authentic-language transfer.

The existing Qwen and QA-SRL studies can be reported as **separate boundaries**. They do not replicate the graph mechanism. If a pretrained-LLM bridge is pursued, require at least two model families or an explicit one-model scope; natural, reversible events; unseen names/verbs or participant pairs; active/passive and question-wording balance; and equal answer-position exposure. Candidate-supported and candidate-free scoring must remain separate. Freeze exact prompts, decoding, parser, and exclusion rules. Use newly collected/selected items not previously examined in exploratory diagnostics. Manual role validation by at least two independent annotators, blinded to model output, is required for a natural-language performance claim. Track ambiguity, non-agentive verbs, grammatical subject/object versus semantic agent/patient, and adjudication; do not treat automatic QA-SRL mappings as ground truth. If review cannot be done, keep this an exploratory appendix or defer it.

## Analysis and minimum publication gate

- **Primary statistics:** paired graph-family effect of rewire on held-out role accuracy; paired family-level interaction of ordered versus participant-only arm. Give all family effects, bootstrap or randomization interval at family level, and an exact sign/randomization test where appropriate. Multiple question/rendering variants are repeated measures. Adjust or clearly label secondary multiplicity; do not interpret seeds as independent graph replications.
- **Predefined failure cases:** graph covariates cannot be isolated; direction varies substantially across new families; role benefit disappears under matched auxiliary controls; familiar/event regression; shortcut baseline explains outcome; second domain fails. Each yields a narrower, still reportable conclusion.
- **Smallest credible Findings/workshop suite:** validity audit + fresh graph contrast with full covariates + ordered/participant/token-matched repair + family-level analysis + released generators/run manifest. The state patch strengthens a mechanism discussion only if replicated with controls.
- **Stronger main-conference suite:** the above, plus an independent architecture or linguistic task with a prospectively frozen test and a clear theoretical account of *which* relational arrangement matters. A pretrained-LLM/natural-data claim additionally needs human-reviewed labels and cross-model evidence. Do not present this threshold as an acceptance guarantee.

No new expensive training was run for this planning document. The sequence is deliberately gated: first resolve validity and graph isolation, then spend compute on confirmation.
