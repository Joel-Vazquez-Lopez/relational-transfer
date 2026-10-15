# Research result and paper position — 12 October 2026

This note separates completed evidence from a possible paper claim. It is not a draft to submit as the student's own writing.

## What we can now say

**Controlled Transformer:** A matched rewire changed unseen-pair role transfer by 26.33 percentage points across eight graph families, and ordered-role supervision attenuated that penalty by 22.59 points on fresh families. Those are strong *controlled-task* results with their stated graph and architecture limits. See the [evidence map](relational_consolidation_stage_a_2026-10-12/RESEARCH_MAP.md).

**Pretrained Qwen:** Natural QA-SRL continuation improved candidate-relative role discrimination on a separate 130-item Bank test in two starts, but those active sentence labels conflate grammatical subject with actor. A later four-form diagnostic found no fully correct 16-answer events in either adapted start and a shift toward the target clause's subject. See the [Qwen diagnostic](qwen_role_equivariance_diagnostic_2026-10-12/RESULTS.md).

**New matched training result:** On Qwen2.5-0.5B, changing only one of three training constructions from subject cleft to passive, with the same starting adapter, events, role questions, 96 updates and 576 answer targets, produced a repeatable *opposite-direction transfer change*. In two related starts and two lexically disjoint constructed tests, passive both-role-correct events gained **+15, +28, +15, +18 of 36**, while active lost **−3, −9, −5, −5**. All 12 distinct name-pair groups gained passive correctness in both starts; no passive event that was correct under aligned training became incorrect under decoupled training. No active or passive outputs were invalid. Patient-relative gains also repeated, but the first bank's relative-clause first-token format errors limit that endpoint. See the [paired analysis](role_subject_balanced_followup_2026-10-12/CROSS_TEST_ANALYSIS.json) and [figure](role_subject_balanced_followup_2026-10-12/figures/tradeoff.pdf).

**Simple coverage follow-up:** Giving the same budget to active, passive and alternating cleft forms slightly raised the sum of both-role-correct form-events (+2 and +3 of 216 over the better old arm), but failed its frozen +6 criterion and did not dominate on all forms. See the [follow-up report](role_subject_balanced_followup_2026-10-12/RESULTS.md).

## Candidate paper question

**When role-labeled training improves one construction, what determines whether the same actor/patient knowledge transfers to other linguistic forms?**

The supported answer is narrow: **the distribution of constructions used during fine-tuning changes transfer sharply, and a single average role score conceals opposing gains and losses**. The controlled Transformer work motivates why evidence structure matters. The Qwen results show that its specific ordered-role mechanism has not yet been established in a pretrained model.

This is already a coherent **course-paper result** with both positive and negative findings. It is not yet a demonstrated general LLM repair, a verified internal mechanism, or an established publication-level novelty claim.

## Where the possible novelty lies

Active/passive/cleft role tests themselves are established. [Denning et al. (2026)](https://doi.org/10.1162/OPMI.a.365) study thematic-role representations across those forms; [Wilson et al. (2023)](https://aclanthology.org/2023.tacl-1.78/) test argument-structure transfer and find order bias. The possible new angle is a **matched, causal training-distribution contrast after natural-role adaptation**, scored jointly for actor and patient across equivalent and role-reversed events, then checked on two disjoint lexical banks. A targeted search did not find the exact combination, but absence from search results is not proof of novelty. A paper would need a fuller related-work audit before making a first-of-its-kind claim.

## What would make the paper substantially stronger

1. **Confirm on natural language.** The sealed active/passive challenge and EWT audit require blind human review. Until then, do not claim authentic semantic-role transfer. A checked set should contain cases where subject, first mention and semantic actor are separable and should track ambiguous or non-agentive verbs.
2. **Test a distinct model family.** The 1.5B Qwen check is observational and the training interventions are only in Qwen0.5B. A second adapted architecture would test generality.
3. **Test a genuine repair objective.** Hold examples, role labels, updates and token budget fixed, then add an explicitly role-equivariant constraint. Compare it to the matched data-only mixed arm and to a generic-consistency control. The endpoint is both roles correct across held-out forms *without* losing active performance, on a newly frozen test. If simple augmentation works equally well, credit the data rather than the objective.
4. **Mechanism only after behavior.** If a repair survives, test whether a role-related signal generalizes across constructions and whether a controlled activation intervention selectively changes role answers. A decodable signal alone is insufficient.

## Paper structure if no stronger repair appears

1. Why role errors matter, and why familiar or single-form accuracy is insufficient.
2. Controlled relational transfer and the ordered-role intervention, with its limits.
3. Natural-role adaptation and the subject/actor confound in its active training set.
4. Matched aligned-versus-decoupled Qwen experiment, two starts and two lexical banks.
5. Negative mixed-coverage control and answer-format audit.
6. Discussion: evaluation validity, limits on interpreting role training, and a precise role-equivariant repair hypothesis.

The paper's defensible contribution would be a **well-controlled account of a transfer trade-off**, not a claim that we have solved relational understanding. The requested “massive” version requires at least the natural confirmation and a convincing cross-form repair or mechanistic explanation.
