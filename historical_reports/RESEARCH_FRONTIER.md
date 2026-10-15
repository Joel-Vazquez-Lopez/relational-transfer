# Relational role transfer — research frontier

Updated 14 October 2026. This file separates confirmed results, development signals, and questions that remain open. The [Phase 2 development report](role_transfer_phase01/phase2_development_v1/RESULTS.md), [blueprint](role_transfer_phase01/phase2_development_v1/BLUEPRINT.md), and frozen [protocol](role_transfer_phase01/phase2_development_v1/PROTOCOL.json) record the completed comparison.

The comprehensive state, including the LLM and authentic-language branches, is preserved in [RESEARCH_CHECKPOINT_2026-10-14.md](RESEARCH_CHECKPOINT_2026-10-14.md).

## Central question

Does role-aware training reduce a Transformer's dependence on the structure of relational training evidence when it must identify the actor and patient in an unseen entity pair?

The broader paper programme also centres **pretrained LLMs and authentic linguistic roles**. The [parallel LLM/natural-language plan](LLM_NATURAL_NEXT.md) records existing Qwen and QA-SRL evidence, the required label-review gate, and a direct bridge if the graph-specific hypothesis survives. The paper may pivot toward a strong LLM evaluation/repair result if that is what independent evidence supports.

## Validated, bounded findings

- In eight fresh 10-entity graph families, ordinary training was 22.82 percentage points worse on high-overlap than low-overlap unseen pairs (family-level exact p=0.0078125). Ordered-role supervision reduced this *within-graph association* by 19.95 points relative to parameter-matched unordered participant supervision, in all eight families. This association does not identify overlap as the cause. See [fresh overlap results](role_transfer_phase01/rq1_overlap_confirmation_v1/RESULTS.md).
- A separate, prospectively frozen eight-family graph-rewiring study changed a package of local relations. The unordered control's focal high-minus-sham accuracy penalty was −24.45 points; ordered-role supervision attenuated it by +22.59 points (family-level exact p=0.0078125). Other local paths changed with common-partner count. See [rewire repair results](role_transfer_phase01/rq2_rewire_repair_v1/RESULTS.md).
- Familiar task accuracy stayed high in those studies. One earlier run lost 5.76 points of unseen-event accuracy, so the intervention does not uniformly preserve every secondary outcome.

## Failed or inconclusive tests retained

- The initial 12-entity 2×2 factorial suggested a −9.94-point common-partner effect but missed its frozen success criterion. The stricter independent 12-entity factorial estimated the same −9.94 points, but its 95% family interval was −20.46 to +0.58 and Holm-adjusted p=0.1563. The three-edge-path contrast and interaction were not established. The post hoc pool of both cohorts is exploratory. See [strict Phase 1 report](role_transfer_phase01/rq1_factorial12_strict_v1/PHASE1_STRICT_REPORT.md).
- The original proposal's role-swap score-consistency objective was tested against a permutation compute control in Phase 2 and showed no consistent distinct benefit: +0.33-point mean focal attenuation, family interval −5.31 to +5.97, descriptive exact p=0.875. The preflight showed that raw-score weights 0.5 and 0.1 damaged familiar learning at 960 updates; 0.05 passed. See [the Phase 2 report](role_transfer_phase01/phase2_development_v1/RESULTS.md) and [preflight report](role_transfer_phase01/phase2_development_v1/PREFLIGHT_REPORT.md).
- Existing pretrained-LLM question-wording and authentic-sentence findings do not establish the graph-based relational-transfer claim. They remain separate research leads. Human verification is still needed for some natural-sentence labels.

## Completed Phase 2 development experiment

All 160 five-arm models finished on **already examined** eight-family sham/high rewire graphs. Ordered-role versus participant-only attenuation of the focal graph penalty was +22.57 points, positive in all eight families; descriptive family interval +14.24 to +30.91 and exact p=0.0078125. The second rendering suite was similar. Familiar accuracy stayed about 99.9%, and the all-45-pair audit completed. However, ordered-role performance on other unseen pairs was about 10.7 points higher than participant-only at the final checkpoint. The closest eligible performance match still missed the frozen two-point tolerance by 9.73 points. Therefore a **specific reduction in structural dependence beyond general ability improvement remains unestablished**. See [Phase 2 results](role_transfer_phase01/phase2_development_v1/RESULTS.md).

## Completed ability-matched development screen

The [frozen ability-matched screen](role_transfer_phase01/ability_match_v1/RESULTS.md) completed 32 calibration models and 24 further models. A nonzero role-loss weight of 0.001 passed the predeclared aggregate nonfocal ability gate: 85.69% versus participant-only 85.01%, a +0.68-point gap. Yet ordered-role **did not attenuate** the focal high-minus-sham graph penalty at this operating point: the estimated interaction was **−6.10 points**, family interval **−14.70 to +2.49**, descriptive exact p=0.140625, with two of eight families positive. Family-specific ability gaps of −4.22 to +7.57 points limit even the aggregate match. This negative result is development evidence on examined graphs. It does not erase the high-weight Phase 2 absolute-scale interaction, but it gives no support for the stronger claim that attenuation survives ability matching.

## Next decision

1. Do not launch fresh-graph confirmation of the ability-independent interaction on the present evidence. Retain the controlled absolute-scale effect and its ability confound; if revisiting it, preregister a better family-aware comparison or a shared-pair panel on fresh graphs.
2. Keep the original role-swap consistency null and the nonfocal/focal heterogeneity visible. Do not relabel ordered-role supervision as role-swap consistency.
3. Prioritize the paper's pretrained-LLM and authentic-language question. Existing Qwen synthetic experiments and QA-SRL/EWT studies are preliminary foundations. The next decisive LLM experiment must test a distinctive, matched relational-transfer or repair hypothesis and use a frozen, independently verified natural-language evaluation. Do not treat the controlled graph result as an explanation of LLM behavior without direct evidence.

## Candidate-free LLM development screen

The [two-start candidate-free screen](research/candidate_free_role_binding_2026-10-14/RESULTS.md) trained matched typed-role and neutral-cue Qwen2.5-0.5B adapters on natural QA-SRL questions without supplied answer candidates. The typed cue improved both-role exact answers by +20/100 in one start but +8/100 in the other, missing the frozen two-start success rule. It improved answer agreement with printed role fields in both starts, while printed role accuracy did not improve consistently. This supports investigating participant extraction rather than scaling the cue alone. The 100 dev items are actor-first and automatically mapped, so this is exploratory development only.

## Paper claim discipline

Current bounded claim: at the high role-loss weight, ordered-role supervision improves unseen actor/patient transfer and attenuates the measured focal penalty from one controlled rewiring package in a small Transformer. A lower, aggregate ability-matched role-loss weight did not reproduce that attenuation. Unproven claims: a structure-specific improvement *beyond* general transfer ability, a unique common-partner mechanism, a general internal role circuit, improvement in authentic language, or transfer of this learning principle to pretrained LLMs. The main-conference target is conditional on a stronger, independent LLM and real-language result.

## Research-to-paper audit

The [14 October audit](research/paper_audit_2026-10-14/ACL_RESEARCH_STRATEGY.md) reviews the controlled and LLM branches, nearby literature, checkpoint ancestry, and next decision gates. New cached-output diagnostics suggest that typed cues improve answer use while role tuples become more question-dependent in the candidate-free pilot. This is post hoc development evidence, not a causal mechanism. The next priority is a clean pretrained-baseline and natural-language validity check, followed by a controlled extraction/answer-conditioning comparison. The earlier event-bound transformation proposal substantially overlaps an already failed transport pilot and is not ready for execution.