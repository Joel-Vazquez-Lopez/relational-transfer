# Phase 0–1 evidence ledger

Date: 6 October 2026. All results here are developmental analyses of existing models or generated evaluation scenes. This document does not change the university proposal.

Protocol chronology: the all-pair factors and seed 260107 were written before that evaluation. The saved-trajectory analysis used pre-existing checkpoints. After the first all-pair result, I added the second scene seed 260108 as an adaptive reliability check. The adapter and decoder pilots were then specified and run in sequence. `PROTOCOL.json` is an evolving local work record, not an external preregistration. No part of this sequence should be described as a new independent confirmation of a mechanism.

## Established by the pre-existing record

- The historical selected-pair screen trained 30 small two-tower Transformers on five 10-node, degree-4 graph layouts, three seeds, and two relabeled conditions. Five selected-pair runs showed roughly 50% role-direction performance with high familiar accuracy. Source: `../d5_indirect_support/RESULTS_20261006.md` and `AUDIT_COMPONENT_RESULTS_20261006.md`.
- The earlier 12-model follow-up passed familiar learning but its selected pair was easy, and its frozen internal predictor failed the stipulated 5/6 sign test. Source: `../d5_indirect_support/RESULTS_20261006.md`.
- Fixing either side to explicit role slots restored selected failures in a small, outcome-selected component study. It supplied structure and changed optimization. It did not show that the original failing model's representations were sufficient. Source: `../d5_indirect_support/AUDIT_COMPONENT_RESULTS_20261006.md`.
- Previous ridge probes underfit familiar data, especially the scene side. Their negative results do not demonstrate absent role information. Source: `../d5_indirect_support/results/probe_20261006.json`.
- Older Track A/Track B studies used different worlds and interventions. Their developmental results motivate possible explanations but do not confirm a mechanism here.

## Phase 0 checks performed here

The new `phase0_audit.json` verified all **42 checkpoint hashes**, eight pairwise nonisomorphic connected degree-4 graphs, 20 observed unordered pairs per graph, 25 nonedges, 1,280 training configurations, and 80 distinct ordered-pair/event meanings per condition. Both directions, both events, all four selected semantic foils, and all four answer positions have equal counts in the training pool. The withheld labeled pair `(0,1)` is a nonedge in each condition. No historical checkpoint was overwritten.

The historical generator has four interpretation limits: the fourth coded-message token shares entity 0's embedding; training distractor identity depends on serial/order; moving source versus nearly stationary target makes scene direction perceptually identifiable; and graph-condition relabeling changes which learned identity symbol occupies each graph node. New training should use the corrected protocol in the plan, but old and corrected experiments must be kept separate.

The training-free task sanity check (`task_baselines.json`) evaluated 23,040 questions across all 45 pairs under one crossed scene suite. First-position choice scored 25.0%; event-only and role-only heuristics each scored about 50%; a simple rule using query identity, relative movement, and event direction scored 100%. The task's intended answer is recoverable in these generated scenes. This rule also reveals that the model does not need a rich visual interpretation to solve the controlled scene half.

## Phase 1A: all-pair map

`all_pair_results.json` evaluated every observed and unobserved unordered pair in each of the 42 saved models, using 512 crossed questions per pair. Thus each model has 25 unseen semantic pairs × 512 scene/question renderings, not 12,800 independent relations. The fixed evaluation suite used seed 260107 and the same scene generation across models.

There are **1,050 unseen model–pair cases** and **840 observed model–pair cases**. Among unseen cases, **337** had binary role accuracy below 80%; every model had at least one such pair. Across all models, mean binary role accuracy on unseen pairs was **85.0%**, while observed-pair binary role accuracy was **99.98%**. The old 12-model follow-up had 82 weak cases among 300 unseen cases even though none of its selected `(0,1)` pairs was weak. Across the old 30-model screen there were 255 weak cases among 750. These proportions describe this finite set of development checkpoints; models and pairs share graphs and training histories, so they are not independent population draws.

Mean unseen-pair event accuracy was 98.5% in the screen and 99.1% in the historical follow-up. The chief weakness is role direction, while some event errors remain. Familiar four-choice accuracy was about 99.9% in each cohort.

`all_pair_replication.json` regenerated the same crossed semantic cases with evaluation seed 260108. This adaptive reliability check yielded 330 weak cases, of which 326 were weak in both suites. Mean unseen-pair binary role accuracy changed from 85.03% to 85.12%; the pair-level role correlation was 0.996. Fifteen model–pair cases crossed the 80% threshold between suites. The result is robust to scene rendering, but it is not a new independent training-world confirmation.

The original controlled focal-pair audit agrees with the new all-pair suite: across 42 models the mean absolute four-choice accuracy difference on `(0,1)` is 0.59 percentage points; the largest is 5.18 points on a model that remains weak in both. Thus the expanded suite did not produce a contradictory focal result.

The weak-pair pattern varies substantially across training seeds and graph relabelings: mean correlation of pair-level accuracies across seeds within a condition was 0.316; after mapping to canonical graph nodes, mean correlation across the two conditions was 0.320 (`pair_stability.json`). This resists a simple claim that graph location alone determines failure. A fuller predictor analysis would need a frozen rule and new models.

## Phase 1B: saved training trajectories

`trajectory_results.json` evaluates steps 0, 1,000, 3,000 and 6,000 from four existing selected `new2` original-component replays. The step-1,000 model in one case had only 50.8% familiar four-choice accuracy, so it cannot anchor an early-stopping comparison. In the other cases, familiar accuracy was already about 99% or more by step 1,000. From step 3,000 to 6,000, all-pair unseen role accuracy was nearly flat or improved in the four models. No unseen pair lost 20 percentage points over that interval. These trajectories do **not** support broad late forgetting in this world. They are from one selected graph and two seeds, not a general training-dynamics result.

## Phase 1C: frozen interface and role readouts

The rank-4 frozen-interface adapter pilot used only observed-pair training and validation examples. The five historical weak focal cases improved by at most about one percentage point in binary role accuracy. One adapter selected its identity initialization. All-pair means also changed negligibly. **The prespecified development advancement rule of three rescues by 20 points failed: zero of five passed.** This rules out this specific low-rank, familiar-supervised adapter as a useful repair under the tested settings. It does not rule out other interface methods or a different training signal.

The role-decoder pilot compared linear and one-hidden-layer decoders on those five models and five matched opposite-condition models. Probes were trained only on observed meanings, with an observed-meaning validation split. Across the 10 models, exact recovery of both roles on unseen meanings was low for both query and scene representations. The operational check failed: at the validation-selected step, the linear query decoder recovered both roles on only 79.1% of its own familiar training meanings on average; the linear scene decoder reached 45.7%. The nonlinear decoders did not improve that fit under observed-meaning validation. Event decoding was much easier. Crucially, the same poor role-readout pattern appeared in controls with high native focal-pair performance. **These probes are inconclusive about whether role information is absent or available to another readout.**

Neither pilot identifies a causal internal mechanism. The adapter adds new observed-pair exposure and lacked a matched full-model continuation control because it failed the initial advancement criterion. The decoders use explicit role labels that the native model did not receive as separate supervision.

## Interpretation and next decision

The all-pair map is the most consequential new result: the selected-pair follow-up appeared easy because that focal relation did not represent the model's other unseen-pair behavior. The phenomenon is broader across this saved set than the original selected-pair account implied. Its cause is still unknown. Pair-level graph structure, identity initialization, optimization, and matching may all contribute. The tested simple readout repair did not help. The next decisive work is a controlled change in learning experience or objective, with fresh graphs left untouched until its method and comparison are fixed.

No current result demonstrates the same mechanism in natural language or pretrained LLMs. The controlled inputs are coded source/target/event messages and generated motion scenes.
