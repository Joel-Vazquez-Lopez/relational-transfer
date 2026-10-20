# Natural-role adaptation and replay controls: exploratory results

**Date:** 12 October 2026. **Model:** local Qwen2.5-0.5B-Instruct. **Status:** two-start exploratory result, not a confirmed general LLM repair.

## Question and design

The earlier participant-only auxiliary training helped the constructed patient-cleft test, but its authentic-sentence role contrast remained modest. I tested whether adding *annotated natural sentences* could improve this transfer. The first two runs combined those examples with replay of the original synthetic task at 0.3 loss weight. A matched follow-up compared natural examples alone and replay alone; the natural-only version was then tested from the second starting adapter. Both starting adapters used different relation graphs, name sets, and random starts. Every natural-data run used the same 256 QA-SRL Bank 2.0 original-training events for 256 updates. The task asked for either the actor or the other direct participant of a target verb and trained the model to output the annotated span.

The 256 training events were selected deterministically from 1,154 candidates with active, non-negated WHO subject and object questions, unique non-overlapping spans of at most eight tokens, and at least two agreeing valid answer judgments per span. The 130-event Bank test was frozen after the first 51-event Gold pilot, before Bank-test inference. Its articles and exact sentences are disjoint from the selected training events, the separate development events, and the previously inspected Gold set. It has 98 source-document groups; 81 of its 130 target verb forms are absent from the 256 training events. The [QA-SRL Bank 2.0 source](https://github.com/uwnlp/qasrl-bank) and [QA-SRL Gold source](https://github.com/plroit/qasrl-gs) describe the corpora.

## Main outcomes: natural examples without replay

“Role contrast” scores the same two answer spans under actor and other-participant questions; an event is correct if the actor becomes relatively more likely under the actor question. It is candidate-relative and does not imply the model can generate a correct answer. “Direct” asks it to generate each span without candidates; exact matching is strict.

| Start | Measure | Starting adapter | Natural-only adaptation |
| --- | --- | ---: | ---: |
| Fresh1 / 830211 | Bank-test role contrast, sentence present | 97/130 | **121/130** |
| Fresh1 / 830211 | Bank-test role contrast, sentence omitted | 78/130 | 81/130 |
| Fresh1 / 830211 | Exact direct actor / other / both | 22 / 30 / 2 of 130 | **78 / 73 / 41** of 130 |
| Fresh3 / 830212 | Bank-test role contrast, sentence present | 93/130 | **121/130** |
| Fresh3 / 830212 | Bank-test role contrast, sentence omitted | 70/130 | 80/130 |
| Fresh3 / 830212 | Exact direct actor / other / both | 24 / 31 / 4 of 130 | **83 / 80 / 49** of 130 |

The paired Bank-test contrast gains are **+24/130 and +28/130** (18.5 and 21.5 percentage points). In the first start, 24 incorrect events became correct and none reversed; in the second, 30 became correct and two reversed. The sentence-omitted changes are smaller, so the improvement is not explained by a stronger question or candidate prior alone. The mean sentence-context contribution to the contrast rose from **1.27 to 3.48** in the first start and **1.57 to 4.18** in the second. Gains occur in both TQA and Wiki1k test material. A document-cluster bootstrap over the 98 shared source groups places the fixed-two-start mean candidate-accuracy gain at roughly **13.6–27.3 points**, but this interval does **not** include uncertainty over training runs or model families. The mixed runs gave the same 121/130 contrast score.

The prior 51-event Gold set also improved from **38 to 48** and **39 to 48** correct, and the separate 75-event Bank development set improved from **55 to 71** in the first start. Gold had already been inspected in earlier experiments, so it is supportive rather than fresh confirmation. Direct generation was added as a post-screen diagnostic; its gains were not part of the frozen directional gate.

## Synthetic retention and ablations

Natural-only kept the active/passive results approximately stable while reducing patient-cleft accuracy moderately:

| Start | Held-out patient clefts | Held-out active + passive |
| --- | ---: | ---: |
| Fresh1 / 830211, starting → natural-only | 618/750 → **595/750** (**−3.07 points**) | 1,488/1,500 → 1,488/1,500 |
| Fresh3 / 830212, starting → natural-only | 697/750 → **672/750** (**−3.33 points**) | 1,446/1,500 → 1,455/1,500 |

Both natural-only starts pass the fixed screen of at least seven extra Bank-test role contrasts and no more than a five-point patient-cleft loss. The cleft score still declines, so the result is a **partial behavioral repair with a measured trade-off**, not improvement on every test.

In the first starting run, matched follow-up controls gave:

| Training after the same source adapter | Fresh Bank role contrast | Patient clefts | Active + passive |
| --- | ---: | ---: | ---: |
| No further training | 97/130 | 618/750 | 1,488/1,500 |
| **Natural examples only** | **121/130** | **595/750** | 1,488/1,500 |
| Natural examples + synthetic replay | **121/130** | 587/750 | 1,499/1,500 |
| Synthetic replay only | 98/130 | 295/750 | 1,500/1,500 |

The replay-only arm did not reproduce the authentic gain and strongly damaged patient-cleft transfer. In the mixed arm, replay did not improve Bank role contrast and was eight cleft items worse than natural-only. On the second start, natural-only also retained more clefts than the mixed arm (672 versus 659/750) while matching its 121/130 Bank contrast. These controls favor the **simpler natural-only continuation** for this pilot. They do not prove replay is always harmful; here it repeats a narrow active/passive task for 256 more updates.

## Interpretation and limits

The new examples substantially improved both candidate-relative role discrimination and exact generated answers on unseen sentences. The sentence-omitted control changed much less, supporting greater use of sentence context. Direct actor/other answers are still jointly exact for only **41/130 and 49/130** events, so the model is far from reliable.

The replay-only control shows that continued training on the narrow synthetic task does not explain the authentic gain. This result still does **not** isolate why natural-role training helped: it combines varied verbs and syntax, role labels, answer-format practice, and continued optimization. It does not establish an advantage for the earlier participant-only auxiliary objective over ordinary initial fine-tuning. Both starts used one 0.5B model family and the same natural training/test material. QA-SRL labels passed the extraction agreement rule but have not been independently reviewed by the student; the “other” participant is not always a semantic patient. Some generated answers are shorter but semantically similar to the annotated span, while some are plainly wrong. Model pretraining exposure is unknown.

The Bank test was chosen after seeing the positive Gold pilot, though its items and pass rule were frozen before either Bank-test inference. Natural-only controls were designed after seeing mixed-run outcomes, so their success on the same Bank items is a recipe screen, not blind confirmation. A stronger paper-level claim needs an independent annotated evaluation set, human label review, a matched natural-language task that does not teach role labels, additional starts/models, and a held-out construction test. The present result supports a concrete next hypothesis: **natural role supervision can increase contextual role use after synthetic fine-tuning, while repeating only narrow synthetic active/passive examples can undermine cleft transfer.**

## Audit trail

- [First training protocol](PROTOCOL.json), [train items](TRAIN.json), [development items](DEV.json), [training trace](mixed_replay/TRAINING.json), [first evaluation](mixed_replay/EVALUATION.json), and [direct-answer controls](EXTRA_source.json), [adapted answers](EXTRA_new.json).
- [Bank-test frozen protocol](fresh_bank_test/PROTOCOL.json), [event list](fresh_bank_test/EVENTS.json), [source scores](fresh_bank_test/source.json), [adapted scores](fresh_bank_test/new.json), [source direct answers](fresh_bank_test/direct_source.json), and [adapted direct answers](fresh_bank_test/direct_new.json).
- [Second-start protocol](replication_fresh3_830212/PROTOCOL.json), [training trace](replication_fresh3_830212/TRAINING.json), [source evaluation](replication_fresh3_830212/EVAL_source.json), [adapted evaluation](replication_fresh3_830212/EVAL_new.json), [source direct answers](replication_fresh3_830212/DIRECT_source.json), and [adapted direct answers](replication_fresh3_830212/DIRECT_new.json).
- [First-start control protocol](controls_fresh1_830211/PROTOCOL.json), [natural-only evaluation](controls_fresh1_830211/natural_only/EVALUATION.json), [Gold check](controls_fresh1_830211/natural_only/GOLD.json), [natural-only direct answers](controls_fresh1_830211/natural_only/DIRECT.json), and [replay-only evaluation](controls_fresh1_830211/replay_only/EVALUATION.json).
- [Second natural-only protocol](natural_only_fresh3_830212/PROTOCOL.json), [evaluation](natural_only_fresh3_830212/EVALUATION.json), and [direct answers](natural_only_fresh3_830212/DIRECT.json).
