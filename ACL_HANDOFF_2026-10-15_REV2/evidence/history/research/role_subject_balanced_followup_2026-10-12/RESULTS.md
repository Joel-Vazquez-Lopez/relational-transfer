# Construction coverage does not remove the role-transfer trade-off

**Date:** 12 October 2026. **Status:** completed prospective exploratory follow-up, with second lexical test frozen before mixed-arm training or any new-test inference. The human-review challenge remained sealed.

## Design

The [first matched pilot](../role_subject_decoupling_pilot_2026-10-12/RESULTS.md) showed that replacing subject-cleft training with passive training helped passive and patient-relative role answers but harmed active and subject-cleft answers. The follow-up tested a simple repair: include **active and passive** on every training event, and alternate subject and object clefts across two epochs. Each event therefore receives both clefts once. The mixed arm continued the same natural-role adapter as the earlier arms, with exactly **96 updates and 576 labeled actor/patient examples**. Its **27,744 prompt tokens** equal the earlier decoupled arm; the aligned arm used 27,936. No extra facts, teacher, or offline consolidation phase entered the mixed arm.

A second bank used **six new name pairs and three new verbs**, all disjoint from the training data and the first test. It had 36 directed events, six sentence forms and two fixed role questions, or 432 prompts per checkpoint. All eight source/aligned/decoupled/mixed checkpoints were evaluated without changing prompts or selecting items. All **3,456** outputs were valid candidate names, eliminating the invalid-first-token issue found in the first bank.

## New-test results

Values in the form columns are events with **both actor and patient answers correct**, out of 36. The broad score sums six form columns, out of 216 form-events. The final column is individual-answer accuracy out of 432.

| Start / checkpoint | Active | Passive | Subject cleft | Object cleft | Actor relative | Patient relative | Broad joint | Individual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh1 source | 13 | 5 | 17 | 9 | 12 | 18 | 74 | 276 |
| Fresh1 aligned | **31** | 17 | **31** | 15 | 34 | 21 | 149 | 360 |
| Fresh1 decoupled | 26 | 32 | 26 | **18** | **36** | **27** | 165 | **377** |
| Fresh1 mixed | 30 | **33** | 30 | 14 | **36** | 24 | **167** | 375 |
| Fresh3 source | 10 | 7 | 18 | 14 | 17 | 15 | 81 | 256 |
| Fresh3 aligned | **33** | 14 | 34 | 19 | 31 | 16 | 147 | 358 |
| Fresh3 decoupled | 28 | 32 | 29 | **22** | **36** | **23** | 170 | **382** |
| Fresh3 mixed | 29 | **35** | **36** | 18 | 35 | 20 | **173** | 380 |

The mixed curriculum slightly raises the broad joint score above the better existing arm: **+2/216** and **+3/216** in the two starts. It fails the [frozen positive screen](PROTOCOL.json), which required at least +6/216 in each start and retention within three events of the best arm on active, passive and patient-relative forms. In Fresh3, active joint performance is four events below the aligned arm. The mixed arm also has fewer all-six-form-correct events than the decoupled arm: **10 versus 12/36** in each start. This is a partial redistribution of errors, **not a convincing repair**.

## Repeated interaction across two test banks

The stronger finding is that the aligned-versus-decoupled comparison repeats in the new bank. The table below reports the decoupled arm's change in *both-roles-correct events* against its matched aligned arm. Each cell is out of 36 events. These are four related panels, **not four independent model replications**: the same two source starts are reused across the two disjoint lexical banks.

| Test / start | Active | Passive | Subject cleft | Object cleft | Patient relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| First / Fresh1 | −3 | **+15** | −6 | +6 | +4 |
| First / Fresh3 | −9 | **+28** | −5 | +2 | +4 |
| Second / Fresh1 | −5 | **+15** | −5 | +3 | +6 |
| Second / Fresh3 | −5 | **+18** | −5 | +3 | +7 |

For **every one of the 12 distinct name-pair groups across both tests**, passive both-role correctness increases in both starts. At item level, decoupled training gains 15, 28, 15 and 18 passive events across the four panels, with **zero previously correct passive events lost**. Active correctness moves in the opposite direction: it loses 3, 9, 5 and 5 events, with **zero previously incorrect active events gained**. There are **no invalid outputs on active or passive items in either test bank**, so the central interaction is not an answer-format artifact. Patient-relative gains also repeat, though their first-bank counts are partly affected by invalid `The` outputs.

The [paired-effect figure](figures/tradeoff.pdf) shows the six construction effects for every test/start panel; its [PNG version](figures/tradeoff.png) is convenient for slides.

This supports a bounded causal conclusion: with source checkpoint, labeled events, role questions, updates and approximate token budget matched, **which constructions are used in training strongly changes where the same model transfers role answers**. The positive passive and negative active changes co-occur even on disjoint names and verbs. It does not show that the model learned a reusable semantic-role representation, nor does it prove that passive exposure *per se* rather than another construction-specific cue caused the change. The aligned and decoupled arms substitute one construction, and all evaluations remain constructed language in one Qwen0.5B family. The two question wordings were also used during training, so these tests do not establish robustness to reworded prompts.

## Position in the research program

The previous controlled Transformer experiments showed that the arrangement and supervision of relational evidence matter. The natural QA-SRL continuation improved a separate active-sentence role benchmark, but that dataset did not distinguish subject from actor. These new tests identify a **transfer limit and training-data trade-off** in a pretrained language model. Broad average accuracy and one benchmark's gain are insufficient: the model can improve on passive items while becoming less reliable on active items, even under the same role labels and compute.

Active/passive and cleft role evaluation is established prior work: [Denning et al. (2026)](https://doi.org/10.1162/OPMI.a.365) examine thematic-role representations across those constructions, and [Wilson et al. (2023)](https://aclanthology.org/2023.tacl-1.78/) find a linear-order bias under difficult argument-structure transfer. Work on [semantic-role supervision for generalization](https://aclanthology.org/2020.repl4nlp-1.11/) also warns that augmentation can improve one diagnostic and hurt another. The possible distinctive contribution here is the **matched training-distribution intervention linked to both-role, cross-form evaluation after natural-role adaptation**. A targeted search did not verify that the exact design is absent from the literature; novelty and publication potential remain provisional.

The next step should test an objective that explicitly keeps actor and patient distinct across *both* form and role reversal, with a matched data-only control. It must be evaluated on a **new frozen test**, preferably human-checked natural active/passive material, a second model family, and direct answers as well as candidate-relative scores. Another unstructured augmentation sweep would not answer the central question.

## Audit files

- [Frozen follow-up protocol](PROTOCOL.json), [training and evaluation code](followup.py), [token audit](TRAIN_TOKEN_AUDIT.json), eight `EVAL_*.json` prediction files, two mixed checkpoints and training traces.
- [Within-bank analysis](ANALYSIS.json) and [analysis script](analyze.py); [paired cross-bank analysis](CROSS_TEST_ANALYSIS.json) and [script](cross_test_analysis.py).
- [Figure source](figures/tradeoff.R), [vector PDF](figures/tradeoff.pdf), and [PNG](figures/tradeoff.png).
- [First-bank results](../role_subject_decoupling_pilot_2026-10-12/RESULTS.md), including its invalid-token limitation.
