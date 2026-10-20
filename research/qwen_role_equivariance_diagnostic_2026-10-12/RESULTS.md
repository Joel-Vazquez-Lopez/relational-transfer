# Role transfer has a construction-dependent answer bias: exploratory diagnostic

**Date:** 12 October 2026. **Status:** real completed experiments on newly constructed items; exploratory, not human-verified natural-language confirmation or a publication-level novelty claim. The 0.5B test protocol was frozen before these inferences. The 1.5B supplement was specified after the 0.5B outputs were inspected and before the 1.5B inference.

## Question

Earlier natural-role adaptation of Qwen2.5-0.5B improved actor-versus-other-participant discrimination on a separate 130-item QA-SRL Bank test from **97/130 to 121/130** and **93/130 to 121/130** in two matched starts. This test asks whether that gain carries over to a stricter capability: give the right **actor and patient** when the same event is phrased in active, passive, object-cleft, and subject-cleft form, with two question wordings for each role and both directions of every participant pair. Those Bank results used active subject/object proxies and unreviewed corpus labels, so this is a meaningful additional challenge, not a direct replication of the Bank score.

The fixed set contains 6 name pairs × 3 verbs × 2 event directions = **36 events**. Each has 4 forms × 4 questions = **576 prompts per checkpoint**. Candidate order is fixed within each name pair; event direction reverses, preventing candidate order from systematically identifying the actor. All six checkpoints gave one of the two candidate names on every prompt. The 0.5B comparison uses untouched Qwen plus two matched source/natural-adapted checkpoint pairs. The supplementary 1.5B model is untouched and has no matched intervention.

## Primary outcome

The frozen primary event measure required **all 16 answers for an event** to be correct. A less strict, explicitly *post hoc* measure uses one comparatively direct actor question and one verb-specific patient question in all four forms (8 answers per event).

| Checkpoint | Individual correct / 576 | All 16 correct / 36 | Explicit 8 correct / 36 |
| --- | ---: | ---: | ---: |
| Untouched Qwen2.5-0.5B | 322 | 0 | 0 |
| Start 1, source adapter | 303 | 0 | 1 |
| Start 1, natural-role adapter | 303 | 0 | 2 |
| Start 3, source adapter | 287 | 0 | 1 |
| Start 3, natural-role adapter | 293 | 0 | 1 |
| Untouched Qwen2.5-1.5B | 369 | 0 | 2 |

The predeclared positive screen for the 0.5B natural intervention required at least **four additional fully correct events in each matched start** and no more than two fewer active-form correct prompts. Neither start meets it: both remain 0/36, while active-form correctness falls **77→71/144** and **80→69/144**. Thus these checkpoints provide **no evidence of robust role-equivariant repair** on the constructed test. The all-16 measure is demanding, but the 8-answer check also remains at just 1–2/36, so this conclusion does not depend solely on the second pair of wordings.

## A sharper diagnostic: a shift toward the local subject

Natural adaptation improved some cells and harmed others. Actor and patient scores below are each out of 72 prompts per form, combining their two respective questions.

| Form and role | Start 1 source → natural | Start 3 source → natural |
| --- | ---: | ---: |
| Active actor | 53→53 | 53→57 |
| Active patient | **24→18** | **27→12** |
| Passive actor | 53→48 | 59→56 |
| Passive patient | **27→39** | **16→35** |
| Object-cleft actor | 31→38 | 27→32 |
| Object-cleft patient | 37→33 | 32→29 |
| Subject-cleft actor | 56→55 | 54→54 |
| Subject-cleft patient | 22→19 | 19→18 |

These changes align with an increased preference to answer with the **subject of the target verb's clause**. That subject is the actor in active and both cleft forms, but the patient in the passive. Counting which participant the checkpoint actually returned, regardless of question:

| Local-subject answers / 144 prompts per form | Start 1 source → natural | Start 3 source → natural |
| --- | ---: | ---: |
| Active | 101→107 | 98→117 |
| Passive | 46→63 | 29→51 |
| Object cleft | 66→77 | 67→75 |
| Subject cleft | 106→108 | 107→108 |

The shift is nonnegative in all **six name-pair groups** for start 1 (positive in five) and positive in all six for start 3. The object cleft helps distinguish this from a universal *first-mentioned person* bias: its local subject is mentioned second, yet local-subject answers also rise there. The 256 natural adaptation examples were selected from **active** subject/object QA-SRL cases, where subject and actor coincide. This is a concrete training-data confound: improved performance on other active items could reflect stronger subject selection. The observed change is a **behavioral pattern consistent with that shortcut**, not proof of what representation the model learned or of the causal mechanism.

## The 1.5B check finds a different shortcut

Untouched Qwen2.5-1.5B produces 369/576 correct individual answers (64.1%), more than any 0.5B checkpoint here, but also 0/36 fully correct events. It returns the **second person mentioned in the sentence on 415/576 prompts (72.0%)**. This is not simply choosing the second name in the candidate list: the latter happens on 235/576 prompts, and event direction is balanced. The second-mentioned rate varies by form: active 97/144, passive 137/144, object cleft 95/144, subject cleft 86/144.

That bias has different consequences depending on the question. In active sentences, the model answers the verb-specific patient question correctly on **35/36** events; in passive sentences, it does so on only **5/36**. Yet it answers the active actor question “Who performed the action?” correctly on **29/36** and the passive version on **34/36**. A single overall accuracy would hide this strong construction-by-question interaction. Candidate-logit contrasts between the explicit actor and patient questions are positive for all 36 active events and all 36 subject-cleft events, despite many absolute answer errors. For object clefts, the same contrast is positive on only 14/36. This separates a response-preference problem in the easier forms from a likely construction-mapping problem in object clefts, though neither mechanism is established by these output-level measurements.

## Scientific interpretation and limits

The result that is closest to a defensible new contribution is a **failure mode of transfer across evaluation regimes**: natural-role fine-tuning can raise a natural active-argument benchmark and still fail to produce stable actor/patient behavior under meaning-preserving syntax changes. In these two matched 0.5B starts, the change is specifically consistent with a stronger local-subject shortcut. The untouched 1.5B check shows that a different positional preference can coexist with better average accuracy and the same failure of full-event consistency. This points to an evaluation design that measures **both roles, both event directions, several equivalent forms, and each question separately**, rather than a single actor score.

This does **not** make active/passive/cleft role evaluation itself new. [Denning et al. (2026)](https://doi.org/10.1162/OPMI.a.365) used active, passive, and cleft constructions to study thematic-role representations and found that syntactic similarity can dominate role similarity. [Wilson et al. (2023)](https://aclanthology.org/2023.tacl-1.78/) fine-tuned models on novel argument distributions, tested many unseen syntactic forms, and documented a linear-order bias in a harder generalization condition. The potential distinguishing question here is **what matched natural-role adaptation changes in behavior**, and whether a carefully controlled role-equivariant intervention fixes the resulting subject/role confusion without merely teaching a template. A targeted literature search did not establish that no one has tested this exact comparison; its novelty remains provisional.

This does **not** yet establish literature-level novelty, a general LLM property, or a successful repair. The items are synthetic, the models are from one Qwen family, common names and verbs may have appeared in pretraining, and the two 0.5B starts share training data and architecture. “Who acted on the other person?” and “Who was acted on?” may be harder than the more explicit wordings. The sentence-omitted target-margin mean in the original JSON summary is exactly zero by balanced role reversal and **must not** be interpreted as absence of prompt or candidate bias. The post hoc analysis instead compares actor-versus-patient question margins per event against its omitted-sentence control. The candidate margins and direct outputs agree in this forced two-name format, so these data do not show a separate hidden knowledge versus output gap. No reserved human-review active/passive challenge was accessed.

The next decisive test is a **new, frozen, human-checked natural set** on which the target verb's grammatical subject, semantic actor, and first-mentioned participant can be separated. Compare the existing checkpoints first, then test a balanced role-equivariant training intervention against data-matched ordinary role supervision and a subject-only control. If the subject shift repeats and only the role-aware intervention reduces it while preserving correct actor *and* patient answers, that would support a stronger mechanism and repair claim. Until then, the current finding is a substantial exploratory lead and a clear limit on earlier positive results.

## Reproduction and audit

- [Frozen 0.5B protocol](PROTOCOL.json), [inference code](evaluate.py), five raw checkpoint JSON files, [analysis code](analyze.py), and [derived analysis](ANALYSIS.json).
- [Prospective 1.5B supplement](QWEN15_SUPPLEMENT.json), [supplement inference code](evaluate_qwen15.py), and [raw 1.5B outputs](untouched_qwen15.json).
- [Earlier natural-adaptation results](../../llm_role_natural_repair/RESULTS.md) for the distinct QA-SRL Bank test and its limitations.
