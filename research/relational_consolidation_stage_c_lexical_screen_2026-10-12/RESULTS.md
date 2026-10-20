# Development-only lexical-composition screen — 12 October 2026

**Exploratory, one seed, no human grammar review, no confirmation family or reserved natural-language challenge used.** The [protocol](PROTOCOL.json) was fixed after the V1 ceiling and before this run. It withheld `followed` from all transformed training pairs, while canonical wake training still included that verb. The primary development set contains 30 events on 15 withheld participant pairs, both directions of `followed`, each with five forms and actor/patient questions.

| Arm | Joint correct events / 30 | Canonical | Passive | Object cleft | Subject cleft | Reversal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake / stop | 0 | 60/60 | 0/60 | 1/60 | 6/60 | 60/60 |
| 200 extra ordinary batches | 0 | 60/60 | 0/60 | 1/60 | 20/60 | 60/60 |
| Transformed-data augmentation | **30** | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 |
| S2 offline consistency | **30** | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 |
| S2 interleaved consistency | **30** | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 |

The three-way ceiling persists even when a verb is absent from transformed training. This does **not** show that all relational generalization is easy: the grammar is finite, the verb remains familiar from canonical sentences, the small model sees all names, and there are only three verbs. It does show that this particular offline objective offers no measurable advantage over ordinary augmentation or identical-information interleaving in a second, more demanding development split. The [V1 run](../relational_consolidation_stage_c_exploratory_2026-10-12/RESULTS.md) had the same tie without a verb split.

**Decision:** stop synthetic schedule experiments on this grammar; do not spend more seeds or open confirmation families to search for a difference. A new benchmark would need a genuinely harder composition and a fresh frozen protocol. The immediate research priority shifts to the already observed pretrained-model gap between candidate role information and stable direct answers.

[Runner wrapper](run.py), [saved config](runs/seed_4201/RUN_CONFIG.json), [wake score](runs/seed_4201/WAKE_RESULT.json), [full results](runs/seed_4201/RESULT.json), and model files are retained. The wrapper reuses the V1 training code without modifying its source; only Stage B family 0 training and development files were read.
