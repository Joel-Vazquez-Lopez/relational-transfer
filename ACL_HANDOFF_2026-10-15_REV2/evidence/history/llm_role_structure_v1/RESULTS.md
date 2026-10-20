# Cross-form position audit on Qwen (10 October 2026)

The [protocol](PROTOCOL.json) fixed 64 events from eight newly paired name pairs and four new verbs. Each event appeared in active, passive, object-cleft and subject-cleft form, with the same two actor questions. The unmodified Qwen2.5-1.5B-Instruct model produced a valid one-name first token on all 512 prompts. [Raw results](RESULT.json) and the [summary](SUMMARY.json) are saved.

| Sentence form | Generic “Who did the action?” | “Who performed the action?” | Actor's name position |
|---|---:|---:|---|
| Active: “Alice rescued Bob” | 4/64 | 63/64 | first |
| Passive: “Bob was rescued by Alice” | 64/64 | 64/64 | last |
| Object cleft: “It was Bob that Alice rescued” | 56/64 | 41/64 | last |
| Subject cleft: “It was Alice who rescued Bob” | 63/64 | 64/64 | first |

The names and verbs in the examples illustrate the templates; each cell contains 64 balanced instances. A universal last-name rule cannot explain the subject-cleft result: the model chose the first name as actor in 63/64 generic-question cases. The wording effect also depends on syntax. “Performed” improves active sentences from 4/64 to 63/64, but **reduces** object-cleft accuracy from 56/64 to 41/64 on the same underlying events. These results are descriptive rather than a significance claim over 512 independent examples; the prompts reuse 64 events.

Together with the block-20 patch experiments, this indicates that Qwen's answer depends on both question wording and sentence structure. The internal patch changes a whole prompt-conditioned state; it does not yet isolate a general role circuit. Before proposing a training repair, the next study should test whether a role-related signal transfers across constructions and name positions, including authentic sentences, with controls for the shortcut that this audit exposed.
