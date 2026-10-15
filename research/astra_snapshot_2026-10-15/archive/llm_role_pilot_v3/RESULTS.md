# Fresh local Qwen role-prompt check (9 October 2026)

The [protocol](PROTOCOL.json) and all [items](ITEMS.json) were fixed before the model run. This independent panel used Qwen2.5-1.5B-Instruct with unmodified local weights: eight new name pairs, four verbs, and both directions of each relation gave 64 events. Each event was rendered in active and passive voice and asked with three fixed prompts (384 responses). The 64 events, not the 384 related prompts, are the main descriptive units. [Raw responses and scoring](RESULT.json) are preserved.

| Prompt and outcome | Active | Passive | Both forms correct, out of 64 events |
|---|---:|---:|---:|
| “Who did the action?”: correct actor | 13/64 | 60/64 | 12/64 |
| “Who performed the action?”: correct actor | 41/64 | 63/64 | 41/64 |
| Explicit `actor=NAME; patient=NAME`: both roles correct | 52/64 | 28/64 | 27/64 |

All 384 outputs were valid under the fixed parser, so these are role-answer errors rather than format failures. The two actor-only wordings changed the chosen name on 33/128 matched sentences: 32 changes corrected an earlier error and one introduced an error. The generic wording chose the sentence's last name in 111/128 cases. A fixed first-name or last-name actor rule would each be correct on 64/128 balanced actor-only items. The pattern occurred across all eight name pairs, although the strength varied by verb; “followed” showed no actor-accuracy gain from the wording change.

The new panel replicates an evaluation-validity concern from the earlier adaptive pilot: changing a short phrase in a role question can change the measured ability of this model on the *same sentences*. The explicit joint format helps active sentences but leaves many passive errors, so it is not a complete prompt fix. These results are limited to one small pretrained model, constructed English sentences, and the chosen prompts. They do not establish unseen training combinations, a shared internal mechanism with the controlled Transformer, or that ordered-role training would improve this LLM. Natural-sentence data and a second model family remain separate next tests.
