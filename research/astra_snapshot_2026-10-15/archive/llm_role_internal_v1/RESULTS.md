# Qwen role-state diagnostic, first frozen panel (10 October 2026)

The [protocol](PROTOCOL.json) was fixed before the hidden-state run. Qwen2.5-1.5B-Instruct was not updated. Eight name pairs, four reversible verbs, two directions, two voices and two actor questions yielded 64 underlying events and 256 related prompts. Names were selected so each could be a single answer token. The [full readouts](READOUT.json), [patches](PATCHES.json), and [summary](SUMMARY.json) are saved.

| Actor question | Active correct | Passive correct |
|---|---:|---:|
| “Who did the action?” | 3/64 | 64/64 |
| “Who performed the action?” | 38/64 | 64/64 |

All first-token answers were one of the two names. There were 35 cases where the generic question gave a wrong actor and the performed question gave the correct one; these were all active sentences. A fixed untrained logit-lens readout of the generic prompt's state favored the true actor in 23/35 of these cases after block 20, but in 0/35 at the final block. Across **all** generic-active cases, the block-20 readout favored the actor in only 32/64. The conditional 23/35 pattern therefore does not establish a generally reliable hidden role representation.

For the 35 discordant cases, a pre-specified patch replaced the generic prompt's final-token state after block 20 with the state from the performed prompt. A control donor used the performed prompt for the same names and verb with the actor and patient reversed. The correct donor produced an actor answer in 21/35 cases, versus 16/35 with the reversed donor. Its actor-minus-patient score margin was higher by a mean 0.524. The equal-weight contrast was positive in all eight name-pair groups (exact two-sided pair-level sign-flip p = 0.0078125). A block-14 patch produced no meaningful separation.

The block-20 result shows that a prompt- and event-dependent state can causally steer the answer. The control donor also changes the sentence, so this test does not isolate a linguistic role signal from name order or sentence matching. The logit lens is a convenient fixed readout, not a calibrated proof of what the state represents. This study supports a question-dependent computation in this one model; it does not establish that the correct role was present but ignored under the failing prompt.
