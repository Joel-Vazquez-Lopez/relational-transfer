# Frozen local Qwen development competence pilot

17 October 2026. **Exploratory development only; no activation intervention, no held-out estimate, no mechanistic claim.** The [script](epistemic_competence_dev.py) froze the [six constructed pairs](EPISTEMIC_COMPETENCE_DEV_ITEMS.json), [protocol](EPISTEMIC_COMPETENCE_DEV_PROTOCOL.json), question templates, whole-output scoring, model metadata and code hash before running local Qwen2.5-7B-Instruct MLX 4-bit. The [raw 24 responses](EPISTEMIC_COMPETENCE_DEV_RAW.jsonl) and [machine-readable report](EPISTEMIC_COMPETENCE_DEV_REPORT.json) are complete. No paid API or model download. Host Metal access was required because the sandbox did not expose a GPU; the run then completed. Raw SHA-256: `d800fe74b65355b62f5ff847a2669bd8920abe2573f1cce845a4b270acaffdf9`.

Each item combined two original source sentences from separate validation documents of the same MAVEN event type: one CT+ and one CT−/PS+, with distinct annotated Agents. Both sentence orders were asked two fixed questions: which Agent is explicitly stated to have completed the action, and which Agent is mentioned without established completion. This is a convenience selection by one reviewer; one CT+ Royal Marines source sentence is reused in two pairs. The event types sometimes cover different objects or action nuances, and “named agent” is unnatural wording for ordinary readers.

## Frozen score

Primary score: casefolded whole-word target Agent string appears in output and rival Agent string does not. No answer discarded. Exact normalized answer is secondary.

| Pair | Primary correct, four questions | Salient raw failures |
| --- | ---: | --- |
| Lukin refused to sign / Cvetković government signed | 0/4 | Swapped Agents when negative sentence first; abbreviated `Cvetković` fails frozen full-string score. |
| Murray failed to retake / Royal Marines recaptured | 3/4 | Gave `Japanese garrison` for noncompleted Agent when factual sentence first. |
| Durham Light Infantry attempted recapture / Royal Marines recaptured | 2/4 | Gave Durham for completed Agent when negative first and Japanese garrison for noncompleted Agent when factual first. |
| Linois tried to drive off opponent / Texians drove troops out | 3/4 | Gave Linois for completed Agent when negative first. |
| Chilean army could enter / Serbs entered | 2/4 | Gave Chilean army for completed Agent when negative first and Serbs for noncompleted Agent when factual first. |
| Kiplagat not present / Cristina Fernández de Kirchner present | 3/4 | Gave `Linet Masai`, another name in the negative sentence, for completed Agent when negative first. |

**Total: 13/24 primary; 8/24 exact; 0/6 pairs with all four answers correct.** By question, completed-Agent answers were 6/12 and nonestablished-Agent answers 7/12. By order and target position, the target Agent was in the **first sentence for 10/12 correct** questions, versus **3/12 when it was second**. The unscored string `Cvetković` is plausibly a shortened correct Agent in one case, but changing alias rules after output would violate the frozen primary metric; it does not repair any pair-level failure.

## Interpretation and next decision

This pilot **fails the competence gate** for a causal status/participant intervention. Its position dependence is consistent with a simple salience or first-sentence account, and the generic “action” wording may contribute. We cannot conclude that Qwen represents factuality but fails to use it, because this protocol did not separately establish status-only and described-participant competence on the same pair. We cannot conclude that any internal event-status variable exists. The right next study would predeclare those separate competence questions and a joint question on new, independently adjudicated pairs, with a first-sentence baseline and event-specific prompts. It must not tune on these six pair outcomes and then treat them as confirmation.

The novel paper claim remains conditional: a reproducible cross-task dissociation and selective causal transfer on **fresh** pairs that a syntax-aware copying baseline cannot explain under matched access. If such pairs and a discriminating prediction cannot be built, report the source and identifiability failures as a course project and stop the mechanistic paper claim.
