# Proposed output-field-order discriminator

**Status:** design only, awaiting team overlap check. No new model inference or training has run. This is an exploratory reuse of the [unsealed synthetic adapter development items](../adapter_readout_diagnostic_2026-10-15/FROZEN_ITEMS.json), not independent replication.

## Question and minimal panel

Does asking for `Patient; Actor` instead of `Actor; Patient` change which mentioned participant the model assigns to each role? Hold the **same 96 sentences per world** fixed: eight withheld pairs × both actor directions × `praised`/`criticized` × active/passive/patient-cleft constructions. Evaluate three worlds with only the untouched base and the matched **ordinary adapter at seed 830211**: six model units, 576 sentence-model comparisons, and 192 underlying event-model combinations with all three constructions. Base is evaluated once per world, not reused as an independent observation for each adapter comparison. The ordered and participant-only adapters remain outside this minimal discriminator.

The same verb and name renderings are grammatically interpretable in all three forms: `Ryan praised Emma.`, `Emma was praised by Ryan.`, and `It was Emma whom Ryan praised.` The analogous `criticized` forms have the same actor/patient order. The two new direct questions are verb-specific and avoid the prior abstract phrase “received the action.”

## Interfaces and exact proposed strings

Keep the original chat template, system/user turns, tokenizer, local checkpoint, CPU float32 setup, greedy decoding, batch size four, and **20 new-token maximum per generated call**. In every condition the user turn remains `Sentence: {sentence}` except direct QA, which appends its question.

1. **Original-order extraction replay:** System: `Extract the actor and patient from the sentence. Reply exactly: Actor: <name>; Patient: <name>.` This is byte-for-byte the earlier extraction system string; the prompt template and frozen sentence are likewise unchanged. Replay the original order in the new run before judging any order effect, and compare exact generated token IDs with the archive. A mismatch is a technical stop requiring diagnosis, with no adaptive prompt edits.
2. **Reversed-order extraction:** System: `Extract the actor and patient from the sentence. Reply exactly: Patient: <name>; Actor: <name>.` Relative to the original, only the requested field sequence after `Reply exactly:` changes. The prefatory `actor and patient` wording stays fixed, as do the sentence and role labels.
3. **Natural direct actor question:** Reuse the prior direct-QA system `Answer with exactly one person's name. Do not add any other words.` User: `Sentence: {sentence}\nQuestion: Who {verb} someone?` For the two verbs this becomes `Who praised someone?` or `Who criticized someone?`.
4. **Natural direct patient question:** Same direct-QA system. User: `Sentence: {sentence}\nQuestion: Who was {verb}?` This becomes `Who was praised?` or `Who was criticized?`.

The old original-order raw predictions may be used as a replay check only if the prior frozen protocol/source/checkpoint/tokenizer hashes and exact prompt strings match. The readiness check finds they do for all six proposed units. Prefer a fresh original-order replay so the two field orders share a run environment and exact decoding implementation. If every replay matches, the archived predictions are redundant evidence of consistency; they are not counted as another sample.

## Locked predictions to test in a future run

For each sentence identify the first and last mentioned participant before inference. Three **fixed policies**, applied unchanged across active, passive, and cleft, make different order predictions:

| Policy | Original `Actor; Patient` | Reversed `Patient; Actor` | Same-event all-form signature |
|---|---|---|---|
| Input-last actor | `Actor = last; Patient = first` | Same role assignments despite text order | `LLL` in either field order: wrong on active, correct on passive/cleft |
| Same emitted name sequence / slot copying | First output field gets the same name under either order, and so does the second; first-to-first input copying is a specific case | Reversing requested fields flips actor/patient assignments while preserving the emitted name sequence | Original `FFF`, reversed `LLL` under first-to-first copying; an analogous `LLL` to `FFF` flip for last-to-first copying |
| Semantic role binding | Gold actor/patient under either requested order | Same gold role assignments; only presentation order changes | `FLL` in both orders, all three forms correct |

The slot-copy prediction above is directional and fixed before scores. Also record a possible last-to-first slot policy separately rather than refitting whichever orientation succeeds. Classify each joint output as first/last mapping, duplicate participant, other names, or malformed. Parse `Actor:` and `Patient:` in **either** field order only when each label occurs exactly once with one name, optional surrounding whitespace, and at most one final period; separately score obedience to the requested field order and exact requested format. Reject extra text/names and retain raw tokens. The fresh parser's allowance for either field order is specific to this new order experiment and does not retroactively rescore the older frozen diagnostic.

## Outcomes and dependence

Primary: within-sentence change in the **label-to-person assignment** when requested field order is reversed, with exact original-order replay as a technical gate. Report the predicted-versus-observed mapping categories, newly correct/newly incorrect role assignments, both-role accuracy, order compliance, and malformed rates for each form, world, and arm. Primary all-form event score requires the **same pair × direction × verb event** to have correct actor and patient for active, passive, and cleft in each output order; report `FLL` versus `LLL` event patterns rather than cherry-picking clefts. Compare ordinary versus its one world base as paired descriptive differences. Natural direct actor/patient responses receive strict one-name semantic and exact-format parsing, with raw output and token counts retained; they are an interface control, not a compute-matched baseline for one joint extraction call.

With only three graph worlds and one adapted seed each, report world estimates and paired event gain/loss counts without treating 576 sentences as independent replications. A pair-cluster bootstrap within world may be exploratory only; shared pair directions, verbs, and forms remain in the cluster. This experiment can distinguish **behavioral output allocation rules** on these synthetic items. It cannot identify an internal semantic representation or prove a general adaptation mechanism.

## Runtime and overlap

The six archived matching units took 29.6–33.0 seconds per 96-sentence extraction condition and 20.2–49.4 seconds for both earlier direct calls. A new reversed extraction plus two verb-specific direct calls projects about **6.3 minutes of model time**; adding an exact original-order replay projects **9.4 minutes**, with about **12–20 minutes wall time** allowed for loading, validation, writing, and variable generation lengths. This is an estimate from prior CPU runs, not a new benchmark. It would require 2,304 new generated responses (four interfaces × 576 comparisons) and no training, network, paid calls, or closed natural sets.

Local overlap: [frontier_fixed_fields_2026-10-15](../frontier_fixed_fields_2026-10-15/RESULTS.md) already crossed field orders on a different untouched 1.5B model and 12 previously examined natural QA-SRL events. That completed fixed-field study is closed to further inference. [fixed_fields_answer_probe_2026-10-13](../fixed_fields_answer_probe_2026-10-13/RESULTS.md) swapped cached fields during answer continuation. Neither is this fresh extraction-order comparison on matched 0.5B base/ordinary adapters and the fixed synthetic panel. Team overlap review found no exact local repeat.
