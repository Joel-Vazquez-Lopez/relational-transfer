# Authentic role contrast: frozen 51-event held-out check

## Why this test was run

The earlier QA-SRL A/B evaluation was dominated by answer-position bias. A direct generated-span test removed A/B options but the 0.5B model often copied the wrong phrase, shortened a span, or answered `Someone`. This follow-up asks a narrower question: **does the model's score favor the correct participant when the role question changes?** It scores the two annotated spans as candidate continuations; it does not ask the model to generate a span freely.

For each event, the prompt contains the original sentence and target verb. Two fixed questions ask for the actor and the other direct participant. The same actor and other-participant spans are scored under both questions. The primary margin is

`D = [log P(actor span | actor question) - log P(other span | actor question)] - [log P(actor span | other question) - log P(other span | other question)]`.

`D > 0` means the model favors the actor more under the actor question than under the other-participant question. Using the same two spans under both questions reduces lexical and span-length preference in this difference of differences. It does not prove that the model would *generate* either answer correctly.

## Split and validity checks

The 63 previously selected QA-SRL Gold events were split into 12 development events (three from each source partition) and 51 held-out events. Prompt wording and score were screened only on untouched Qwen2.5-0.5B and Qwen2.5-1.5B: the raw-sum contrast was positive on 10/12 and 12/12 development events, respectively. Before holdout inference, `PROTOCOL.json` froze the event list, code and model hashes, criterion, scoring, and controls. All 22 holdout evaluations completed and passed file/hash validation.

| Untouched model | With sentence | Sentence omitted | Difference |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 37/51 (72.5%) | 24/51 (47.1%) | +25.5 pp |
| Qwen2.5-1.5B-Instruct | 49/51 (96.1%) | 27/51 (52.9%) | +43.1 pp |

The sentence-omitted prompt retained the target verb, both candidate spans, and both role questions. These controls show that sentence context materially affects the role contrast. They also show a substantial model-size difference on this candidate-relative task. The frozen gate required at least 70% with sentence context and at least a 15-point drop without it for the 0.5B model; both conditions were met.

## Fine-tuning comparison on the same 51 events

The existing Qwen2.5-0.5B adapters were evaluated in six matched graph/start units: three worlds × two starts. Each unit used the identical 51 authentic events.

| Fine-tuning condition | Positive role contrasts | Equal-unit mean change from ordinary |
| --- | ---: | ---: |
| Ordinary | 217/306 (70.9%) | — |
| Ordered-role auxiliary | 219/306 (71.6%) | +0.65 pp |
| Participant-only auxiliary | 228/306 (74.5%) | +3.59 pp |

Ordered-role training improves accuracy in three units, worsens it in two, and ties in one. Participant-only training improves in five units and ties in one, but each gain is small: **+0 to +7.84 points**, below the frozen criterion of at least +10 points in five of six units. Mean raw-margin changes are also small (+0.020 for ordered-role; +0.073 for participant-only). Neither arm meets the prespecified transfer-improvement rule.

For comparison, the untouched 0.5B model scored 37/51 (72.5%) on these same events. The fine-tuned arms do not establish a large improvement over that baseline either. The 306 pooled evaluations are repeated measurements of only 51 authentic events; they are not 306 independent observations.

## Conclusion and limits

**The answer is now clearer:** the pretrained models show candidate-relative sensitivity to linguistic roles on these authentic sentences, and the sentence-omission control supports a contribution from context. However, the auxiliary training that helped the controlled synthetic cleft task **does not produce a substantial, consistent gain** on this authentic role-contrast test. A small participant-only advantage remains exploratory and is below the frozen criterion. This separates *role information present in the pretrained model* from *a successful transfer of the synthetic training intervention*.

The 51 events were held out from this particular contrast screen, but the full set of 63 was previously inspected and used for other evaluations. This is therefore not a fresh blind corpus confirmation. The answer spans come from QA-SRL Gold and all 126 span ranges matched the source sentences in the earlier audit, but independent student verification is pending. The second QA-SRL argument is not uniformly a semantic patient. This test supplies candidate spans to a likelihood calculation and cannot replace direct-answer evaluation. Model pretraining overlap is unknown. Six graph/start units share one 51-event set, so no prompt-level significance claim is made.

Saved files: `PROTOCOL.json` and `EVENTS.json` freeze the design; `ANALYSIS.json` contains all summary cells and matched differences; each model JSON contains four candidate log probabilities and both margins for every event. The development-only screen is in `../llm_role_natural_contrast_screen/`.
