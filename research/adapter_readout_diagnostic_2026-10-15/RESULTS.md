# Matched adapter readout diagnostic: development results

**Status:** completed on 2026-10-15 under the [frozen protocol](FROZEN_PROTOCOL.json). This is a synthetic development diagnostic, not a new confirmation result or an internal-mechanism test. The [analysis](ANALYSIS.json) and 21 [raw unit outputs](full_outputs) retain every prompt response and all unit-level estimates. The earlier [smoke](PREFLIGHT.md) events were excluded.

## Design and completion

Three existing 10-name worlds supplied eight deterministic withheld pairs each. Both actor directions, two new verbs (`praised`, `criticized`), and active, passive, and patient-fronted cleft forms produced 96 new sentences per world. Each world was evaluated with its untouched Qwen2.5-0.5B model and with ordinary, ordered-role, and participant-only adapters at two matched starts. There are 21 model evaluations, 2,016 sentence-model comparisons, and 8,064 interface prompts. All units and records are complete. The run used local CPU inference only; summed model inference time was 24.85 minutes.

The primary behavioral comparison used two fixed direct questions, one for actor and one for patient, versus one fixed joint extraction request on the same sentence. Each generated call had a 20-token greedy limit. Semantic parsing allowed surrounding whitespace and one final period while separately recording exact requested-format compliance. The archived training-compatible actor question was evaluated by its first token as a secondary compatibility measure. No prompt or parser was changed after the protocol lock.

## Main result: patient-fronted clefts

Percentages below average the three world baselines once each or the six matched trained units equally. Each base world contributes 32 cleft sentences; each trained unit contributes 32. “Both” requires the actor and patient to be correct for the same sentence.

| Model | Strong direct actor | Strong direct patient | Strong direct both | Joint extraction actor | Joint extraction patient | Joint extraction both | Archived actor first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untouched | 82.3% | 9.4% | 3.1% | 90.6% | 90.6% | 90.6% | 81.2% |
| Ordinary | 69.8% | 14.6% | 2.1% | 85.9% | 85.9% | 85.9% | 62.5% |
| Ordered-role | 63.5% | 17.2% | 1.6% | 84.9% | 84.9% | 84.9% | 77.1% |
| Participant-only | 72.9% | 16.1% | 2.6% | 85.9% | 85.9% | 85.9% | 89.1% |

Extraction exceeded the two-call direct-question joint score on every unit's patient-cleft subset. At sentence level, extraction was newly joint-correct on 84/96 untouched, 161/192 ordinary, 160/192 ordered-role, and 160/192 participant-only cleft comparisons; it was newly joint-incorrect on zero in each arm. The across-world mean extraction-minus-direct joint difference was +87.5 points for untouched, +83.9 ordinary, +83.3 ordered, and +83.3 participant-only. Exploratory 95% pair-cluster bootstrap intervals, resampling the eight unordered pairs within each world and averaging two seeds within each trained graph, were respectively **[+79.2, +94.8]**, **[+76.6, +91.1]**, **[+76.0, +90.6]**, and **[+77.1, +89.6]** points. These intervals do not represent replication across model families or many graph worlds.

The archived first-token question reproduced the familiar prompt-specific trend on this new subset: ordinary cleft actor accuracy was below untouched; ordered and participant-only were above ordinary. That ranking did **not** carry across fixed interfaces. Under the stronger direct actor question, ordered-role was below ordinary on average; under joint extraction, all three trained arms were close to one another and slightly below untouched. This is evidence of interface-sensitive behavior, not evidence that ordered labels preserve a particular internal role representation.

## Construction differences and format

Each form has 96 untouched and 192 comparisons per trained arm. Percentages are joint actor-and-patient accuracy on the two direct calls versus the extraction response.

| Form | Untouched direct → extraction | Ordinary direct → extraction | Ordered direct → extraction | Participant direct → extraction |
|---|---:|---:|---:|---:|
| Active | 9.4% → 21.9% | 32.8% → 51.0% | 28.1% → 41.7% | 16.1% → 38.0% |
| Passive | 6.2% → 71.9% | 12.5% → 88.0% | 34.9% → 70.8% | 28.1% → 70.8% |
| Patient cleft | 3.1% → 90.6% | 2.1% → 85.9% | 1.6% → 84.9% | 2.6% → 85.9% |

The direct patient call is the main interpretation limit. Across all forms, semantic-format failure was **40.6%** for untouched, **2.1%** for ordinary, **3.1%** for ordered, and **14.6%** for participant-only. Even valid one-name patient answers were often the actor. For example, the untouched model sometimes answered with a full sentence despite the one-name instruction. Extraction semantic-format failure was lower: **2.4%, 0%, 3.3%, 3.6%** in the same order. Exact-format noncompliance for extraction, often just a terminal period, was **16.0%, 11.1%, 15.5%, 17.7%**. No generated strong-QA or extraction response reached the 20-token limit.

The interfaces are not compute matched: strong direct QA makes two separate calls with up to 20 new tokens each; extraction makes one joint call with up to 20. Observed mean unpadded generated tokens per sentence were approximately 7.36 direct versus 8.04 extraction for untouched, 4.17 versus 8.11 ordinary, 4.25 versus 7.99 ordered, and 5.19 versus 7.99 participant-only. The archived actor measure uses only one first-token forward pass.

## Every patient-cleft unit

The columns are strong direct actor/patient/both, extraction both, and archived actor first-token accuracy, all percentages over 32 sentences. The full analysis also includes every unit by construction and actor/patient extraction separately.

| World / seed / arm | Direct A | Direct P | Direct both | Extraction both | Archived A |
|---|---:|---:|---:|---:|---:|
| fresh1 / base | 78.1 | 12.5 | 6.2 | 87.5 | 71.9 |
| fresh1 / 830211 / ordinary | 50.0 | 31.2 | 3.1 | 81.2 | 46.9 |
| fresh1 / 830211 / ordered | 68.8 | 21.9 | 6.2 | 87.5 | 75.0 |
| fresh1 / 830211 / participant | 65.6 | 31.2 | 3.1 | 87.5 | 78.1 |
| fresh1 / 830212 / ordinary | 59.4 | 31.2 | 9.4 | 87.5 | 50.0 |
| fresh1 / 830212 / ordered | 46.9 | 18.8 | 0.0 | 87.5 | 53.1 |
| fresh1 / 830212 / participant | 62.5 | 28.1 | 12.5 | 87.5 | 78.1 |
| fresh2 / base | 81.2 | 9.4 | 3.1 | 96.9 | 81.2 |
| fresh2 / 830211 / ordinary | 68.8 | 9.4 | 0.0 | 87.5 | 65.6 |
| fresh2 / 830211 / ordered | 56.2 | 25.0 | 3.1 | 68.8 | 71.9 |
| fresh2 / 830211 / participant | 78.1 | 9.4 | 0.0 | 68.8 | 90.6 |
| fresh2 / 830212 / ordinary | 75.0 | 6.2 | 0.0 | 81.2 | 65.6 |
| fresh2 / 830212 / ordered | 81.2 | 6.2 | 0.0 | 90.6 | 90.6 |
| fresh2 / 830212 / participant | 81.2 | 6.2 | 0.0 | 100.0 | 93.8 |
| fresh3 / base | 87.5 | 6.2 | 0.0 | 87.5 | 90.6 |
| fresh3 / 830211 / ordinary | 81.2 | 6.2 | 0.0 | 87.5 | 65.6 |
| fresh3 / 830211 / ordered | 71.9 | 9.4 | 0.0 | 87.5 | 90.6 |
| fresh3 / 830211 / participant | 87.5 | 9.4 | 0.0 | 87.5 | 93.8 |
| fresh3 / 830212 / ordinary | 84.4 | 3.1 | 0.0 | 90.6 | 81.2 |
| fresh3 / 830212 / ordered | 56.2 | 21.9 | 0.0 | 87.5 | 81.2 |
| fresh3 / 830212 / participant | 62.5 | 12.5 | 0.0 | 84.4 | 100.0 |

## Interpretation

These fixed interfaces produce markedly different answers on the same synthetic event. The patient question is a poor standalone readout in this model and wording; its low accuracy and format failures dominate the two-call joint score. Joint extraction also fails often on active sentences, so its high cleft performance is construction dependent. The result does not show that extraction accesses a preserved internal relation or that fine-tuning changed a specific internal representation. It shows that the archived ordinary-fine-tuning cleft regression is strongly sensitive to the query interface, and that the ordered and participant-only adapter rankings are similarly interface dependent on this development panel.
