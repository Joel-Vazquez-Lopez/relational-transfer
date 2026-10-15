# Cross-family exploratory diagnostic

Exploratory reused constructed examples, six name-pair clusters; not independent natural-language confirmation

## Joint actor/patient correctness

Each cell counts events with both role answers exactly correct, out of12. L/E denotes lexical versus explicit wording.

| Model/interface | Active L/E | Passive L/E | Subject cleft L/E | Object cleft L/E |
|---|---:|---:|---:|---:|
| llama | 11/12 | 8/10 | 12/12 | 2/0 |
| llama_no_prefill | 12/11 | 12/9 | 12/12 | 4/0 |
| gemma | 12/12 | 12/12 | 12/12 | 12/12 |
| phi3 | 12/12 | 12/12 | 12/12 | 9/1 |
| qwen05 | 3/0 | 0/1 | 4/0 | 0/0 |
| qwen15 | 5/12 | 1/1 | 6/3 | 0/0 |
| qwen7b | 12/12 | 10/8 | 11/12 | 3/8 |

## Gemma observations

- active/lexical: joint12/12; actor12/12; patient12/12; compliant24/24.
- active/explicit: joint12/12; actor12/12; patient12/12; compliant24/24.
- passive/lexical: joint12/12; actor12/12; patient12/12; compliant24/24.
- passive/explicit: joint12/12; actor12/12; patient12/12; compliant24/24.
- object_cleft/lexical: joint12/12; actor12/12; patient12/12; compliant24/24.
- object_cleft/explicit: joint12/12; actor12/12; patient12/12; compliant24/24.
- subject_cleft/lexical: joint12/12; actor12/12; patient12/12; compliant24/24.
- subject_cleft/explicit: joint12/12; actor12/12; patient12/12; compliant24/24.

## Interpretation limits

Reused constructed events are exploratory comparability checks, not new natural-language confirmation. Only six participant-pair clusters support the paired intervals in CROSS_ANALYSIS.json. All-zero/all-one bootstrap intervals are degenerate, not population certainty.

Gemma uses a hosted31B model with no documented native assistant prefill and no arbitrary candidate scores. The matched no-prefill Llama condition checks that specific interface difference; sizes, families, precision, templates and server implementations remain different. Do not interpret this as pure scaling or infer an internal mechanism.

The rejection of one request produced an HTTP429. Its originalEUR0.52430 worst-case reservation remains held. The following amount is conservative budget accounting, including that unknown-charge hold, NOT an invoice estimate: 0.58085600 EUR. The total cap was1 EUR.

The next decision must follow the completed model-level results: if Gemma succeeds, prominently report the capability boundary; if errors remain, inspect compliance and positional/grammatical shortcuts before a mechanistic study. Independently reviewed natural examples remain necessary.
