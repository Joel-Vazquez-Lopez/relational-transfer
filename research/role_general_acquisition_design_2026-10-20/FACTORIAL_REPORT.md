# Role-demand factorial: outcome

**Status: STOP_DEVELOPMENT_GATE**

No prompts, hypotheses, sites or training settings were changed after the frozen model run began.

## Completed models

| Phase | Seed | Arm | Source paired | Target paired | Familiar |
|---|---:|---|---:|---:|---:|
| development | 7284101 | both | 0.807 | 0.469 | 0.972 |
| development | 7284101 | neither | 0.042 | 0.016 | 1.000 |
| development | 7284101 | source | 0.755 | 0.031 | 1.000 |
| development | 7284101 | source_double | 0.703 | 0.016 | 1.000 |
| development | 7284101 | target | 0.052 | 0.630 | 0.997 |
| development | 7284101 | target_double | 0.052 | 0.729 | 0.997 |

## Development gate

```json
{
  "baseline_headroom": true,
  "familiar_all_pass": true,
  "identity_all_pass": true,
  "required_comparisons_learned": false,
  "some_treatment_effect": true
}
```

## Interpretation boundary

This is one synthetic ordered-argument task. A failed prerequisite is not evidence that role-general transfer is impossible. A positive prototype is not a unique mechanism, a linguistic invariance claim, or evidence of novelty. All stopped/failed arms are retained. See FACTORIAL_RESULTS.json and MODEL_ENDPOINTS.csv for full results.
