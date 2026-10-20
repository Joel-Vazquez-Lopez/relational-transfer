# Frozen pretrained query diagnostic

Exploratory development; 24 active subject/direct-object events, one pretrained family, two sizes. No training or semantic agent/patient confirmation.

## Qwen 0.5b

| Condition | Individual exact /48 | Both exact /24 | Mean span F1 |
|---|---:|---:|---:|
| direct | 17 | 0 | 0.432 |
| query_visible | 1 | 0 | 0.021 |
| query_blind | 8 | 2 | 0.261 |

{
  "query_visible_minus_direct": {
    "both_correct_difference_pp": 0.0,
    "paired_event_bootstrap_95_pp": [
      0.0,
      0.0
    ]
  },
  "query_blind_minus_direct": {
    "both_correct_difference_pp": 8.333333333333334,
    "paired_event_bootstrap_95_pp": [
      0.0,
      20.833333333333336
    ]
  }
}

## Qwen 1.5b

| Condition | Individual exact /48 | Both exact /24 | Mean span F1 |
|---|---:|---:|---:|
| direct | 19 | 2 | 0.549 |
| query_visible | 2 | 0 | 0.042 |
| query_blind | 31 | 11 | 0.686 |

{
  "query_visible_minus_direct": {
    "both_correct_difference_pp": -8.333333333333334,
    "paired_event_bootstrap_95_pp": [
      -20.833333333333336,
      0.0
    ]
  },
  "query_blind_minus_direct": {
    "both_correct_difference_pp": 37.5,
    "paired_event_bootstrap_95_pp": [
      16.666666666666664,
      58.333333333333336
    ]
  }
}

