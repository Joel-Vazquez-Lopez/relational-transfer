# Dense early-stopping control: 25, 30, and 35 updates

## Question

The [coarse learning-curve experiment](../llm_role_trajectory/RESULTS.md) showed a steep drop in patient-cleft accuracy between 20 and 40 ordinary fine-tuning updates. This follow-up tests whether a checkpoint at 25, 30, or 35 updates could match the auxiliary training arms at their familiar-form accuracy. The [protocol](PROTOCOL.json) was saved before the new training runs.

We used the same model, examples, update order, optimizer, and evaluation screen as the coarse experiment. Four development units (fresh1/fresh2, two starts each) determined one **global** ordinary checkpoint for comparison with each auxiliary arm. The selection rule required ordinary training to reach that arm's development familiar-form accuracy, then chose the eligible checkpoint with the best patient-cleft accuracy. The two starts on fresh3 and two unused question wordings checked the choices. Every dense run reached an adapter at 40 updates that exactly matched the previously saved 40-update adapter.

## Development selection

| Ordinary updates | Patient cleft | Familiar active/passive | Eligible against ordered roles? | Eligible against participant-only? |
|---:|---:|---:|:---:|:---:|
| 20 | 67.4% | 98.25% | No | No |
| 25 | 57.0% | 99.70% | No | Yes |
| 30 | 46.4% | 99.80% | Yes | Yes |
| 35 | 42.8% | 100.0% | Yes | Yes |
| 40 | 41.6% | 100.0% | Yes | Yes |
| 60 | 44.5% | 100.0% | Yes | Yes |
| 80 | 45.4% | 100.0% | Yes | Yes |
| 100 | 44.2% | 100.0% | Yes | Yes |
| 120 | 43.2% | 100.0% | Yes | Yes |
| **Ordered roles, 120** | **67.5%** | **99.8%** | — | — |
| **Participant-only, 120** | **74.5%** | **98.95%** | — | — |

The fixed rule selected **30 ordinary updates** against ordered roles and **25 ordinary updates** against participant-only training. No eligible ordinary checkpoint among the nine sampled steps reached either auxiliary arm's development cleft accuracy. The 20-update ordinary point retained more cleft accuracy but did not meet either familiar-form floor.

## Check on fresh3 and unused question wordings

Each start has 500 patient-cleft prompts and 1,000 familiar active/passive prompts. Both question wordings here were excluded from checkpoint selection. All comparisons use the same rows.

| Comparison | Start | Ordinary patient cleft | Auxiliary patient cleft | Ordinary familiar | Auxiliary familiar |
|---|---:|---:|---:|---:|---:|
| 30 updates vs ordered roles | 830211 | 319/500 | 421/500 | 979/1000 | 982/1000 |
| 30 updates vs ordered roles | 830212 | 336/500 | 398/500 | 989/1000 | 964/1000 |
| **Combined** | | **65.5%** | **81.9%** | **98.4%** | **97.3%** |
| 25 updates vs participant-only | 830211 | 367/500 | 458/500 | 970/1000 | 979/1000 |
| 25 updates vs participant-only | 830212 | 412/500 | 471/500 | 976/1000 | 947/1000 |
| **Combined** | | **77.9%** | **92.9%** | **97.3%** | **96.3%** |

Both auxiliary arms retain more patient-cleft accuracy than their development-selected ordinary checkpoint in **both** reserved starts. In the first start, each auxiliary arm also slightly exceeds ordinary familiar accuracy; in the second, ordinary familiar accuracy is higher. Combined across starts, each auxiliary arm gains substantial cleft accuracy while losing about one percentage point on familiar forms. These are trade-offs, so we do not describe any arm as uniformly best.

## What this does and does not establish

The result rules out a **simple early-stop explanation at the nine tested ordinary checkpoints** under the same learning rate and examples. The additional 25/30/35 points make that conclusion more informative than the coarse sweep, particularly because cleft accuracy fell rapidly in this interval. It does **not** prove that no update step, learning rate, regularizer, or alternative ordinary fine-tuning schedule can reach the same trade-off. Nor does it establish that ordered actor/patient labels are required: the participant-only auxiliary arm remains stronger on patient clefts at the tested weight. The design uses synthetic templates, one small Qwen model, three layouts, and two starts per layout. Fresh3's final-arm results were known before this follow-up was designed, although its dense checkpoints and reserved-wording scores did not determine the checkpoint choice.

The scientifically useful claim now is: **auxiliary participant supervision can retain patient-first role-test performance during fine-tuning at a modest cost on familiar forms, and the gain is not reproduced by the sampled ordinary early-stop controls.** A stronger paper needs an independent model family and authentic role-labeled sentences. A further mechanism claim requires controls that separate participant identity supervision from generic auxiliary gradients.

The [machine-readable analysis](ANALYSIS.json) verifies all six dense runs, exact step-40 adapter matches, the fixed selection, and row-by-row equality of the reserved tests. [choose_checkpoint.py](choose_checkpoint.py) and [confirm_selected.py](confirm_selected.py) reproduce the decision and confirmation from saved artifacts.
