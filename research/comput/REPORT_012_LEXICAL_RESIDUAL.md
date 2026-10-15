# COMPUT-012 — Independent lexical-margin residual audit

Date 15 Oct 2026. Source: original user-uploaded `llm_role_internal_v2/RESULT.json`, SHA256 `653a33a0a56f34a84c5c39649dda7bdb100990462daea0390cfcb2fcb8d7ea27`. **Executed retrospective reanalysis**, no new model inference or sealed data.

The historical simple wording × voice response policy predicts 368/384 answers; all 16 errors are active "Who performed the action?" (admired 13, assisted 3). Within those 64 outputs, the actor-minus-patient candidate score margin is strongly lexical:

| Verb | Mean margin (16 instances) |
|---|---:|
| admired | -1.72044 |
| assisted | 1.49751 |
| greeted | 4.23561 |
| warned | 7.04114 |

The strict ordering admired < assisted < greeted < warned holds in **all 16** matched pair/direction configurations. Scalar regression using only pre-estimated verb means explains **R²=0.825394** of archived margins; estimating verb means while leaving each complete name-pair group out yields **Q²=0.812372**, with **58/64** correct predicted margin signs. The adjacent verb comparisons are positive in each of eight name-pair groups (descriptive two-sided paired sign p=0.0078125 per contrast, unadjusted). Pair-cluster bootstrap (10,000 seeded draws) gives warned-minus-admired mean difference 8.7616 and descriptive 95% percentile interval [8.0512, 9.4636].

**Crucial qualification:** original actor margins require **reference actor/patient candidates** for scoring, so the scalar mean-margin model is not a fully label-free autonomous behavioral baseline. Prediction does not generalize to an unseen verb, because no new-verb margin can be learned without a prior reference. The four verbs and eight name pairs have been examined repeatedly. No claim of a new neural lexical algorithm or population significance follows.

The user-facing COMPUT_012_LEXICAL_FALSIFICATION.zip artifact contains standalone Python source, eight passing tests, machine-readable summary and more detailed scientific report. The raw 384 records remain outside this GitHub worktree. Next: prospective question/voice vs syntax/lexical competition on unseen verbs, name pairs, and human-verified counterfactual sentences, all hypotheses frozen before observing outcomes. No compute or paid inference authorized or executed.