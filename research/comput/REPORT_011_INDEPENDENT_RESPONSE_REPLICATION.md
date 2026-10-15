# COMPUT-011 — Independent historical response replication (2026-10-15)

**Status:** independent reanalysis of original user-uploaded `llm_role_internal_v2/RESULT.json`, no new model inference. Original SHA-256: `653a33a0a56f34a84c5c39649dda7bdb100990462daea0390cfcb2fcb8d7ea27` (matches archived Astra provenance). Original embedded protocol SHA: `3f28d7ae8cc089a01a522287469da3264d00cda2f312567d6107d78bb931ed63` (protocol file not provided for independent byte comparison).

An independently implemented text parser validated 384 records, eight participant pairs, four predicates, two orientations, both voices and three question wordings. Model answers were name-valid in 384/384 and correct in **240/384**.

## Positional explanation independently verified
- Always select last mentioned participant: **336/384** agreement with model output.
- Majority position selected by sentence voice and question wording: **368/384** (95.83%). First only for *active + Who performed the action?*, otherwise last.
- Leave-one-name-pair-out **368/384**, leave-one-predicate-out **368/384** (retrospective cross-validation on inspected factorial panel).
- Adding predicate: **378/384** in sample and leave-one-pair-out; back to **368/384** when leaving each predicate out with wording/voice fallback. Extra 10 fits **do not** transfer to withheld verbs.
- All **16** wording/voice errors are active + *Who performed the action?*: admired **13**, assisted **3**, greeted **0**, warned **0**. Every participant pair contains 1–3 errors.

## Matched question tests (same sentence)
Of 64 active sentence configurations, two synonymous actor questions (*Who did the action?* versus *Who performed the action?*) yield different participant answers **48/64** times; in 64 passive configurations they differ **0/64** times. The generic-actor and verb-patient questions (opposite reference roles) return identical answers on **128/128** matched sentences. This is a strong **question wording × voice behavioral asymmetry**, not proof of a neural algorithm.

## By condition
| Wording/voice | Model correct /64 | First /64 | Last /64 |
|---|---:|---:|---:|
| generic actor/active | 0 | 0 | 64 |
| generic actor/passive | 64 | 0 | 64 |
| performed actor/active | 48 | 48 | 16 |
| performed actor/passive | 64 | 0 | 64 |
| verb patient/active | 64 | 0 | 64 |
| verb patient/passive | 0 | 0 | 64 |

Original 48 patch summaries independently re-count: actor-donor condition actor correct 41/48, patient-donor condition 0/48, positive patch score differences 48/48. Interventions **not rerun**; positions and roles remain confounded.

## Integrity / limitations
- 11 local tests passed, including original-source SHA validation, complete condition coverage, label exclusion, and all empirical counts.
- Predictors derive order, voice, question type and verb from the sentence/question text. Gold/reference actor/patient fields used **only for scoring and data-integrity assertions**, never predictions.
- 384 observations repeatedly reuse 8 pairs and four verbs; retrospective analyst-selected rule family. No population inference from independent sample counts, no novel circuit claim.
- The comprehensive independently executable report, full analysis script, unit tests, 16 exceptions and machine-readable results are in the COMPUT-011 conversation archive (`COMPUT_011_INDEPENDENT_REPLICATION.zip`); the raw original source is not redistributed here.

## Next
Freeze this wording×voice rival and evaluate on untouched counterbalanced verbs, constructions, meaning-preserving rephrasings and human-reviewed natural roles, with parser failure and response compliance scored separately. No further synthetic headline should substitute for actual model generalization.
