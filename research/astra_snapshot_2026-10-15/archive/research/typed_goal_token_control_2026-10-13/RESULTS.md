# Typed requested-role cue: token-matched, new-sentence check

**Completed 13 October 2026.** This is an adaptive follow-up to the positive fresh2 result. The null control was designed after that result; the 20 fresh3 sentences and success rule were [frozen](PROTOCOL.json) before training the null control or evaluating either arm on fresh3. No human-reviewed challenge set was opened.

## Design and audit

Two Qwen2.5-0.5B starts were compared on the same 160 QA-SRL Bank training items and 20 new sentence-disjoint QA-SRL Bank items. The existing `typed_goal` checkpoint receives `Requested role: ACTOR` or `PATIENT`, predicted from the question by a fixed train-only classifier. A newly trained control receives `Requested role: NO ROLE`. These tags are both two Qwen tokens. Within each start, the arms use the same source adapter, item order, 160 optimizer updates, `Actor; Patient; Answer` targets, **56,700 nonpadding input-plus-target tokens**, and **8,476 supervised target tokens**. Training/evaluation files and checkpoint hashes are checked in [ANALYSIS.json](ANALYSIS.json). The classifier gave the correct requested-role tag on all 40 distinct fresh3 questions (80 prompt rows); this QA-SRL question pattern is simple and templatic.

## Frozen outcome

| Start | Cue | Exact final answers /80 | All four role/order answers /20 | Both printed roles right /80 | Final answer matches requested printed field /80 |
|---|---|---:|---:|---:|---:|
| fresh1 | Typed | **73** | **15** | 73 | **79** |
| fresh1 | Neutral, token matched | 43 | 0 | 69 | 42 |
| fresh3 | Typed | **70** | **14** | 65 | **75** |
| fresh3 | Neutral, token matched | 44 | 0 | 66 | 44 |

The frozen screen passed in both starts: typed gains were **+30/80** and **+26/80** exact answers and **+15/20** and **+14/20** all-four-correct items. The printed role fields changed much less (+4/80 and −1/80). The large gain is concentrated in selecting the requested answer from those fields, rather than extracting more accurate fields. Descriptive paired bootstraps over the 20 sentences give exact-answer gain intervals of **+30.0 to +43.75 percentage points** and **+23.75 to +41.25 points**. They describe uncertainty within this selected source; the two starts and their test items are not independent studies.

The gain is present in actor and patient questions and both candidate orders. Fresh1 typed versus neutral exact answers were actor/order0 **19 vs 6**, actor/order1 **17 vs 10**, patient/order0 **20 vs 16**, patient/order1 **17 vs 11**, each out of 20. Fresh3 counts were **18 vs 6**, **17 vs 12**, **20 vs 16**, and **15 vs 10**. The null control still generated correct printed roles in most rows but often failed to use the requested one in its final answer.

## What this does and does not establish

The result strengthens the behavioral finding from [fresh2](../typed_goal_conditioning_2026-10-13/RESULTS.md): an explicit requested-role cue can repair final-answer routing for candidate-supported actor/patient questions in this small model and source distribution. Equalizing input length did **not** remove the effect. Merely putting the requested role first in the output had failed in the earlier [goal-ordering pilot](../goal_first_natural_routing_2026-10-13/RESULTS.md), so the location and use of the goal cue matter here.

This does **not** show that the model learned a general decomposition procedure, an internal causal representation, or improvement on unconstrained authentic QA. The 20 item actor/patient mappings were made automatically and remain unreviewed; [blind label-review sheet](FRESH3_BLIND_LABEL_REVIEW.csv) excludes model predictions. Questions are templatic, answers are supplied as candidates, both starts share a model family and training data, and this comparison was motivated by prior positive results. The strongest next test is a frozen, independently reviewed natural-language set with new question formulations, constructions, and no candidate support, plus another model family and matched structured-output baselines. Those are necessary before claiming a conference-level LLM repair.

## Candidate-free stress check

After seeing the candidate-supported result, I [froze](NO_CANDIDATE_PROTOCOL.json) a diagnostic that removed only the candidate line at test time for the same 20 sentences and fixed checkpoints. There was no retraining. It has **40 distinct questions per checkpoint** because candidate order no longer applies. The [analysis](NO_CANDIDATE_ANALYSIS.json) shows:

| Start | Cue | Exact final answers /40 | Both printed roles right /40 | Answer matches requested printed field /40 |
|---|---|---:|---:|---:|
| fresh1 | Typed | 13 | 4 | 39 |
| fresh1 | Neutral | 8 | 4 | 27 |
| fresh3 | Typed | 12 | 4 | 38 |
| fresh3 | Neutral | 9 | 4 | 27 |

On the corresponding candidate-supported order-0 prompts, typed had **39/40** and **38/40** exact answers. Thus the current model uses the explicit role cue to route its own generated fields, but those fields fail badly when participants are not supplied. This is a strong boundary on the present claim. Since this is an out-of-distribution, post hoc stress test on the same items, it does not establish what would happen after candidate-free training. The next intervention should target role extraction without candidate support, with a fresh external evaluation rather than another test on these 20 items.
