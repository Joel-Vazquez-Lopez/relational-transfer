# Stronger-model development diagnostic — 15 October 2026

## Scientific decision

**Qwen2.5-7B largely removes the small-model question-contrast reversals under the clearer interfaces, but does not eliminate construction-dependent answering errors.** The residual is not yet evidence for an internal question-mapping mechanism. Prioritize fresh reviewed behavioral validation over a large mechanistic campaign.

All results below are exploratory. The examples were previously examined constructed development items. No independent natural-language confirmation was conducted.

## Completed experiment

Three untouched, pinned MLX 4-bit Qwen2.5 Instruct models: 0.5B, 1.5B, 7B. All use 4-bit/group-64 weights, identical rendered prompts and token IDs, and greedy decoding. Each completed 1,152 records: 576 archived-interface prompts on all 36 old events; 192 candidate-free generations on a predetermined 12-event subset; 384 candidate-list contrasts with both candidate orders. No fine-tuning, sealed data, or paid inference.

The stronger interface uses a fixed `Answer:` assistant prefix, a names-only instruction, and two question pairs. Lexical questions ask “Who [verb] someone in this sentence?” / “Who was [verb] by someone in this sentence?” Explicit questions ask which person carries out the action / has the action done to them. This is a fixed comparison, not a prompt search.

## Candidate-free joint actor/patient correctness

Each cell counts events for which **both** role answers are exactly correct, out of the same 12 events.

| Construction | 0.5B lexical | 1.5B lexical | 7B lexical | 7B explicit |
|---|---:|---:|---:|---:|
| Active | 3 | 5 | 12 | 12 |
| Passive | 0 | 1 | 10 | 8 |
| Subject cleft | 4 | 6 | 11 | 12 |
| Object cleft | 0 | 0 | 3 | 8 |

At 7B, object-cleft actor accuracy is 12/12 under each wording; patient accuracy is 3/12 lexical and 8/12 explicit. Across both wordings, 12 of the 13 object-cleft errors return the actor. The other output is a full clause instead of a name. Overall 7B candidate-free compliance is 191/192, so formatting alone does not explain the errors.

Object-cleft joint accuracy changes from 3/12 to 8/12 when question wording changes. Neither wording fixes every construction: passive joint accuracy changes from 10/12 to 8/12. These are paired descriptive results, not a selected winning prompt.

## Contrasts are distinct from correct answers

Let mA and mP be actor-minus-patient candidate logit contrasts under actor and patient questions. We record a reversal when (mA − mP)/2 < 0. A positive contrast change does **not** require either answer to cross the correct decision boundary.

For candidate-free object clefts, reversals fall from **9/12 at 1.5B to 0/12 at 7B** under lexical questions, and from **11/12 to 1/12** under explicit questions. Thus the precise small-model reversal account is substantially weakened by scaling and interface choice, even though answering errors remain.

Under the identical old interface on all 36 events, 7B still has 8/36 object-cleft reversals for the actor-performed/patient-verb pair, versus 20/36 in the matched 1.5B run; joint correctness is 17/36 versus 6/36. The other old question pair gives 5/36 reversals at 7B. Clearer candidate-free questions and old candidate-list prompts are separate conditions, so their differences cannot be attributed to wording alone.

## Simpler explanations remain

Changing candidate order flips 7B contrast-based choices in 15/192 paired comparisons; 11 of these occur in the 48 object-cleft comparisons. Candidate-free errors show that candidate-list position is not the whole account.

The 12 wrong object-cleft names coincide with the event's grammatical subject, its actor, and its last-mentioned participant. These properties coincide in these templates: **this experiment cannot distinguish them.** No single tested fixed heuristic reproduces the complete four-construction response pattern, but a construction-dependent fallback remains a strong simple alternative to interacting internal mechanisms. Contrast reversals are behavioral measurements, not causal evidence about representations.

## Uncertainty and individual changes

Bootstrap intervals resample six participant-pair clusters, retaining both event directions and all matched forms. For 7B lexical questions, the object-cleft-minus-active joint gap is **−75 pp**, descriptive 95% cluster-bootstrap interval **[−91.7, −58.3] pp**. The matched 7B-minus-1.5B change in that gap is **−33.3 pp [−83.3, +16.7]**: better absolute accuracy does not establish reduced structural sensitivity. Under explicit wording the interaction is **+66.7 pp [+33.3, +91.7]**, showing that the conclusion depends on the question interface.

Across both question wordings, the matched 7B-versus-1.5B candidate-free object-cleft comparison has **25 newly correct individual answers and 13 newly incorrect answers**, not uniform improvement. All transitions are retained in ANALYSIS.json. Six reused name pairs provide limited uncertainty estimates; degenerate bootstrap intervals at zero observed failures are not proof of a zero population failure rate.

## Archive comparison and validity

Archived full-precision predictions are preserved separately. Moving to quantized MLX changes 165/576 individual 0.5B answers (73 gains, 92 losses) and 91/576 1.5B answers (65 gains, 26 losses). These are backend/precision comparisons, not scaling effects. The new scaling comparison holds backend, quantization scheme, prompts, and scoring fixed; it still cannot isolate parameter count from all model differences.

All three downloaded weight checksums match pinned Hub objects. Cached/uncached technical smoke checks give zero maximum logit difference. All 3,456 result records are complete and uniquely keyed. The 37 closed diagnostic files and archived source hashes remain unchanged. Local M2 inference took approximately 54 s, 150 s, and 663 s; 7B peak MLX-tracked allocation was 4.50 GB (not total OS/process memory). External inference cost: €0.

## Independent review and next decision

NEW_REVIEW.csv contains 48 sentences: 12 new constructed events in four forms, with blank reviewer labels and no model predictions. REVIEW_INSTRUCTIONS.txt explains the check. It is **not human-validated, not authentic corpus data, and has not been evaluated**. Proposed AI labels are stored separately and should not be shown before independent annotation.

**Next:** independently review and freeze the new development set, then run the unchanged 7B protocol on it. Before interpreting a replicated residual mechanistically, the next discriminating design should separate last mention from event subject/actor, for example by moving a matched nonparticipant attribution before versus after the event. This is a proposed control, not a completed result. Llama-3.1-8B-Instruct is a possible later independent-family check; it was not downloaded or run.

For October 23, this merits a bounded robustness/evaluation branch. It does not yet justify making pervasive stronger-model question-mapping reversal the central mechanism claim.

## Reproduction

Model IDs, exact revisions, questions and subset are in PROTOCOL.json; source hashes in SOURCE_LOCK.json; per-model provenance in *_META.json. After installing the recorded MLX runtime and downloading those pinned files into models/qwen05, models/qwen15, models/qwen7b, run run.py for each model sequentially, then analyze.py. The runner skips completed records only after checking matching metadata and source hashes. Model weights and the isolated runtime are excluded from Git by the experiment's .gitignore.
