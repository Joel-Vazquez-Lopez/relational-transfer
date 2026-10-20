# Next experiment: candidate-free relational extraction

**Decision from 13 October 2026 results.** The typed cue repairs final-answer routing when the two participants are supplied. Removing them at test time makes printed role fields mostly wrong (4/40 jointly correct in each start). The next intervention must improve identifying the event and participant spans in the sentence, then preserve correct routing. Repeating candidate-supported training alone would not address this failure.

## Freeze the evaluation first

1. Build a new test set from a source or partition absent from all local training, fresh2/fresh3, and the 51 previously inspected QA-SRL Gold items. Ensure sentence, predicate, and participant spans are deduplicated. The old 20 and 40 item banks are development diagnostics only.
2. Independently review actor/patient labels and mark ambiguous, non-agentive, or non-reversible cases before revealing predictions. If independent review cannot be done, call the set provisional and make no publication-quality natural-language transfer claim.
3. Include direct and oblique participants, active/passive where a meaning-preserving rewrite is valid, and unfamiliar question wording. Evaluate without candidate lists. Keep candidate-supported prompts as a separate diagnostic.
4. Freeze exact-match and span-overlap scoring, accepted alternate spans, malformed-output treatment, construction cells, exclusion rules, and sample size before model inference. The unit for uncertainty is the source event or sentence, not repeated prompts.

## Matched comparison

- **A: direct answer** from sentence and question.
- **B: actor/patient fields then answer**, without a typed cue.
- **C: actor/patient fields then answer with an explicit requested-role cue** predicted from the question. Measure role classification separately; test both a predicted cue and a gold-cue diagnostic, never mix them in the main score.
- **D: C plus span-copy supervision**, requiring both role fields to be substrings of the source sentence and a named predicate to anchor the event. Match training examples, updates, target-token budget, and initial checkpoint as closely as possible; include a token-matched neutral cue for C/D.
- **Deterministic routing from printed fields** is a diagnostic baseline for B/C/D. A model should not get credit for consistent routing if it extracts wrong fields.

Use a small development screen on one start before committing the full factorial comparison. After choices are frozen, run at least two starts and another model family if available. Any additional model or data should be an explicit replication, not pooled silently with the screen.

## Decision rule for the stronger claim

The candidate-free intervention must improve **gold final-answer accuracy and joint role-field accuracy** on the new reviewed set, with gains across actor/patient questions and multiple constructions. If only answer-field agreement rises, the result is another routing repair. If fields improve but answers do not, routing remains broken. If both improve only with oracle role cues, question interpretation is the remaining bottleneck. A robust gain in both extraction and routing across an external source and another model family would support a substantially stronger paper.

Prior work already studies role querying and structured inference; see [literature position](LITERATURE_POSITION.md). The research contribution would be the isolated failure chain, matched repair tests, and generalization evidence, not the use of semantic-role labels alone.
