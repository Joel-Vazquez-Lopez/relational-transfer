# Prompt wording remains a transfer limit after role training

**12 October 2026.** Prospective, read-only diagnostic on eight existing Qwen2.5-0.5B checkpoints. The [protocol](PROTOCOL.json), 36 events, five question phrasings, and analysis endpoints were fixed before inference on these new items. No model was trained in this study. The reserved human-review challenge was not opened.

## Design

Six new name pairs and three new reversible verbs give 36 directed events. Each appears in active, passive, subject-cleft and object-cleft form. We ask three actor questions (the previously trained “performed” question and two questions unseen by these continuations: “did” and “carried out”) and two patient questions (the previously trained verb-specific question and a new “which person was VERBed” variant). Every checkpoint receives 720 prompts; all 5,760 responses are valid candidate names. Names and verbs are disjoint from the preceding matched construction tests, but they are still constructed English and may resemble pretraining text.

The models are the two natural-role source adapters plus the aligned, decoupled and mixed continuations from each source. Candidate order and actor direction are balanced by design. Raw first-token outputs and candidate-score margins are in eight `EVAL_*.json` files; [analysis](ANALYSIS.json) is reproduced by [analyze.py](analyze.py).

## Result: a gain in joint accuracy can conceal weaker prompt transfer

Each denominator in this table is **144 event–form groups** (36 events × four forms). “Actor all three” requires all three actor phrasings to be correct on the *same* event and form. “All five” additionally requires both patient phrasings. “Actor contradictions” counts event–forms whose three actor phrasings yield different names, whether right or wrong.

| Start | Checkpoint | Actor all three | All five | Actor contradictions |
| --- | --- | ---: | ---: | ---: |
| Fresh1 | Natural source | 95/144 | 27/144 | 32/144 |
| Fresh1 | Aligned | 82/144 | 62/144 | 46/144 |
| Fresh1 | Decoupled | 79/144 | 52/144 | 39/144 |
| Fresh1 | Mixed | 76/144 | 58/144 | 46/144 |
| Fresh3 | Natural source | 92/144 | 31/144 | 23/144 |
| Fresh3 | Aligned | 81/144 | 50/144 | 42/144 |
| Fresh3 | Decoupled | 70/144 | 47/144 | 52/144 |
| Fresh3 | Mixed | 72/144 | 54/144 | 56/144 |

All three role-trained continuations improve the demanding all-five measure relative to their source, largely because the source adapters were weak on patient questions. **None improves actor correctness across the three phrasings.** Contradictory actor answers increase for every continuation in both starts. This is a behavioral trade-off, not evidence that role information disappeared internally.

The clearest paired cell is the active sentence with “Who did the action?” The source gets **36/36** in each start; the mixed continuation gets **23/36** in each. All 13 source-correct/mixed-wrong events in each start are losses; none are gains. Yet on those same active items, the trained verb-specific patient question improves **6→35/36** and **20→36/36**. On passive sentences, the trained “Who performed the action?” actor question improves **20→33/36** and **16→32/36**. These sharply different changes are why one average role score is inadequate.

The source checkpoints themselves are imperfect: active “did” is excellent here, while their patient answers and some passive actor wordings are poor. The mixed curriculum does not dominate the aligned or decoupled continuations. Which construction and which wording one measures changes the conclusion.

## Interpretation and next test

This new lexical test supports the proposed prompt-robustness direction: the earlier Qwen1.5B wording effect was not merely an isolated anecdote, and role-labeled fine-tuning in Qwen0.5B can improve some semantic-role answers while reducing correctness on an unseen actor question. It does **not** show the same specific failure across models: the untouched/adapted 0.5B source is strong on active “did,” whereas the untouched 1.5B model was weak there. Nor does output contradiction distinguish a prompt-conditioned decision rule from an unstable internal role representation.

Before another training intervention, specify a matched comparison of (a) ordinary role supervision with multiple question phrasings and (b) an objective that rewards *correct* actor/patient answers across wording and sentence transformations. Both must see the same examples, labels and update budget. Hold out entire question families, fresh lexical combinations and some sentence forms; measure worst-question accuracy, paired contradictions, actor-and-patient joint correctness, and regression in previously strong cells. A generic hard-example-weighting control is needed if the proposed objective emphasizes low-margin examples. Human-checked authentic sentences and a second model family remain necessary before a broad LLM claim.

The established literature already studies prompt consistency and its training remedies, including [Elazar et al. (2021)](https://aclanthology.org/2021.tacl-1.60/) and [Hejabi et al. (2026)](https://aclanthology.org/2026.acl-long.71/). The possible contribution here is the matched connection between role-supervised training, paired actor/patient answers, question wording and construction transfer—not prompt sensitivity alone. No first-of-its-kind claim is made.

## Audit

The saved protocol hashes the exact script and rows, and each output file records the protocol and source checkpoint hashes. The analysis was checked against all eight complete files: **5,760 distinct, valid candidate-name answers**. The source checkpoints and prior result files were not modified.
