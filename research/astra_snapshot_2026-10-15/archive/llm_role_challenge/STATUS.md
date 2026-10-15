# Status, 12 October 2026

## What the present evidence says

The existing 75-item EWT result is a real improvement on its frozen **grammatical subject/object** evaluation. A new source-only AI audit found 42 clear agents and 14 clear semantic patients among those 75, under a narrow definition. This audit has not been verified by an independent human, so it does not justify re-scoring a post hoc semantic subset as confirmation. The previous EWT aggregates remain valid for the original syntax-based question; their interpretation must stay narrow.

## New confirmation test

- Prepared 32 previously unevaluated candidate active/passive pairs from the public Platt/KCL text-pair dataset. The passive originals are from Project Gutenberg; the active rewrites were generated automatically and require review.
- Confirmed zero exact normalized-sentence overlap with the 256 natural role-training items. Paraphrase and pretraining overlap remain unknown.
- Froze the questions, exact-answer and candidate-contrast measures, patient subset, two wordings, active/passive stratification, six checkpoint identities, and decision rule in [PROTOCOL.json](PROTOCOL.json).
- Prepared a [blind human-review sheet](BLIND_HUMAN_REVIEW.csv), [review instructions](REVIEW_GUIDE.md), a [freeze gate](freeze_review.py), and a [checkpoint evaluator](evaluate.py). The evaluator correctly refuses to run while no reviewed frozen set exists.

**Current gate:** 0/32 new challenge items and 0/75 EWT items have independent human verification. No checkpoint result exists for the new challenge. Calling the candidate set “human verified” or claiming a natural-language repair now would be inaccurate.

## Decision after review

If the two natural role-trained starts beat their matched source checkpoints and the copy control on exact **both-roles-correct** accuracy for passive as well as active sentences, under both question wordings, investigate more constructions and model sizes. A candidate-score improvement by itself is insufficient. If the gain does not survive, use the separated actor, other/patient, voice, and wording outcomes to decide whether the main failure is positional shortcuts, voice change, or answer generation before modifying training.

Even a positive 32-pair result would be an initial confirmation with wide uncertainty. A paper-level generalization claim would need a larger independently verified set and ideally another model family. The present protocol contains no within-event role-reversal sentences; those would be a separately reviewed extension.
