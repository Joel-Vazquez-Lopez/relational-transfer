# COMPUT-025 — Input-only text classifier predicts 85.22% SemEval relation orientation

**16 October 2026 | independently executed via connected GitHub original-format TRAIN blob; exploratory, not pretrained LLM inference.**

## Original source and classification contract
Original-format training text from third-party public GitHub mirror `nb20593/Semeval-2010-task-8-Relational-extraction`, immutable Git blob `0797b23e4907e12e26739a79a454dfae9bb71c13`; `TRAIN_FILE.TXT` has 8,000 rows, including 6,590 with one of nine **oriented human relation labels** and 1,410 `Other`. The official ZIP's bit-for-bit identity with this mirror is not yet proven. **No official TEST was inspected**. Input features use only the sentence with e1/e2 spans marked. The binary orientation classifier does **not** receive gold relation *type*, direction, ID, comment or source gold at prediction time; it DOES use orientation labels from 4/5 human training data and is evaluated only on the known non-Other positive subset. This is **NOT a complete 19-way task, not zero-shot, and not a fair untrained LLM baseline**.

## Real source-connected five-fold results

Identical **untagged** sentence text grouped in the same fold (25 repeated extra occurrences; original **document** groups not established). Binary multinomial naïve Bayes, add-one smoothing, binary lexical features per item, empirical train priors. Four exploratory variants; **not preregistered**, feature comparison partly post hoc.

| Predictor | Correct / 6,590 | Accuracy |
| --- | ---: | ---: |
| Always say `(e1,e2)` | 3,588 | 54.45% |
| Marked entity words only | **5,079** | **77.07%** |
| Sentence bag of words, entity markers erased | **4,705** | **71.40%** |
| Local words before/after and between marked entity spans; **do not feed span contents** as features | **5,464** | **82.91%** |
| Combined: entity words + bag + lexical local context | **5,616** | **85.22%** |
| For context, COMPUT-023 gold-relation-TYPE oracle majority direction | 4,819 | 73.13% (PRIVILEGED, not same input contract) |

Five held-out fold accuracies for combined model: **85.28%, 85.36%, 85.47%, 84.49%, 85.51%**. Per-label direction recall: e1→e2 **85.65%**, e2→e1 **84.71%**. Relation-type heterogeneity is large: combined predicts `Entity-Destination` 845/845 (underlying type imbalance **844:1**), but `Product-Producer` only 477/717 and `Message-Topic` 453/634. Local-context features exclude the *contents of marked entity spans* but a name may appear again in surrounding sentence context.

**Scientific meaning:** a competent supervised lexical/position learner is an important comparator for future pretrained LLM outputs. High SemEval orientation accuracy on this old source is not sufficient to show compositional event-role understanding, and this control does not prove any particular LLM uses it. It is not a lower bound a zero-shot untouched LLM must beat: we used training labels and an already known positive-relation subset. It also does not validate performance on unseen documents, because groups are duplicate sentences, not verified documents.

## Reproducibility

Portable code `comput025_text_direction_nb.js` and source-only ZIP `COMPUT_025_SEMEVAL_TEXT_ONLY_DIRECTION_BASELINES.zip` available to research lead. **Six local Node fixture assertions pass.** Execute Node 22 on an authorized copy of `TRAIN_FILE.TXT` to reproduce all four five-fold results; no corpus text or model weights in ZIP. The exact original-source numbers above were computed using in-session connected GitHub JavaScript, while the portable script has *not yet been cold-run against the full original local bytes*. Keep these two receipt/evidence statuses separate.

## Recommended division and next

Comput: add direction-controlled, syntactic and lexical rivals to Nova's two-family pilot review; ensure per-type and directional recall. Nova owns original human QA-SRL span-scored Qwen/Phi inference (unexecuted), with its separately owned secondary joint question pilot. Mira and Sentry cover event-label semantics and positional bias; Brain-X scorer integrity. No duplicate LLM runner, paid inference, rights decisions, mergers or claim of mechanism. **Status: substantial learned shortcut comparator, no EUREKA.**
