# Cross-corpus role check and copy control: exploratory results

**Date:** 12 October 2026. **Model:** local Qwen2.5-0.5B-Instruct. **Status:** cross-corpus behavioral result with provisional labels, not a confirmed general repair.

## What was tested

The earlier continuation with 256 QA-SRL Bank 2.0 natural role examples improved scores on held-out QA-SRL sentences. I tested whether that result also appears on a separate corpus: the [Universal Dependencies English Web Treebank, release 2.18](https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.18). Its [nominal-subject](https://universaldependencies.org/u/dep/nsubj.html) and [object](https://universaldependencies.org/u/dep/obj.html) annotations provide **syntactic proxies** for actor and other direct participant. They are not gold semantic-role labels. The corpus is licensed CC BY-SA 4.0; a copy of its [license](data/LICENSE.txt) is stored with the data.

Before running any model on this corpus, I froze all 75 automatically qualifying test events, the extraction rule, the four checkpoint identities, and the evaluation questions in [PROTOCOL.json](PROTOCOL.json), [EVENTS.json](EVENTS.json), and [EVAL_PROTOCOL.json](EVAL_PROTOCOL.json). The selected sentences have 50 source-document IDs. They are disjoint by exact normalized sentence from the selected QA-SRL training, development, Bank-test, and Gold examples. There were no post-result item exclusions. The item-level scores are saved in the four checkpoint JSON files linked below.

The **primary measure** asks the original actor and direct-object questions, scores both candidate spans, and checks whether the two questions relatively favor the appropriate spans (contrast \(D>0\)). It measures candidate discrimination, not correct generation. The sentence-omitted condition tests how much of that distinction survives without the sentence. A separately frozen paraphrase probes question-wording sensitivity. Greedy answers are also scored by strict normalized exact span match.

## Cross-corpus outcomes

All counts below are out of the same **75** EWT events.

| Start and checkpoint | Original contrast | Sentence omitted | Paraphrased contrast | Exact actor | Exact object | Both exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh1 source | 56 | 43 | 69 | 16 | 2 | 0 |
| Fresh1 + natural role examples | **75** | 52 | **75** | **37** | **30** | **15** |
| Fresh3 source | 56 | 45 | 72 | 16 | 2 | 0 |
| Fresh3 + natural role examples | **74** | 58 | **75** | **41** | **32** | **19** |

On the primary contrast, the first matched start changed **19 incorrect items to correct and 0 correct items to incorrect**; the second changed **18 to correct and 0 to incorrect**. Mean contrast margin rose from **2.40 to 6.02** and **2.79 to 6.94**, respectively. With the sentence omitted, mean margin rose only from **2.00 to 2.22** and **2.36 to 3.38**. This pattern supports an increased contribution from the sentence under these prompts. The paraphrase was easier for the source checkpoints already (69 and 72 correct), so its smaller gains also show that wording changes the apparent difficulty.

A document-cluster bootstrap, resampling the 50 EWT source IDs while holding these two starts fixed, gives a mean primary accuracy gain of **24.7 percentage points** with an approximate 95% interval of **14.9 to 35.4 points**. This interval covers this selected corpus and these checkpoints; it does **not** include uncertainty over model families, training data choices, or label validity. The two starts share the same EWT items and natural training set.

## Matched diagnostic without role questions

After seeing the first EWT adaptation result, I froze a diagnostic control in [the copy-control protocol](../llm_role_copy_control/PROTOCOL.json). It starts from the same Fresh1 adapter and uses the same 256 natural sentences, the same two target answer spans per step, 256 updates, and the same learning rate. Its question simply supplies the answer phrase and asks the model to copy it. This gives natural-language and short-span answer practice without requiring a role decision. The prompts are different, so it is **not** a perfectly matched causal isolation of role supervision.

| Fresh1 checkpoint | EWT original contrast / 75 | EWT sentence omitted / 75 | QA-SRL Bank contrast / 130 | EWT exact actor / object / both |
| --- | ---: | ---: | ---: | ---: |
| Source | 56 | 43 | 97 | 16 / 2 / 0 |
| Copy control | **54** | 47 | **86** | 5 / 2 / 0 |
| Natural role examples | **75** | 52 | **121** | 37 / 30 / 15 |

Relative to the copy control, natural role training fixes 21 EWT primary items and loses none. The copy control therefore does not explain the measured gain as generic exposure to these natural sentences, the answer spans, or copying practice alone. It still does not isolate the effect of role labels from the specific role-question wording, and the control was designed after one EWT result had been inspected. Its direct-answer check was added after its contrast result and is diagnostic.

## Interpretation and limits

This is **encouraging evidence that the natural role adaptation transfers behaviorally to a different text source**. It is stronger than seeing a gain only within QA-SRL. The exact-answer results also move in the same direction, although jointly correct actor and object spans remain low (15/75 and 19/75). The original candidate contrast reaches a ceiling on this selected corpus, so it cannot measure further improvement here.

Several constraints still limit a paper-level claim:

1. EWT annotates syntax, not semantic roles. A source-only, prediction-blind [AI-provisional semantic audit](SEMANTIC_AUDIT_AI_PROVISIONAL.csv) classified all 75 grammatical subject/object pairs: 42 clear agents, 29 non-agents, and 4 unsure; only 14 clear semantic patients, 55 non-patients, and 6 unsure under the audit's narrow definition. These judgments are **not independent human verification**. They sharpen the interpretation: the 75-item gain is primarily a grammatical subject/object result, not a demonstrated agent/patient gain. A separate [HUMAN_REVIEW.csv](HUMAN_REVIEW.csv) is still unfilled. Some spans are awkwardly tokenized or omit part of a phrase.
2. The selection requires a root verb with a direct nominal subject and object. It therefore favors relatively straightforward **active** clauses. Success could partly reflect subject/object order; it does not yet show robust passive, cleft, or other construction transfer in authentic text. The earlier synthetic patient-cleft retention result is separate.
3. Both starts are from **one 0.5B model family**, sharing the same natural examples and EWT items. Neither pretraining exposure to the corpus nor independence across model families is known.
4. The natural role questions match those used during evaluation. The copy control argues against simple answer-format practice but does not prove a general training principle independent of question phrasing. The much easier paraphrase for source models shows this sensitivity directly.
5. EWT was chosen after the positive QA-SRL results, although its item list and metrics were frozen before EWT inference. The copy control was chosen after the first EWT result. Treat both as exploratory evidence.

The immediate next gate is independent human review of **all 75** EWT labels while blinded to model outputs. A separate [new active/passive candidate set](../llm_role_challenge/BLIND_HUMAN_REVIEW.csv) has been prepared without consulting item-level model predictions. It also requires independent review before checkpoint inference. A larger or different model should be tested only with metrics that avoid the present candidate-score ceiling. Internal-state claims require separate causal tests.

## Untouched larger-model feasibility check

I also ran the locally available **untouched Qwen2.5-1.5B-Instruct** on the same 75 EWT events, under a [fixed baseline protocol](qwen15/PROTOCOL.json). It scored **72/75** on the original candidate contrast, **56/75** with the sentence omitted, and **73/75** under the paraphrase. Exact direct answers were **15/75 actor**, **10/75 object**, and **3/75 both**. Thus candidate contrast is near ceiling even without adaptation at this size, while direct generation still has substantial headroom. This is an observational baseline, **not** a second-model training-repair result; it does not compare matched adapted and unadapted 1.5B checkpoints. Its [item-level outputs](qwen15/EVALUATION.json) are saved.

## Untouched 0.5B diagnostic baseline

After the adapted EWT results were known, I froze and ran an [untouched Qwen2.5-0.5B-Instruct baseline](qwen05/PROTOCOL.json) on the same 75 items. It scored **57/75** on the original candidate contrast, **48/75** with the sentence omitted, and **72/75** under the paraphrase. Direct answers were **16/75 actor**, **2/75 object**, and **0/75 both**. These are close to the two source-adapter outcomes (56/75 primary, 16/75 actor, 2/75 object, 0/75 both), so the earlier source adapters do not explain the large observed adaptation gain on this diagnostic set. This baseline is post-result and does not create a new independent confirmation. The [item-level outputs](qwen05/EVALUATION.json) are saved.

## Audit trail

- Dataset and selection: [original EWT test CoNLL-U](data/en_ewt-ud-test.conllu), [license](data/LICENSE.txt), [frozen events](EVENTS.json), [selection protocol](PROTOCOL.json), [evaluation protocol](EVAL_PROTOCOL.json), [review sheet](HUMAN_REVIEW.csv).
- Item-level outputs: [Fresh1 source](fresh1_source.json), [Fresh1 adapted](fresh1_natural.json), [Fresh3 source](fresh3_source.json), [Fresh3 adapted](fresh3_natural.json).
- Control: [training trace](../llm_role_copy_control/TRAINING.json), [candidate evaluations](../llm_role_copy_control/EVALUATION.json), [direct EWT answers](../llm_role_copy_control/DIRECT_EWT.json).
- Scripts: [selection](../llm_role_independent_eval.py), [four-checkpoint evaluation](../llm_role_independent_run.py), [copy control](../llm_role_copy_control.py), [copy direct diagnostic](../llm_role_copy_direct.py).
- Larger-model baseline: [fixed protocol](qwen15/PROTOCOL.json), [item-level outputs](qwen15/EVALUATION.json), and [script](../llm_role_independent_qwen15.py).
- Untouched 0.5B baseline: [fixed protocol](qwen05/PROTOCOL.json), [item-level outputs](qwen05/EVALUATION.json), and [script](../llm_role_independent_qwen05.py).
- Source-only semantic audit: [provisional annotations](SEMANTIC_AUDIT_AI_PROVISIONAL.csv) and [reproducible source-only script](../llm_role_ewt_semantic_audit.py). These labels await human verification.
