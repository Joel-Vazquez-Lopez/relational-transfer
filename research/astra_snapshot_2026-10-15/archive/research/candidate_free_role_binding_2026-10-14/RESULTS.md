# Candidate-free natural role-binding development screen

14 October 2026. This is exploratory development, not independent confirmation. The original protocol is preserved in `PROTOCOL_ORIGINAL.json`; `AMENDMENT_001.md` records a runner/import repair before any training or evaluation. The amended frozen `PROTOCOL.json` has SHA-256 `33b5a5d31184d21829883f8346112b500c42580aa52d6e12485dca63572b411e`.

## Question and design

Can a typed requested-role cue improve a Qwen2.5-0.5B-Instruct model's **direct actor/patient answers when no answer candidates are supplied**? We compared `Requested role: ACTOR/PATIENT` with `Requested role: NONE`. Both arms used the same 160 QA-SRL Bank training events, original human-authored role questions, gold span targets, role-first output format, 160 optimizer updates, target-token count (4,238), learning rate, and source adapter within each start. A question-only classifier trained on the training items supplied the typed role at evaluation; it was correct on 191/200 questions. We tested two matched starts on 100 sentence-disjoint Bank dev events, asking both actor and patient questions for each event.

## Results

| Start | Arm | Both answers correct /100 | Exact answers /200 | Both printed roles correct /200 | Answer follows requested printed role /200 | Requested printed role exact /200 |
|---|---|---:|---:|---:|---:|---:|
| fresh1 | typed | 31 | 111 | 82 | 166 | 119 |
| fresh1 | neutral | 11 | 88 | 79 | 112 | 118 |
| fresh3 | typed | 21 | 93 | 69 | 156 | 105 |
| fresh3 | neutral | 13 | 88 | 81 | 112 | 118 |

Typed minus neutral was **+20/100 both-correct and +23/200 exact** in fresh1, but only **+8/100 and +5/200** in fresh3. The frozen exploratory screen required at least +10/100 and +20/200 in **both** starts without more than 10/200 loss in printed-role correctness. **It failed**: fresh3 missed both answer thresholds and its both-printed-role count fell by 12/200. Descriptive sentence-paired bootstrap 95% intervals (20,000 resamples; analysis performed after the frozen screen) were +10 to +30 and +9 to +36 for fresh1's two answer differences, versus −1 to +17 and −10 to +19 for fresh3. These are item-sampling intervals on the same development set, not independent-model uncertainty.

The typed cue improved the *routing* of the answer to the model's printed role fields in both starts: 166 versus 112 and 156 versus 112. It did **not** consistently improve the correctness of those fields. Selecting the requested printed field by a deterministic rule would yield 119/200 and 105/200 exact in the typed models, compared with 118/200 in both neutral models. Thus the cue helped answer-field agreement more reliably than underlying role extraction. Wrong answers that matched the opposite printed field were 27 versus 74 in fresh1 and 37 versus 68 in fresh3. These are diagnostic counts, not a new success criterion.

## Interpretation and decision

The result is evidence that candidate-free role answering involves separable **participant extraction** and **answer routing**. A typed cue can improve routing, but it has not passed a two-start screen for an end-to-end natural-language repair. The next method should target reliable event/participant/role extraction under changes of voice and participant order, and compare it with equally supervised controls. A cue-only scale-up is not justified by this screen.

All 100 evaluation sentences put the automatically mapped actor before the other participant. The semantic actor/patient conversion has no independent human review, and some QA-SRL objects may not be semantic patients. The dev corpus was used in earlier broad development. The two starts share one 0.5B model family and training source. This study cannot establish generalization to actor-last syntax, new constructions, different pretrained models, or a publication-level method. Those claims require a genuinely new, prediction-blind, human-verified and position-balanced evaluation.

`ANALYSIS.json` contains all frozen scores; `EVAL_*.json` contains every raw response and parsed score; `TRAINING.json` and saved adapters preserve the four matched models. No sealed human-review challenge was inspected or used here.
