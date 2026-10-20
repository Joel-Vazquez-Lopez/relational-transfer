# Typed requested-role conditioning: fresh natural-sentence result

**Completed 13 October 2026.** This is an adaptive development experiment with a new, frozen QA-SRL Bank item set. The independent human-review queue was not opened or evaluated.

## Comparison

The two arms continue the same Qwen2.5-0.5B source adapter within each of two starts. Both see the same 160 training items, actor and patient questions in both candidate orders, 160 optimizer updates, identical `Actor; Patient; Answer` targets, and **8,476 supervised target tokens**. The only experimental difference in the input is `Requested role: ACTOR/PATIENT` versus `Requested role: NONE`. At test time, a question-only character-gram classifier trained on the 160 training items supplies the typed tag; no gold requested-role tag is read from the fresh2 set. The [protocol](PROTOCOL.json) and 40 sentence-disjoint [fresh2 items](SPLITS.json) were fixed before training or inference. The actor/patient conversion from human QA-SRL questions and spans is automatic and has not been independently human-reviewed.

## Frozen result

| Start | Training input | Exact final answers /160 | All four answers /40 | Both printed roles right /160 | Final answer agrees with requested printed role /160 |
|---|---|---:|---:|---:|---:|
| fresh1 | Typed role | **128** | **24** | **126** | **153** |
| fresh1 | Constant tag | 84 | 2 | 121 | 88 |
| fresh3 | Typed role | **125** | **23** | **125** | **151** |
| fresh3 | Constant tag | 86 | 3 | 123 | 89 |

The prespecified screen [passed](ANALYSIS.json) in **both** starts: gains of +44/160 and +39/160 exact answers, and +22/40 and +20/40 all-four-correct items, without a loss in role-field accuracy. A descriptive paired bootstrap over the 40 sentences gives exact-answer gain intervals of **+18.1 to +36.2 percentage points** and **+15.6 to +33.1 points** for the two starts. The starts share all items and a model family; these intervals do not create independent replications.

The effect appears in both role questions and both answer-candidate orders. For fresh1, typed versus constant exact scores were actor/order0 **31 vs 15**, actor/order1 **28 vs 27**, patient/order0 **37 vs 27**, patient/order1 **32 vs 15**, each out of 40. Fresh3 shows the same broad pattern (32 vs 16, 29 vs 26, 35 vs 28, 29 vs 16). The constant arm's large order differences are consistent with a candidate-position shortcut; the typed condition reduces them.

The typed arm's own answer nearly reaches deterministic readout from its generated fields: **128 versus 129** and **125 versus 128** exact. On the same fresh2 bank, the older answer-only checkpoints scored **100/160, 101/160** and **5/40, 6/40**; the older role-first checkpoints scored **86/160, 82/160** and **3/40, 1/40**. Those older models differ in output target and token count, so the constant-tag arm is the matched causal comparison.

## Interpretation and limits

This is a substantial **behavioral repair of final-answer routing** in one small pretrained model family: giving the model an explicit requested semantic role during training and inference makes its final answer use the role fields it can already often generate. The tag classifier was correct on all **80/80** fresh2 questions, including **37** exact question strings absent from its training questions. These QA-SRL questions are highly templatic: the simple rule *“`someone` is the final word”* also classifies all 80. The classifier is not itself a novel linguistic achievement.

This result does not establish that the model learned a general procedure, an internal causal role state, or robustness to unfamiliar syntax. It is one corpus, 40 fresh sentences, two related starts, candidate-supported answers, and automatically mapped role labels. A conference claim needs independently checked semantic labels, new syntactic forms and question wordings, a second model family, and comparison to structured prediction or decomposition methods already in the literature. The next useful stress test is whether the fixed typed-goal checkpoint maintains the gain when question wording and sentence form change; that should be designed and frozen before inference.
