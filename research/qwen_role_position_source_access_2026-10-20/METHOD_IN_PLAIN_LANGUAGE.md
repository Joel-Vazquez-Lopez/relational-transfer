# What this experiment asks

## One changed relationship, two unchanged answers

An example source pair is:

| | Source A | Source B |
|---|---|---|
|Passage|Grace sent Helen to Jack. Kate waited nearby.|Kate sent Helen to Jack. Grace waited nearby.|
|Sender|Grace|Kate|
|Sent person|Helen|Helen|
|Destination person|Jack|Jack|

The model receives each passage separately with each of the three questions. Every answer is generated freely over the whole vocabulary. We also save the unrounded first-token logits for all four people. The example illustrates the design; the actual full outputs are in NATIVE.jsonl.

A greater score for Kate can reflect a changed sender relationship, a general preference for Kate, or another response to the edited passage. We therefore measure how much the Kate-versus-Grace difference changes on the sender question **relative to its changes on the other two questions**. That difference is R. The names remain present in both passages; the two unchanged answers cannot uniquely determine the changed one from the four-person candidate set.

This is an output-response interaction computed by the analyst across separate prompts. It is not a semantic probability, a hidden variable assumed to exist in the model, or a repair of its generated answer. Logit differences are log odds between two token alternatives conditional on a particular prompt; they are not odds between complete linguistic interpretations.

## Which states carry the response?

At three fixed boundaries, we recombine the original passage's states with the changed passage's states. One hybrid receives the changed source region and original subsequent question/assistant states. The complementary hybrid receives the original source region and changed subsequent states. The new states are then processed by the remaining native model blocks.

Both hybrids can contain inconsistent information. This is a sensitivity experiment on complete residual regions, not a clean replacement of a semantic variable. Question states can already contain source information. A late preference for subsequent states does not establish that linguistic computation has ended.

We compare each hybrid's three-question response contrast and normalize by the original source-pair contrast R. The resulting S_R is positive when the changed-source hybrid carries more of the contrast; negative when the changed-suffix hybrid does. Values beyond ±1 are retained. Small native denominators are excluded by the fixed prerequisite, not clipped.

## What would make a pattern useful?

Four fixed predictors compete on combinations of names, question wordings and constructions that were not used to estimate their parameters:

1. Queried-role identity.
2. Surface mention position.
3. Generic depth schedule.
4. Native output scores, entropy and token distances.

We ask whether the role predictor materially outperforms all three rivals in every held-out construction/name set. This is a test of a specific predictive regularity. Even a success would leave syntax-aware copying and future-answer-table implementations possible.

## What the controls establish

Exact state hashes verify what was replaced. Identity/joint-donor and endpoint controls verify native execution and region alignment. The meaning-preserving waited→stood sham tests sensitivity to a small irrelevant change. Full generated names, unchanged-role answers and complete three-question cases remain separate from graded logits. All failures are retained.

Three measured boundaries, one verb, two lexical/question sets and one Qwen checkpoint cannot provide a complete process map or a model-family law. The separate Llama prerequisite failure remains a failure. The protocol states these limits before Qwen outcomes and requires stopping when its prerequisites fail.
