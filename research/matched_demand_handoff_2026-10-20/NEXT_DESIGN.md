# Preferred reformulation: isolate one relationship without answer elimination

Design only. No additional inference is authorized by this file or launched after the failed native gate. No novelty claim.

## What the audit changed

The previous three-person/three-role panel fixes a binary problem, but remains exhaustive: two role answers determine the third. Qwen's 36/36 complete assignment recovery survives removing the sender score row entirely. New observed role scores give 15/15 complete assignments despite only 2/15 correct native recipient answers. Neither result establishes separately accessible information for every queried relationship.

The preferred question is: **Does the model's evidence for one queried relationship respond to a controlled change in that relationship when all other queried answers remain the same?** This is narrower than matching whole trajectories and directly addresses an identified measurement failure.

## Smallest next native panel

Four distinct people, three queried event roles and one unrelated participant. Each person appears exactly once. Use two new single-token name quartets, active/passive forms, and distractor-first/distractor-last orders. For each stratum, author a base assignment and three counterparts exchanging one event participant with the distractor. This gives 2×2×2×4=32 source cases and three questions per case:96 native responses. The question strings should reuse the previously validated grammatical prompts, not the failed label `recipient`:

- Who sent someone to another person?
- Who was sent to another person?
- Who was someone sent to?

Illustration only, not fresh held-out stimuli:

|Case|Passage|Sender|Sent person|Destination|
|---|---|---|---|---|
|Base|Alice sent Bob to Carol. Dave waited nearby.|Alice|Bob|Carol|
|Sender changed|Dave sent Bob to Carol. Alice waited nearby.|Dave|Bob|Carol|
|Sent person changed|Alice sent Dave to Carol. Bob waited nearby.|Alice|Dave|Carol|
|Destination changed|Alice sent Bob to Dave. Carol waited nearby.|Alice|Bob|Dave|

Every contrast changes one queried answer and preserves the other two. Counterbalancing distractor order prevents a universal recent-name account; active/passive controls distinguish role identity from fixed mention position. These controls do not uniquely distinguish a semantic representation from a syntax-aware parser. The new irrelevant participant may itself induce competence failure, which must be reported rather than repaired after outputs.

## Measurements and alternative predictions

Keep per-question unconstrained full generation as the native competence screen. Record all four candidate-name scores. The primary descriptive evidence is the target-name contrast for the changed question across its counterpart passages, with two invariant-question contrasts as controls. Scores on different prompts are not a native joint posterior.

Report independent per-question winners, a fixed source-column calibration, and the maximum among24 injective three-of-four assignments separately. Never identify perfect composite decoding with native answering or complete internal representation. With four candidates the column-bias invariance of the old exhaustive assignment criterion no longer holds, because the unused candidate can change. This new salience confound must be explicitly controlled; the mathematical row-necessity fix is not a complete experiment by itself.

`nonexhaustive_certificate.py` proves why a zero-information query no longer permits a unique complete assignment: the omitted row has at least two unused names to choose from. However, shared exclusion of the distractor could still assist decoding. The role-isolated counterfactual, not decoder accuracy alone, is the essential test.

Competing accounts:

1. **Exhaustive completion:** apparent full assignment recovery comes from two reliable questions plus elimination. It cannot by itself identify the changed answer when those two answers are held fixed and two candidates remain.
2. **Shared candidate salience:** a question-independent name/position preference drives the apparent loss. It predicts coordinated candidate shifts across questions and improvement from a correction transferred across lexical sets; it need not predict query-specific target changes.
3. **Question-specific relational evidence:** the changed question's target contrast tracks its counterpart across both voice and distractor order while invariant questions retain their answers. This supports role-sensitive evidence at the measured output; it still does not identify an internal algorithm or prove causal use.

Freeze exact stimuli, definitions, scores, numerical thresholds and stopping rules before outputs. Native competence must pass before a causal stage is designed or run. Do not select fresh pairs because they look clean after a screen without reporting all rejected documents.

## Developmental continuation, conditional on a valid assay

Only after the native panel passes, reuse a separately frozen whole-prefix/question-state recombination design on complete single-token names. Compare the isolated changed role and the two invariant roles across declared depths. Whole-joint donor swaps, identities, architecture-forced endpoints and lexical controls must pass. Logit changes are causal sensitivity of the declared intervention, not proof of a discrete stage. A low-rank semantic map, a syntax-aware copying program and output preparation can still agree on some outcomes; declare where they remain indistinguishable.

Llama3B is currently the feasible local checkpoint. Qwen's earlier memory-pressure failures remain an operational restriction; a cross-model inference would require a separately feasible resource plan. No cloud, paid API or model download is needed for the96-prompt native pilot. Family-level claims require more than one checkpoint per family. Additional verbs and natural corpus examples are later generalization gates, not factors to add to the first identifiability test.

## Paper direction

A plausible project is **measurement of relational access under readout and answer-completion constraints**. The existing prospective24-family competence-selection result supplies a stronger replication base than these two-triple exploratory panels. The current audit explains why another apparently perfect metric is insufficient. A credible paper needs a prospective non-exhaustive validation and a careful comparison with existing calibration, structured QA, probing and binding research. An elementary identifiability correction or a larger score alone is not a publishable novelty claim. The long-term linguistic-process objective remains intact; the assay must first measure the relationships it purports to track.
