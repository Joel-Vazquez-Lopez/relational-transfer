# Semantic specificity gate for relational interventions — 16 Oct 2026

**Independent scientific proposal, not an executed experiment.** Motivation: Compass Investigation 19 reports 690/1128 output changes under instruction edits vs 302/1128 under operand edits, but only 270/1128 vs 457/1128 semantically correct new answers; irrelevant B-cell edits change outputs 104/546 times. Brain-O E051–055 finds high answer rescue with zero/random source representations. Together these falsify the assumption that answer changes, even after carefully chosen patches, necessarily reveal semantic-role transport.

## Distinguish three intervention effects
1. **Volatility:** output changes at all (can arise from lexical perturbation, off-manifold patch, output interface).
2. **Semantic tracking:** output moves specifically to the *correct counterfactual answer*, not merely any changed answer.
3. **Causal specificity:** correct tracking is selectively higher for a role-relevant patch than matched role-irrelevant patches, random/zero-source controls, and matched lexical donor shams.

## Matched factorial contrast
Construct target/donor pairs with identical queried role and event identity and same participant multiset, crossing:
- semantic role assignment changes vs unchanged;
- source position relevant to queried event vs irrelevant event;
- lexical identity preserved vs changed while gold remains fixed;
- voice and mention order balanced when independently realizable.

If the grammar cannot cross an axis without changing another, explicitly label non-identifiable. Before patching, require native competence and target/donor both-correct subset, with full-population results also retained to avoid selection bias.

## Metrics, grouped by independent world
- Change rate P(pred_after != pred_before)
- Correct counterfactual tracking P(pred_after = gold_donor) on role-changing pairs
- False-positive volatility P(pred_after != pred_before) on same-gold irrelevant patches
- Specificity contrast = counterfactual tracking minus matched-sham donor tracking, with world-clustered uncertainty
- Answer log-odds of donor vs target gold; event-index stability; output validity
- Stratify original target correct/incorrect, donor correct/incorrect; do not count any changed answer as a success.

## Adversarial pass/fail
- **Potential positive:** donor-gold tracking exceeds matched shams, with native answer accuracy adequate, survives lexical/name/voice/position crossing and novel structures, and depends on trained source state more than random/zero state.
- **Negative for semantic specificity:** high change rate with low donor-gold tracking, irrelevant edit volatility comparable to relevant edit, random-source equivalent to trained-source, or a grammar-aware copier matches effects.
- **Inconclusive:** low native competence, no independently realizable role reversal, or patch induces off-manifold distribution shift.

## Ownership and implementation
Compass owns Investigation 19; Brain-O owns selector/rescue models; Comput/Sentry independently audit causal and shortcut controls; Armaggedon sets priority. This note requests no new runs or changes to their branches. If accepted, predefine the paired sampling protocol, checkpoint hashes, intervention sites and pass/fail thresholds before testing.
