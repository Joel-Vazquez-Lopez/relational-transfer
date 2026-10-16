# Source-state directionality: causal identifiability protocol v0.1

**16 October 2026. Proposal only; no model inference or mechanism claim.** Independent design for Brain-O / Comput / Sentry / Armaggedon review. The existing E051–055 near-perfect pointer is a grammar-aware copier: trained source states can be zeroed without loss. This design must not reproduce that confound.

## Primary estimand
For frozen native answer function f and target world t, donor world d, source site s:
Delta_role = log p_f(y_d | do(h_s(t) := h_s(d)), q_t) - log p_f(y_t | do(h_s(t) := h_s(d)), q_t).
Report this **alongside** the same log-odds change for sham donors, random-source donor states, and donor states with unchanged roles. This is an intervention on a model state, not a guaranteed causal variable for semantic role; off-manifold effects are possible.

## Critical crossed 2x2x2
- Same queried predicate/event identity and same queried role across target/donor.
- Flip participant-to-role assignment (actor A / patient B versus actor B / patient A).
- Independently cross voice (active/passive) and surface mention order (A-first/B-first) where grammatical validity permits.
- Keep participant name multiset, total token budget, event count and event selector output matched; verify these with a *pre-patch* gate. If an axis cannot be independently varied in the grammar, report **non-identifiable** instead of claiming it was controlled.

**Source and donor selection:** same requested event and predicate/tag identity, different gold role answer, distinct sentence realization; choose matched and mismatched lexical controls. Exclude worlds where target/donor answers are equal. Maintain a paired, world-clustered sampling ledger.

## Patch conditions
1. Native unpatched target/donor.
2. Patch source event-participant state at candidate layer and position, with event selector frozen/matched.
3. Patch an irrelevant event's corresponding state (sham).
4. Patch same-role-answer donor state (answer-invariant control).
5. Patch random/untrained model source state (trained-source specificity).
6. Patch source state with participant identities preserved but role mapping reversed (key test).
7. Optional causal interchange at a narrower subspace, **only if** subspace learned on separate train worlds and frozen before test.

**Must measure:** final answer token probabilities, answer correctness, donor-following fraction, unchanged-event selection fraction, and invalid-format outputs. Auxiliary event-index changes alone are insufficient. A donor-following answer is not a success if donor names leak via the patched state and lexical copier explains it.

## Baselines / falsifiers
- Source-text grammar parser; first/last name; voice-aware subject/object parser; name-frequency and nearest-name copier; fixed verb-tag table.
- Random-source and zero-source controls; equal-parameter sham bridge if a new readout is trained.
- Native answer-head-only matched retraining if rescue is claimed.
- Tokenization and lexical identity perturbations; novel event counts and verb/tag combinations.
- Original source hidden-state role probe with nested heldout splits is diagnostic only, not a mechanism.

## Decision
- **Positive functional evidence (bounded):** role swap causes answer to track donor mapping significantly more than all matched shams, with event selection fixed, survives name/voice/order crossing and novel grammar; native answer pathway shows effect.
- **Negative for source-state binding claim:** random/zero source indistinguishable, grammar copier matches, or only auxiliary event head changes.
- **Inconclusive:** native answer competence is near chance, patch is off-manifold, crossing impossible, insufficient independent worlds, or effect only appears in a new engineered readout.

## Scientific priority
This synthetic intervention cannot replace authentic QA-SRL Gold evaluation on untouched pretrained model families. It is a mechanism-disambiguation side track. Brain-O owns implementation; no experiment authorized by this file. Reviewers should first decide whether the native answer head's ~33% competence makes donor-following estimable at all; if not, stop and do not over-interpret null effects.
