# Next experiment: learn a procedure rather than print one

The first pilot rules out a simple formulation: moving a printed plan ahead of the answer did not make the scratch model recombine operations or reliably bind the second participant. The next test must alter **what is learned**, with a fresh data bank. Do not reuse the examined mixed items as a confirmatory test.

## Core hypothesis

Supervising each goal dependency as an independently checkable decision, and training the model to execute its own derived state, will transfer better to unseen dependency combinations than supervising a single free-form trace. The alternative is that any gain comes from extra labels, extra updates, or a symbolic executor doing the work.

## Frozen benchmark before training

Use a new deterministic generator seed and new test names/verbs where tokenization permits. Generate two banks:

1. **Learnability bank:** every two-operation combination appears in training; hold out fresh lexical instances and a syntactic realization. This checks whether the architecture can solve ordinary two-hop binding. Do not interpret mixed-composition scores until direct and structured systems clear a preset in-domain floor (suggest **≥80% exact answers and ≥90% operation validity**).
2. **Composition bank:** all primitive operations are trained, including as independently queried one-step tasks, but specified two-step *dependencies* are absent from training. Test only newly generated held-out combinations. Preserve equal answer frequencies, voice balance, distractor placement, and full sentence/query disjointness. Audit every generated item with the surface-only parser and inspect heuristic baselines.

The original benchmark can remain a development/debugging bank. Any threshold or architecture selected using it must be frozen before the new test bank is scored.

## Matched training arms

All arms need the same initial checkpoint, example order, updates, approximate target-token count, and access to primitive operation labels. If an arm receives more labels or more data, add a matched extra-supervision control and name the imbalance.

| Arm | Training target | Why it is needed |
|---|---|---|
| Direct | Answer, plus token-matched nonprocedural targets | Does extra output alone help? |
| Forward trace | Operations and intermediate values in execution order | Familiar structured-reasoning comparator. |
| Goal-first trace | Goal, required operation, prerequisites, execution, answer | Tests order of the same substantive information. |
| Factorized goal state | Separate operation decisions for each clause, then a state linking them to execution | Tests whether the model learned to *construct* a procedure, not copy a whole trace. |
| Oracle state + learned executor | Correct operations supplied, entity binding still learned | Separates planner limits from executor limits; an upper-bound diagnostic, not an end-to-end method. |

Do not interpret a factorized-versus-free-form gain unless its extra primitive supervision is matched. A deterministic symbolic executor may be included as a transparent system baseline, but it cannot establish that the neural model learned execution.

## Evaluation and stopping gates

- **Plan validity:** are the selected operations and their dependencies correct before any entity answer appears? Score each primitive, their combination, and the full state separately.
- **Execution:** with a correct state, does the model bind the first and second participants? Report exact intermediate and final values, not only final accuracy.
- **End to end:** score answer accuracy using the model's *own* state. Compare with direct and forward arms on matched examples and multiple seeds. Include invalid-output rate.
- **State use:** replace one operation in the printed state while keeping the sentence and question fixed; record whether the answer changes in the predicted way. Include a semantically neutral edit control. Report state-following and gold-answer accuracy separately.
- **Transfer:** hold out dependency combinations and, later, an additional task family. Never call success on one fixed relational algorithm “learning to decompose arbitrary tasks.”

If in-domain execution stays below the learnability floor, stop before making a compositional-generalization claim and revisit architecture or representation. If an oracle state works but the model's own state fails, improve planner training. If the model's state is valid but execution fails, improve the state-to-execution objective. If both work on the composition bank, freeze a separate task-family transfer protocol; that is where the broader hypothesis begins to face a serious test.

## Novelty boundary

Backward chaining, decomposition graphs, and plan-supervised language-model training already exist. A publishable result would require a **well-isolated learned procedure-construction effect** on unseen dependency structure, robust to matched supervision and output-format controls, with state interventions showing functional use. The present pilot supplies a failure diagnosis and a benchmark scaffold; it is not that result.
