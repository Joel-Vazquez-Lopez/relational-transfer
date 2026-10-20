# Post-lock order-compliance sensitivity plan

Declared after the source/items/protocol lock and before any output-order inference. This does **not** change the frozen primary analysis or parser.

For every same-sentence AP/PA pair, report separate denominators:

1. Both generated responses have semantically parseable, nonduplicate `Actor` and `Patient` labels.
2. Both are parseable **and** AP emits `Actor; Patient` while PA emits `Patient; Actor`, as requested.

Within each denominator, count invariant actor/patient label assignments, changed assignments, identical emitted name sequences, and role-correctness gained/lost after the requested order changes. Report by world, arm, and construction. A model that ignores the PA order instruction can have invariant role labels without passing the second denominator; do not treat that as evidence of semantic role binding. Keep the all-output frozen primary estimates and invalid/malformed counts visible. This is a descriptive sensitivity analysis, not a new success gate or causal mechanism claim.
