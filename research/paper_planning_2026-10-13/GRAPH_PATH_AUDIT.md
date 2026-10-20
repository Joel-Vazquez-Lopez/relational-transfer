# New graph-path check: a concrete confound to isolate

**13 October 2026; exploratory, metadata-only.** I counted shared partners and simple three-edge paths between each withheld focal pair directly from the two frozen rewire manifests. The read-only [script](graph_path_audit.py) reproduces the table with standard Python. No model was trained or retested, and no sealed evaluation set was opened. Because this check followed the outcome reports, it is a **diagnostic for the next design**, not an independent discovery claim.

| Study | Families | High-minus-sham shared partners | High-minus-sham three-edge paths |
|---|---:|---:|---:|
| RQ1 matched rewire | 8 | +2 in every family | −5, −7, −4, −1, −1, −3, −5, −3 |
| RQ2 fresh-family repair | 8 | +2 in every family | −6, −5, −4, −5, −4, −2, −3, −2 |

Thus, in **all 16 families** where high-overlap rewiring is used, it also removes some three-edge paths between the withheld entities. These paths are another form of indirect relational support. Their loss is a specific alternative explanation for the lower transfer accuracy; the two features cannot be distinguished by the current rewire result. The existing data still establish that the *rewiring package* changes behavior and that ordered-role supervision attenuates its penalty. They do not establish which of these co-moving graph changes is responsible.

The next graph-only search should ask whether 10-node degree-four graphs can contrast shared partners while matching three-edge path count, triangle counts, endpoint exposure, focal pair, and edit distance. A quick random-walk search found such a **graph pair** with matched path and triangle signatures, but not under the existing two-swap rule for the first historical focal pair; this is feasibility evidence only, not a finished matched design or proof of impossibility under two swaps. If exact matching is too restrictive, predeclare a factorial or stratified design that varies shared partners and path count separately. Choose graph pairs **before** any new model outcomes are inspected. If the interaction cannot be isolated, frame the paper around a causal effect of the relational-evidence arrangement rather than a law about shared partners.

The proposal's RQ1 is broad enough for either outcome. This check improves the scientific specificity of the next experiment without forcing an unverified explanation into the course proposal.

## Graph-only factorial feasibility screen

The deterministic [feasibility script](graph_factorial_feasibility.py) made 300,000 attempted degree-preserving edge switches from one historical 10-node graph, with its withheld pair fixed. It retained 42,917 connected graph states with either one or three shared partners. It counted possible three-edge-path values within triangle strata; no model outcomes entered the search.

- With **total triangle count and both focal endpoint triangle counts** matched, the sampled states contained one common cell pair: one versus three shared partners, both with five three-edge paths and signature `(8 total, 2 actor, 2 patient triangles)`. The sample contained **no complete 2×2 factorial** at that stricter signature.
- With **total triangle count only** matched, the sample contained feasible 2×2 cells: at seven triangles, both overlap conditions occurred at three and four paths; at eight triangles, both occurred at three, four, and five paths.

This is a Monte Carlo feasibility result, **not** a proof that stricter factorials are impossible or that the sampled graphs are independent families. It suggests two defensible design paths: (a) build a single matched contrast under the strict triangle signature, then replicate with new graph classes if enough exist; or (b) use a predeclared 2×2 design matched on degree and total triangles, measure focal triangle differences, and include those as explicit competing explanations. A larger graph size may make strict matching easier, but would also change task difficulty and needs its own baseline. The design should be selected by graph properties alone, then frozen before training.

An additional [exhaustive local search](graph_two_swap_exhaustive.py) enumerated every valid graph reachable by one or two degree-preserving switches from each of the **eight original RQ1 base graphs** (5,205–6,722 distinct states per family). For none of the eight could it find a high-overlap graph that simultaneously matched that family's sham graph on three-edge paths, total triangles, and both focal endpoint triangle counts. This **does** establish infeasibility under the original two-switch neighborhood for those eight fixed bases; it does **not** establish infeasibility for new bases, more switches, larger graphs, or a relaxed control set. The paper should not demand an impossible local rewire and then quietly change its design after outcomes appear.
