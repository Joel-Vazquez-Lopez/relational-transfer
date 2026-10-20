# B-D28 frozen discovery protocol: gauge information budget

Status: **discovery only**. Selected after B-D27's held-identity 48-dimensional orthogonal query alignment improved cross-model function. B-D27's map used 24 identity pairs × two event queries and has substantial capacity.

## Question

How much query correspondence is needed for the cross-model alignment to predict held relational identities and downstream scores? Does an event-difference-only fit suffice?

## Fixed design

Use the exact B-D27 six held ordered pairs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,0). Deterministically permute the other 24 pairs with NumPy RNG seed 202628 and take nested fit prefixes of **4, 8, 12, 18, 24** pairs. For each matched seed 3801/3802 and each direction NO_CUE↔CUE, fit one 48×48 orthogonal Procrustes map using either (a) both full event queries per fit pair, or (b) only the event-query difference per fit pair. Do not use candidate vectors, outcomes, held queries, or geometry to fit or select a map. No capacity/regularization adjustment after outcomes.

Report held full-query cosine, held event-difference cosine, held-pair score RMSE relative to recipient, concordant/conflict accuracy and relation/cue margins after the aligned swap. Also report effective rank of the fitting query matrix and held-query residual relative to its row span. The recipient's own query is the functional ceiling and the raw donor is the unaligned baseline.

If only large fitting sets work, the B-D27 gauge is a high-capacity calibration rather than a compact reusable coordinate rule. If small sets transport across held identities and both directions, the low effective complexity is an empirical discovery candidate, still not an operator or independent confirmation. P3.2-E/canonical artifacts stay unchanged; protected seeds 1704–1706 and other confirmation material remain sealed.
