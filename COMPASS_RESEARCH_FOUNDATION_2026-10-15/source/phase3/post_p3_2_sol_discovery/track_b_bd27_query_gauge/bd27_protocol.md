# B-D27 frozen discovery protocol: constrained query gauge alignment

Status: **discovery only**. Selected after B-D26 raw cross-swaps failed while matched CUE/NO_CUE event-query directions had low cosine.

## Hypothesis and strong null

The two trained models may use a shared functional event readout expressed in different global coordinates. If so, one orthogonal map fitted only to oracle-message query vectors should align their event-query directions on relational identities withheld from fitting, and should make a query from one model function with the other model's untouched candidate vectors. Failure would weaken the *global orthogonal gauge* explanation; it would not prove irreducible co-adaptation or rule out richer nonlinear/contextual gauge maps.

## Fixed fit and test

For each matched seed 3801/3802, use only the final B-D21 NO_CUE/CUE models. Enumerate all 30 ordered source-target identity pairs and both events. Hold out six ordered pairs **(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)**; use the other 24 pairs × 2 events (48 query vectors) to fit a single 48×48 orthogonal Procrustes map from donor to recipient query coordinates. No candidate states, scores, scene geometry, or held query states enter the map fit. Fit both directions separately. No capacity selection, regularization tuning, or post-outcome rescue.

On the twelve held query vectors, report full-query and event-difference cosine against recipient query vectors. On the B-D21 frozen 2×2 scenes belonging to the six held identity pairs, score untouched recipient candidate vectors using (a) recipient queries, (b) raw donor queries, (c) orthogonally aligned donor queries, and (d) a fixed random orthogonal control. Report concordant accuracy, conflict relation accuracy, relation/cue score contrasts and score error versus recipient. The recipient's own score is the functional reference; no map is fit to it.

A useful global gauge requires held query alignment **and** downstream functional recovery. Merely raising query cosine on fit identities does not count. This is a high-capacity coordinate test with 48 dimensions, not a compact computational operation. P3.2-E and canonical artifacts remain unchanged; protected seeds 1704–1706 and other confirmation material stay sealed.
