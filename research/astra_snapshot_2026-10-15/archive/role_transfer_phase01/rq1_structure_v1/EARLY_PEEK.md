# User-requested interim read

At 2026-10-08 07:24:58 UTC, the user asked whether the run was heading in the
predicted direction. The first **fully completed network pair** in frozen plan
order was inspected. It has all three starts in both conditions (six models).
The other network pairs were not inspected. The primary protocol, graph
manifest, training plan, analysis code, and stopping rule were left unchanged.

For pair0, the sparse graph has four triangles and the clustered graph has
eight. They share 15 withheld entity pairs, which are the primary evaluation
items for this comparison. Mean binary actor/patient accuracy across the three
starts was **90.78% sparse** and **89.57% clustered**, a **+1.21 percentage-point**
sparse-minus-clustered difference. The start-level differences were **+8.89,
-0.43, and -4.84 points** in seed order 281001–281003.

The first graph pair therefore has a small average difference in the predicted
direction, with inconsistent start-level signs. It is below the frozen
three-point practical threshold and cannot establish whether the eight-pair
effect exists. No significance test or final decision is computed from this
interim read. All 48 models remain in the plan.

## Second user-requested status read

At 2026-10-08 08:12:40 UTC, 28/48 models were complete. The first four full
network pairs (24 models) were inspected in their fixed plan order. Their
sparse-minus-clustered differences in shared-withheld-pair binary role accuracy
were **+1.21, -11.03, -5.91, and -1.60 percentage points** for pair0–pair3.
The provisional four-pair mean was **-4.33 points**, opposite to the predicted
direction; three of four pairs favored the more clustered network. The
remaining four pairs are needed for the frozen analysis. No graph, start,
training process, endpoint, statistical test, or threshold was changed after
this look. The current active pair was not included.

## Third user-requested status read

At 2026-10-08 08:31:54 UTC, 34/48 models were complete. The first five full
network pairs (30 models) were inspected in fixed plan order. Pair4 favored the
sparse network by **+6.70 points**, so the provisional five-pair mean changed
to **-2.13 points** (sparse minus clustered). Three of five pairs favored the
clustered network and two favored the sparse network. This makes the interim
direction less consistent than the four-pair read. Three network pairs remain
for the frozen final analysis. No plan or analysis choice was changed.
