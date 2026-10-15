# Decision after Phases 0–1

The main observation worth carrying forward is the stable **all-pair evaluation gap**: 337 of 1,050 unobserved model–pair cases were below 80% on binary role choice on one rendered suite, while seen-pair performance was almost perfect. The later suite retained 326 of those weak cases. Every saved model had at least one weak unseen pair. These are clustered, developmental observations from eight existing graphs.

The present internal story is unresolved. Four saved trajectories from one world did not show broad late forgetting. A low-rank frozen readout fit from familiar examples did not rescue the five historical focal failures. Linear and small-MLP role decoders underfit familiar role labels, including on models with good native transfer. Their negative unseen scores cannot tell us whether information is absent.

## Next bounded test for Sol

**Question:** Does the all-pair gap persist after removing two known training-generator nuisances: aliased padding and distractor identity correlated with example serial/order?

1. Write a versioned generator. Give the fourth message token its own embedding. Draw the distractor independently and uniformly from identities other than source and target for every candidate, balancing it over an epoch where feasible. Keep the same motion physics, graph, model size, four semantic choices, optimizer, update count, and number of training examples. Add a unit-level audit for every changed marginal. Never write into historical result directories.
2. Use three already studied development graphs chosen by the graph-only rule `new0`, `new2`, `confirm0`, two new development seeds per graph, and two arms: corrected padding with historical distractor rule, and corrected padding with independently balanced distractors. Match initialization, graph label map and the intended pool order per seed; document any unavoidable episode-level difference. This is 12 full trainings. A small preflight can run first to verify both arms learn the familiar task.
3. Evaluate all 25 unseen pairs and all 20 seen pairs with the fixed 512-question crossing in `PROTOCOL.json`, plus another scene seed for reliability. The primary developmental comparison is mean unseen binary role accuracy averaged within graph, then across graphs. Report the fraction of weak pairs, four-choice and event accuracy, and every graph/seed result. Check that familiar four-choice remains at least 99% before interpreting transfer.
4. If balancing distractors materially shrinks the gap across all three graphs and six paired starts, investigate the shortcut mechanism and check a fresh graph only after freezing a precise claim. If the gap persists, build one role-sensitive learning intervention and a matched non-role control under the corrected generator. If padding alone changes the result, report that separately; it is part of the common baseline here, not the comparison of interest.

This gate is developmental. It tests a concrete alternative explanation before a larger training sweep. The numbers above are a workload cap, not a claim of statistical power. Do not use protected Track A/B seeds or reserved worlds. Do not reinterpret the historical 12-model focal follow-up as an all-pair confirmation: its other pairs were inspected only now.

For a publication claim later, freeze one explanation, repair, and all-pair metric on untouched graph layouts and seeds. A controlled language task and pretrained-model evaluation are separate scope tests, not substitutes for this step.
