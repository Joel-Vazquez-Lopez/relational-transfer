# Evaluation audit, written before new evaluation outcomes

Reevaluate all 42 existing checkpoints. This is a diagnostic audit of known
results, not an independent confirmation of a new hypothesis.

Use 1024 test items per evaluation mode, crossing two directions, two events,
four answer positions, eight distractor identities, and eight repetitions.
Shuffle the crossing before scene generation. Mode `positions` uses the
existing scene generator with the newly independent answer positions. Mode
`controlled` also fixes the distractor identity to the crossed value, shared
by all four candidates. Every candidate remains a separately rendered scene.
Use identical audit scenes across every checkpoint and condition.

Report four-choice accuracy (chance 25%), role-only two-choice accuracy
(chance 50%, ties half credit), event accuracy, and the proportions of
role-only, event-only, and combined errors. Report score margin [-20,20]
and softmax probability margin [-1,1] separately, including results by event,
role direction, and answer position.

For localization, reverse the query roles on the same candidate scenes.
Measure query distance, correct/reverse scene-vector distance, their
directional alignment, and the average margin over the original and reversed
queries. Algebraically this symmetric margin equals
5*(q_forward-q_reverse) dot (c_forward-c_reverse). This is observational
localization, not a causal intervention.

Also verify the 1280 training entries contain 80 distinct ordered
entity-pair/event meanings. Audit the fourth message token, currently entity
0, as an implementation detail requiring a retraining control if the main
failure survives. Do not interpret an inference-time padding replacement as
an adequate retraining control.
