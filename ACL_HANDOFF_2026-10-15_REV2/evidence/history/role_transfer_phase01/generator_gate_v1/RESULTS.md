# Developmental generator gate, 6 October 2026

This is a development comparison on three previously studied graph layouts.
It is not an independent publication confirmation. The protocol is in
`PROTOCOL.json`; the implementation is `../generator_gate_v1.py`. Historical
checkpoints and result directories were not modified.

## Question and audit

Does the all-pair unseen-role gap survive giving the fourth message token its
own symbol, and does independent balancing of distractor identities remove
that gap? Both arms use the corrected padding token. The only training-data
contrast is the distractor schedule. Three graph layouts (`new0`, `new2`,
`confirm0`) each have 20 observed and 25 unseen unordered pairs. Each graph
has two new paired starts, with matched initial weights, semantic pool order,
graph labeling, optimizer, and 6,000 updates. The corrected schedule uses all
eight allowed distractor identities equally within every semantic candidate
group in every epoch. The generator audit passed on all three graphs.

Each trained model was evaluated on every pair using 512 crossed renderings
per pair and two scene seeds. All 12 model-checkpoint hashes match their saved
metrics. Every model passed the prespecified 99% seen-pair four-choice gate.

## Primary developmental results

The table shows mean unseen-pair binary role accuracy with evaluation scene
seed 260107. Delta is balanced minus historical distractor schedule.

| Graph | Start | Historical | Balanced | Delta (points) | Weak unseen pairs (<80%), historical → balanced |
| --- | ---: | ---: | ---: | ---: | ---: |
| new0 | 261001 | 73.47% | 85.71% | +12.24 | 13 → 7 |
| new0 | 261002 | 90.13% | 80.30% | −9.84 | 4 → 10 |
| new2 | 261001 | 88.02% | 83.03% | −4.98 | 7 → 8 |
| new2 | 261002 | 90.57% | 92.39% | +1.82 | 6 → 3 |
| confirm0 | 261001 | 83.49% | 92.85% | +9.36 | 8 → 4 |
| confirm0 | 261002 | 93.03% | 91.23% | −1.80 | 3 → 4 |

Equal-weight mean across the six paired starts: 86.45% historical versus
87.58% balanced, a +1.13-point contrast. The three graph-mean contrasts are
+1.20, −1.58, and +3.78 points. Across all six models in each arm, seen-pair
four-choice accuracy averaged 99.91% versus 99.93%; unseen event accuracy
averaged 99.09% versus 98.80%. There were 41 versus 36 weak unseen model-pair
cases out of 150 per arm. These 150 cases are clustered within only six
trainings and are not independent replications.

The second rendering suite gave 86.64% versus 87.48% unseen binary role
accuracy and again 41 versus 36 weak cases. Thus the mixed sign is not a
scene-rendering accident. Individual unseen pairs can improve or worsen even
when the mean improves.

## Decision

The unseen role gap persists after the padding fix: familiar role accuracy is
essentially perfect while unseen accuracy averages below 88% in both arms.
Balancing distractors is **not a reliable repair** here. Its effect changes
sign across starts on every graph, and the mean gain is small. Do not select
the best start, treat the +12-point case as a general result, or launch fresh
confirmation for distractor balancing alone.

The next development test should compare one explicitly role-sensitive
learning objective with a matched control that supplies entity information
without ordered roles, under the corrected generator. Both must train only on
observed pairs and keep the all-pair evaluation fixed. A fresh graph block
should be reserved until the intervention and criterion are frozen.

Limits: The scene source remains visually cued by motion; this gate used one
labeling per graph; and the controlled messages are coded inputs, not natural
sentences. No result here establishes a mechanism in pretrained LLMs.
