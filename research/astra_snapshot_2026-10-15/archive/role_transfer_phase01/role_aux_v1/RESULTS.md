# Developmental ordered-role auxiliary comparison, 6 October 2026

This is development work on three previously inspected graph layouts, not an
independent confirmation or a result about natural language or pretrained
LLMs. The design was recorded in `PROTOCOL.json` after the generator gate and
before these auxiliary trainings. The six balanced-generator baselines are
the saved runs from `../generator_gate_v1/full`; the two auxiliary arms were
then trained with the same worlds, starts, semantic pool order, corrected
generator, model backbone, and 6,000-update schedule.

The ordered auxiliary task asks the same two 10-class heads to recover source
and target identity from the coded query and from the correct scene. The
matched unordered control uses the same heads and loss weight but predicts
the lower and higher identity indices, without role order. Only observed
training pairs supply these labels. Auxiliary heads are unused at test time.
The original four-choice loss remains in all arms.
The baseline has 78,096 trainable parameters; each auxiliary arm has 79,076,
so the two auxiliary arms are parameter-matched (980 extra parameters).

## All-pair outcome

Each model was tested on all 25 unseen and 20 seen unordered pairs. Each pair
has 512 crossed renderings per scene seed. Primary development numbers below
are mean unseen binary role accuracy for evaluation seed 260107.

| Graph | Start | Balanced baseline | Unordered identity | Ordered roles | Ordered − baseline | Ordered − unordered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| new0 | 261001 | 85.71% | 93.86% | 99.39% | +13.68 | +5.53 |
| new0 | 261002 | 80.30% | 88.59% | 98.86% | +18.56 | +10.27 |
| new2 | 261001 | 83.03% | 85.65% | 99.54% | +16.51 | +13.89 |
| new2 | 261002 | 92.39% | 81.72% | 99.26% | +6.87 | +17.54 |
| confirm0 | 261001 | 92.85% | 91.59% | 98.82% | +5.97 | +7.23 |
| confirm0 | 261002 | 91.23% | 87.37% | 98.95% | +7.72 | +11.58 |

Equal-weight means over six paired starts: **87.58% baseline, 88.13%
unordered control, and 99.14% ordered roles**. The ordered role contrast is
+11.55 percentage points over baseline and +11.01 points over unordered
identity. Every graph mean is positive for both contrasts. The three
ordered-minus-baseline graph means are +16.12, +11.69, and +6.84 points.
The three ordered-minus-unordered graph means are +7.90, +15.71, and +9.40.

There are 150 unseen model-pair cases per arm. Cases below 80% binary role
accuracy: 36 baseline, 36 unordered, **0 ordered**. Mean unseen four-choice
accuracy was 86.63%, 86.86%, and 98.04%, respectively. Mean unseen event
accuracy was 98.80%, 97.97%, and 98.73%. Mean seen four-choice accuracy was
99.93%, 99.96%, and 99.97%. Thus the ordered arm passed the familiar-task
gate, and its event accuracy was essentially unchanged in aggregate. The
largest individual ordered-versus-baseline event decline was about 1.21
percentage points, within the protocol's 2-point bound.

The second scene-rendering seed gave 87.48%, 88.00%, and 99.10% unseen
binary role accuracy for baseline, unordered, and ordered arms. Weak-pair
counts were 36, 38, and 0. This checks rendering stability, not independence
of graph or training start. All 18 saved model hashes (six baseline and 12
auxiliary) match their metrics files.

## Decision and limits

The frozen development advancement rule passed: the ordered-role arm beat
both comparators on all three graph means, all six ordered models exceeded
99% seen-pair four-choice accuracy, and no individual ordered model lost more
than two points of unseen event accuracy relative to its baseline. This is a
strong reason to prepare untouched-graph confirmation.

The intervention adds **explicit ordered-role supervision** derived from
observed training examples. It is therefore not evidence that the original
models already encoded roles but failed to use them, and not a label-free
repair. Shared heads encourage source/target information to be accessible
from both query and scene representations; the present output does not prove
that this is the sole causal mechanism. A generic multi-task or alignment
baseline beyond the unordered identity control may still be needed for a
publication claim. Source motion remains perceptually cued in this synthetic
scene task. None of these runs was on natural sentences or a pretrained LLM.

All graphs and starts here are development material. Do not change the
auxiliary weight, checkpoint step, metric, or success criterion after looking
at these results and then call the same worlds confirmation. Select fresh
nonisomorphic graphs by a graph-only rule and freeze the exact protocol and
analysis before the first confirmation training.
