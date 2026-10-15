# Fresh-graph shared-partner confirmation

Completed 8 October 2026. **Both hierarchical success rules passed** under the
frozen [protocol](PROTOCOL.json).

## Design and integrity

Eight previously unused, mutually nonisomorphic 10-entity, degree-four
training graphs were selected by the [graph-only manifest](GRAPH_MANIFEST.json).
The selection excluded 32 historical graph classes. Each graph has twenty
observed unordered actor/patient pairs and twenty-five unseen pairs. Every
entity appears with four partners; observed pairs occur in both role orders
and events. Each chosen graph has at least three unseen pairs whose two
entities share 0–1 observed partners (*low overlap*) and at least three whose
entities share 3–4 (*high overlap*). Thus pair exposure counts are controlled
while the amount of shared relational context differs.

The study trained three paired starts for each graph in three arms: an ordinary
four-choice baseline, an auxiliary task predicting unordered participant
identities, and a parameter-matched auxiliary task predicting ordered actor
and patient identities. All arms used the same balanced generator, 6,000
updates, initial base weights within paired starts, and fixed final-checkpoint
evaluation. **72/72** models completed. The final audit reproduced the frozen
manifest and protocol; all checkpoint hashes matched; each model had 20 seen
and 25 unseen pairs in both 512-rendering evaluation suites. No graph, start,
or arm was excluded. User-requested [interim reads](EARLY_PEEK.md) did not
change training or analysis.

## Primary result: which unseen pairs are hard?

The frozen primary endpoint was each graph's mean binary actor/patient
accuracy on high-overlap unseen pairs minus that on low-overlap unseen pairs
for the ordinary baseline. Pair scores were averaged within category, then
the three starts within graph, and finally the eight graphs equally. Negative
values mean high overlap is harder.

| Fresh graph | Low-overlap pairs | High-overlap pairs | Baseline high − low | Unordered high − low | Ordered high − low | Ordered − unordered penalty |
|---|---:|---:|---:|---:|---:|---:|
| overlap0 | 9 | 4 | −22.73 | −17.11 | −2.07 | +15.04 |
| overlap1 | 12 | 3 | −24.72 | −21.38 | +0.09 | +21.46 |
| overlap2 | 10 | 3 | −30.11 | −23.01 | +0.04 | +23.05 |
| overlap3 | 14 | 3 | −9.73 | −17.34 | −0.39 | +16.95 |
| overlap4 | 17 | 4 | −12.44 | −11.34 | +0.07 | +11.41 |
| overlap5 | 11 | 5 | −29.11 | −22.56 | −1.51 | +21.04 |
| overlap6 | 8 | 3 | −33.25 | −35.42 | −4.00 | +31.42 |
| overlap7 | 10 | 4 | −20.48 | −21.60 | −2.38 | +19.22 |

All differences in the table are percentage points. The baseline penalty was
negative in **all eight graphs and all 24 starts**. Its equal-weight graph
mean was **−22.82 points**. The prespecified exact, two-sided graph sign-flip
test gave **p = 0.0078125**; the graph-level t-based 95% interval was
**−29.82 to −15.83 points**. Every baseline model exceeded the required 95%
familiar-pair accuracy; the minimum among all 72 models was 99.25%. The
frozen primary rule (at least a 10-point penalty, p ≤ 0.05, familiar gate)
passed.

Mean baseline role accuracy was **94.39%** on low-overlap unseen pairs and
**71.57%** on high-overlap pairs. The primary endpoint measures a within-graph
difficulty association. It does not by itself show that changing overlap
causes an error; other local properties of a pair can co-vary with overlap.

## Secondary gated result: does ordered-role training reduce the penalty?

The prespecified second endpoint was the ordered-role model's high-minus-low
penalty minus that of the parameter-matched unordered-identity model, paired
by graph and start. The primary rule passed, so the hierarchical second test
was eligible. Its graph mean was **+19.95 points**, positive in **all eight
graphs and all 24 paired starts**. The exact two-sided graph sign-flip test
gave **p = 0.0078125**, with a graph-level t-based 95% interval of **+14.94 to
+24.96 points**. The frozen +10-point practical target passed.

Unordered-identity training averaged **95.96%** on low-overlap pairs and
**74.74%** on high-overlap pairs. Ordered-role training averaged **99.14%**
and **97.87%**, respectively. On all 25 unseen pairs, regardless of overlap,
ordered-role accuracy was **98.64%** versus **89.48%** for unordered
identities, a +9.16-point descriptive gain positive on every graph. This is
also a fresh-graph replication of the prior ordered-role transfer finding.

The mean familiar four-choice loss relative to unordered training was
**−0.001 percentage points** (effectively zero), below the 1-point limit.
The mean unseen-event loss was **0.265 points**, below the 2-point limit.
One individual paired start (`overlap7`, seed `291003`) did lose **5.76
points** of unseen-event accuracy despite the acceptable mean; its ordered
model scored 93.75% on events versus 99.51% for its unordered control.
This degradation is retained in all results and limits a blanket claim that
the intervention never harms another ability.

## Rendering reliability and limits

The second prespecified scene rendering gave a baseline high-minus-low
penalty of **−23.35 points** and an ordered-minus-unordered penalty
attenuation of **+20.20 points**, close to the primary rendering's −22.82 and
+19.95. These are reliability checks on the same trained models, not new
independent graph replications.

The supported conclusion is that, in this controlled architecture, unseen
entity pairs with more shared observed partners are markedly harder for the
ordinary model, and ordered-role supervision largely removes that
vulnerability beyond the effect of equally parameterized unordered-identity
training. The graph was selected to contain both overlap categories, so the
estimate applies to this graph population. Overlap is predictive here; the
study did not manipulate overlap alone, identify a unique internal mechanism,
or test natural language or pretrained LLMs. A degree-preserving edge-rewiring
study on fixed withheld pairs would test causality. A language-model study is
needed to assess transfer beyond coded messages and generated scenes.

Machine-readable per-start, per-graph, and primary-test results are in
[SUMMARY.json](SUMMARY.json). Reproduction code is
`../rq1_overlap_confirmation_v1.py` and
`../run_rq1_overlap_confirmation_v1.py`.
