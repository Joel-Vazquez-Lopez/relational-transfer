# Exploratory lead: shared-partner overlap and role-transfer failure

Date: 8 October 2026. This is an **adaptive analysis of saved results**, not
an independent prospective confirmation. The feature screen compared several
graph descriptors after seeing the failed global-triangle experiment. No new
models were trained for this analysis.

## Main observation

For an unseen entity pair `(a,b)`, define *shared-partner overlap* as the
number of other entities observed with both `a` and `b` during training. Every
entity has four observed partners, so this count ranges from zero to four.
The pair `(a,b)` itself is never trained in either direction.

Across the **16 new baseline networks** from the triangle experiment, the
within-network Spearman association between overlap and mean unseen-pair
binary role accuracy was negative in **16/16** networks (median −0.66). Among
pairs at distance two, it was negative in all **15** networks with any overlap
variation; the remaining network had a constant overlap count in that subset.
Thus the overall relationship is not caused only by distance-three pairs with
zero shared partners.

| Shared partners | Unseen pairs across 16 networks | Mean role accuracy | Pair means below 80% |
|---:|---:|---:|---:|
| 0 | 32 | 97.56% | 0.0% |
| 1 | 137 | 93.93% | 4.4% |
| 2 | 168 | 87.36% | 23.8% |
| 3 | 59 | 67.61% | 72.9% |
| 4 | 4 | 58.32% | 100.0% |

Each table row is an entity pair's mean across three training starts. The
400 rows are nested within 16 graphs and are **not** 400 independent graph
replications. A second independent rendering of the same models gave nearly
identical role accuracies by overlap count: 97.54%, 94.01%, 87.50%, 68.17%,
and 58.30%. Unseen event accuracy was around 97–99% at every count, so the
large gradient is principally about role direction rather than event type.

The feature search is saved in [SUMMARY.json](SUMMARY.json). A simple
common-neighbor ridge predictor lowered leave-one-network-pair-out mean
absolute error from 11.42 to 9.31 percentage points on these 16 graphs, with
an out-of-pair R² of 0.36 relative to each fold's training mean. Identity-pair
features alone performed worse than that mean baseline. These are exploratory
model comparisons; selecting the best feature after seeing them invalidates
using this same cross-validation as a confirmation test.

## Separate graph-set check

The previously trained **eight fresh graphs** from ordered-role confirmation
used the same corrected generator but were not part of the 16-network screen.
Their ordinary baseline showed a negative within-network overlap–role
association in **8/8** graphs; it remained negative when restricted to
distance-two pairs in 8/8. Their role accuracy averaged 96.67% at one shared
partner, 92.83% at two, and 82.71% at three.

This check was retrospective. The graphs and their overall outcomes had been
used in other analyses, although the common-partner pattern had not been used
to select or train them. It strengthens the lead but is not a new prospective
confirmatory study. A numeric predictor trained on the 16-network set did
not calibrate well across sets: its common-neighbor-only external R² against
the development mean was −0.011. The two sets had different overall mean
accuracies (87.22% and 92.89%). The *within-graph ranking* is more stable than
the predicted absolute percentage.

## Connection to ordered-role training

In the same earlier eight graphs, the mean within-graph linear slope per extra
shared partner was **−4.95 percentage points** for the ordinary baseline,
**−7.85** for the unordered-identity auxiliary control, and **−0.24** for
ordered-role training. The ordered-minus-unordered slope difference was
positive in all eight graphs. At one versus three shared partners, ordered
models scored 98.63% versus 98.59%, whereas ordinary models scored 96.67%
versus 82.71%.

Using high overlap (3–4) minus low overlap (0–1) as a graph-level contrast,
the eight-graph means were **−12.64 points** for the ordinary baseline,
**−19.52** for unordered-identity training, and **−0.58** for ordered-role
training. Ordered-role training reduced this descriptive penalty relative to
the parameter-matched unordered control in every one of the eight graphs.

This links the new difficulty lead to the already confirmed ordered-role
behavioral result. It is a **post hoc interaction analysis**, however. Ordered
models are also near ceiling, so slope flattening could partly reflect the
general accuracy gain. It does not by itself establish a mechanism or show
that changing partner overlap causes the failure.

## Interpretation and next experiment

The defensible current statement is: *in this controlled architecture,
unseen pairs with more shared training partners tend to have worse role
accuracy, and explicit role training largely removes that association.*
One possible explanation is that shared partners make the two entities'
learning contexts less distinct. Other local graph properties co-vary with
overlap, and training distractor co-occurrences may also matter. The old
selected focal-pair common-neighbor experiment failed to establish a general
rule; the all-pair, corrected-generator analysis here does not erase that
failure.

**Recommended next run:** freeze overlap count as the single primary risk
factor before new training. Select at least eight unused, nonisomorphic
10-node degree-four graphs by graph-only rules that each supply at least three
unseen pairs with low (0–1) and at least three with high (3–4) overlap. A
graph-only feasibility search found eight unused classes satisfying this rule.
Train three paired starts
of the ordinary baseline, unordered-identity control, and ordered-role model
on every graph. Evaluate every unseen pair with the fixed two-rendering audit.
Primary graph-level endpoint: high-overlap minus low-overlap role accuracy in
the ordinary baseline, averaged across starts and then graphs. Secondary,
predefined endpoint: whether ordered-role training reduces the high-overlap
penalty relative to its parameter-matched unordered control while retaining
familiar accuracy. Preserve all graphs and starts regardless of result.

If the predictor confirms, use a separate degree-preserving edge-rewiring
study on fixed withheld pairs to test whether altering overlap itself changes
transfer. Then test the same exposure pattern in natural-language sentences
and open-weight pretrained language models. The latter is required for an LLM
claim; neither the current coded messages nor generated scenes establish one.

## Literature boundary

[Bogin, Gupta & Berant (2022)](https://aclanthology.org/2022.emnlp-main.175/)
already show that local training structures predict semantic-parsing difficulty
and can guide example selection. [Oren et al. (2021)](https://aclanthology.org/2021.emnlp-main.843/)
and [Gupta et al. (2022)](https://aclanthology.org/2022.findings-emnlp.365/)
study structurally diverse training-example selection. Thus neither local
structure prediction nor diverse data selection is a new general concept.
The possible contribution here is the **specific inverse shared-partner
pattern in role transfer, its internal explanation, an independently tested
role-sensitive repair, and a valid language/LLM extension**. Publication
novelty remains to be established with a fuller literature review and fresh
experiments.

Reproduction data: [ROW_DATA.json](ROW_DATA.json), [SUMMARY.json](SUMMARY.json),
[EXTERNAL_CHECK.json](EXTERNAL_CHECK.json). Analysis scripts are
`../rq1_feature_screen_v1.py` and `../rq1_external_check_v1.py`.
