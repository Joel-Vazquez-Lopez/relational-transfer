# Matched edge-rewiring study: completed results

The frozen protocol in `PROTOCOL.json` completed all 72 planned models: eight
graph families, three paired random starts per family, and three conditions
(original network, sham rewiring, high-overlap rewiring). Graphs and endpoints
were selected before model training. `SUMMARY.json` contains every family and
start result; all saved checkpoints passed the summary script's hash checks.

## Primary result

For the same unseen pair, high-overlap rewiring reduced binary actor/patient
accuracy relative to matched sham rewiring by **26.33 percentage points** on
average across eight equally weighted graph families. All eight family effects
were negative: −29.88, −43.23, −9.57, −24.48, −32.68, −27.60, −20.70, and
−22.53 percentage points. The frozen exact two-sided family-level sign-flip
test gives **p = 0.0078125**. The graph-level t(7) 95% interval for the mean
difference is **[−34.52, −18.15] percentage points**. The prespecified success
rule passes.

Mean focal-pair role accuracy was 94.89% in the original networks, 99.12% in
the sham rewires, and 72.79% in the high-overlap rewires. The high-minus-base
difference was −22.10 points; the sham-minus-base difference was +4.23 points.
The result also repeats under the second fixed scene rendering: high minus sham
was −25.45 points. Every model cleared the familiar-pair guardrail: the lowest
familiar four-choice accuracy among all 72 models was 99.25% (threshold 95%).

## Interpretation and limit

This is evidence that a matched, degree-preserving **local network rewiring**
changes transfer to the same unseen pair while familiar performance remains
high. The manipulation moved the pair from one to three shared observed
partners. Sham and high conditions each used two effective swaps from the
same base graph and matched total triangle count, focal endpoint triangle
counts, and the number of changed neighbors at each endpoint.

The manipulation also changes other local paths and training co-occurrences.
The result therefore does **not** isolate shared-partner overlap as the sole
cause. It concerns this controlled 10-entity Transformer task; it is not yet
evidence about natural-language LLM behavior. A follow-up should identify
which accompanying local change predicts or mediates the effect, then test
whether an analogous role-transfer weakness occurs in pretrained models.
