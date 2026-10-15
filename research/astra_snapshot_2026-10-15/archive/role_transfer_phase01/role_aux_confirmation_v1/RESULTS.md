# Fresh-graph ordered-role confirmation

**Completed 7 October 2026. The frozen success rule passed.**

## Design and data integrity

The [protocol](PROTOCOL.json) and [fresh-graph manifest](GRAPH_MANIFEST.json) were fixed before confirmation training. Eight new, mutually nonisomorphic 10-entity graph layouts were selected by a graph-only rule; eight historical graph classes were excluded. Each graph had three paired training starts and three conditions: the ordinary balanced baseline, an auxiliary objective predicting the *unordered* identities of the two participants, and an otherwise matched auxiliary objective predicting their *ordered actor/patient roles*. This is 72 models, 6,000 updates each. The unordered condition is the primary control because it uses the same architecture, number of heads, examples, auxiliary weight, and training budget as the ordered condition.

The runner completed **72/72** models without failure. The audit reproduced the frozen manifest and protocol hashes. All 72 saved checkpoints matched the SHA-256 hashes stored with their metrics. A separate data-integrity pass verified that each model had 20 observed and 25 unseen entity pairs in each of two rendering suites, and that every reported group mean recomputed from its 45 pair-level rows. No graph, seed, or arm was excluded. The two [user-requested interim reads](EARLY_PEEK.md) were logged; they did not change the run plan or analysis.

## Primary result

The primary model metric was mean binary actor/patient accuracy over all 25 unseen pairs. The primary analysis averaged paired starts *within each graph* and then weighted the eight graphs equally. Pairs, renderings, and starts were not treated as independent graph replications.

| Fresh graph | Balanced baseline | Unordered identities | Ordered roles | Ordered − unordered |
|---|---:|---:|---:|---:|
| fresh0 | 96.83% | 94.43% | 97.76% | +3.32 points |
| fresh1 | 93.94% | 90.45% | 98.80% | +8.35 points |
| fresh2 | 90.81% | 87.61% | 99.21% | +11.60 points |
| fresh3 | 92.14% | 92.54% | 98.99% | +6.44 points |
| fresh4 | 94.19% | 91.25% | 99.23% | +7.98 points |
| fresh5 | 95.64% | 92.87% | 97.66% | +4.79 points |
| fresh6 | 88.40% | 82.07% | 99.58% | +17.51 points |
| fresh7 | 91.15% | 87.30% | 98.36% | +11.06 points |
| **Mean** | **92.89%** | **89.82%** | **98.70%** | **+8.88 points** |

All eight graph-level differences favor ordered roles. The prespecified exact, two-sided sign-flip test over the eight graph differences gives **p = 0.0078125** (2 of 256 sign assignments). The graph-level t-based 95% interval for the mean difference is **+5.12 to +12.64 percentage points**. The point estimate exceeds the protocol's +5-point threshold. Relative to the ordinary baseline, the ordered condition gains **+5.81 points** on the same primary metric. These are graph-distribution estimates for this controlled task, not population estimates for natural-language models.

## Safeguards and secondary outcomes

The prespecified guardrails also pass. Observed-pair four-choice accuracy was 99.971% for unordered identities and 99.957% for ordered roles: a **0.014-point loss**, well below the allowed 1 point. Unseen-event accuracy was 98.537% versus 99.023%: ordered roles were **0.486 points higher**, whereas the rule allowed up to a 2-point loss. The ordinary baseline's unseen-event accuracy was 99.498%; ordered roles were 0.476 points lower than that secondary comparator.

Unseen four-choice accuracy was 88.87% for unordered identities, 92.50% for the ordinary baseline, and 97.93% for ordered roles. On the second independent scene rendering, unseen binary role accuracy was 89.89% for unordered identities and 98.75% for ordered roles, close to the primary rendering. The number of unseen pairs below 80% binary role accuracy averaged **5.33 per model** for unordered identities, **3.42** for baseline, and **0** for ordered roles.

The graph averages are uniformly positive, but individual starts vary. Ordered roles fell below unordered identities on the primary metric in **one of 24 paired starts** (fresh0, seed 271003, by 0.10 point) and below the ordinary baseline in **two starts** (fresh0 and fresh5, both seed 271003, by 1.40 and 1.27 points). Twelve starts had some observed-pair accuracy loss versus unordered identities, all under 0.15 point. Eleven starts had some unseen-event loss versus unordered identities; the largest was 1.02 points and none exceeded the 2-point guardrail. All 24 paired-start comparisons, including every degradation, are in [PER_START_COMPARISONS.csv](PER_START_COMPARISONS.csv).

## Interpretation and limits

In this architecture and task, making the roles of the two entities explicit during training improves transfer to entity pairs never shown together. Predicting which two entities are present, without their order, does not reproduce the gain. The result survives new graph layouts, paired starts, balanced distractors, all unseen pairs, and a second rendering suite. That supports a controlled effect of **ordered role supervision**, not a claim that the model developed a particular internal mechanism.

The task uses coded messages, generated scenes, 10 entities, one small two-tower Transformer architecture, and a fixed auxiliary weight. Ordered actor/patient targets are supplied during training. We have not yet shown that the same intervention improves a pretrained language model, reduces prompt sensitivity, or transfers to authentic sentences. Those require a separate open-weight LLM study with matched fine-tuning data and new role evaluations. The local Qwen answer-level pilot is reported separately in `llm_role_pilot_v2/RESULTS.md`.

Machine-readable graph-level and arm summaries, including the frozen success decision, are in [SUMMARY.json](SUMMARY.json). Reproduction code is `../confirmation_runner_v1.py` and `../run_confirmation_all_v1.py`.
