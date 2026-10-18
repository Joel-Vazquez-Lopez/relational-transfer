# A concrete measurement finding: readout-dependent construction selection

19 October 2026. **Post hoc descriptive synthesis**, using every Qwen native evaluation cell from the two completed frozen investigations. No new forwards, probe fits, endpoint redefinition, significance tests or discarded cases. This is not a retrospectively preregistered hypothesis.

## Observed result

Across six evaluation lexical families (g03..g06,h01,h02), 48 native prompts are unique. All use the same fixed Yes/No prompt and a single scalar threshold estimated from eight earlier exposed calibration examples; it was not refit on any of these 48 outcomes.

|Syntax|Native next-token correct|Fixed calibrated correct|All-four-correct native family/voice units|All-four-correct calibrated units|
|---|---:|---:|---:|---:|
|Active|14/24|24/24|0/6|6/6|
|Passive|24/24|24/24|6/6|6/6|

Thus an all-four-native-correct competence gate would admit **all six passive units and zero active units**. The unchanged external threshold would admit both. The models' fixed scores still distinguish the controlled relationship in both constructions. This demonstrates that the screen's admitted linguistic sample depends on the response decision rule in this specific panel.

## Available Llama comparison

All four original Llama evaluation families were included without selecting favorable cases; Llama was not run on the two new families.

|Checkpoint/panel|Syntax|Native correct|Fixed calibrated correct|Native complete units|Calibrated complete units|
|---|---|---:|---:|---:|---:|
|Qwen, matched original 4 families|Active|9/16|16/16|0/4|4/4|
|Qwen, matched original 4 families|Passive|16/16|16/16|4/4|4/4|
|Llama, same 4 families|Active|13/16|16/16|2/4|4/4|
|Llama, same 4 families|Passive|12/16|15/16|2/4|3/4|

The sharp active/passive split is **not a cross-model law**: Llama does not show Qwen's native construction split, and its own fixed threshold does not fix every passive answer. An all-correct screen would also admit different construction mixtures in these two checkpoints on matched inputs. That is the relevant potential confound for comparing their computational development; it is not proof that any particular measured circuit difference is caused by the filter.

## What this means

A mechanistic study that keeps only fully correct examples could end up studying different constructions depending on its answer convention. Later differences in maps/circuits would then be conditional on that selected sample. That is a concrete empirical issue to measure, and a better research target than declaring every structured hidden-state difference a new semantic mechanism.

This is **not** evidence that active and passive computations are identical, that all native errors are constant biases, that external calibration is native understanding, or that all competence filtering is invalid. A sensitivity of the reported mean causal profile to filtering is measured below; no switch in a preregistered mechanistic conclusion is demonstrated. The calibration rule changes the classifier, not the model. Full generated responses were not collected. Agent/Patient labels and synthetic sentences have AI-only annotation.

The six lexical families share narrow templates, two voices and a small verb/name inventory. They are not48 independent linguistic phenomena. The direction was recognized after outcomes; no p values were computed, and no statistically significant discovery is claimed. The test/family classifications of the original investigations remain unchanged. One quantized checkpoint from each family and this prompt cannot establish a family-level or general law.

## Does selection change an actual reported causal profile?

A further **post hoc arithmetic audit** used all fixed source-access profiles from the original matched four-family panel, without new forwards or changing scores. Every mode and cut, own-selected and common-selected subsets are retained in FILTER_PROFILE_RESULTS.json. The illustration below uses the period-source-isolated condition at cut 14, the site already fixed for this investigation.

|Eligibility rule|Qwen mean retention|Llama mean retention|Llama minus Qwen|Selected units Qwen/Llama|
|---|---:|---:|---:|---:|
|All original units|.7380|1.0211|.2831|8/8|
|All-four-native-correct|.5069|1.0389|.5320|4/4|
|All-four-fixed-calibrated-correct|.7380|1.0222|.2842|8/7|

The native gate makes this reported cross-checkpoint gap about 1.88 times its full-panel value, **without changing weights, examples' scores, interventions or the retention metric**. Only eligibility changes. Qwen loses every active unit, while Llama loses a different mixture of constructions and lexical families. This demonstrates a sensitivity of the conditional reported comparison to the analysis policy; it does not prove a change in the model's computation or isolate syntax as the only source of the shift. The means estimate different selected samples, so neither is automatically an incorrect conditional estimand. The error would be generalizing one of them to an unmatched/full population without accounting for selection.

These data show a change in a descriptive causal-profile comparison. They do **not** demonstrate a reversal of any preregistered mechanistic conclusion, validate a universal filtering law, or repair the old failed gates. Ratio averaging and different native sensitivity magnitudes also require care; raw D means are retained alongside every ratio. No favorable layer was selected as a new hypothesis and no significance tests were added.

## Potential paper investigation, not a paper claim

A credible follow-up could prospectively ask: **How much do answer conventions and competence screens change which linguistic constructions—and which apparent processing patterns—an interpretability study admits?** Independently vary construction, semantic relation and answer encoding on a fresh corpus; freeze raw-generation and calibrated metrics; retain the full panel; compare the selected subsets' causal profiles against the full sample in multiple checkpoints. Same-answer/different-relation controls are necessary. A construction-selective screen plus a prospective change in inferred causal conclusions would be more informative than another cross-model cosine.

Calibration itself is established, including Zhao et al.(2021), https://proceedings.mlr.press/v139/zhao21c.html. Readout, task metrics, probe capacity and causal assay choices also have extensive precedent. The empirical link to construction-specific sample admission is a candidate contribution requiring a dedicated literature audit and independent prospective replication. We have not established novelty, generality or an immediate paper.

`analyze.py` reproduces the complete source-identified synthesis, per-family results and all 80 original rows (48 Qwen, 32 Llama) from saved RAW files. RESULTS.json preserves source SHA256 hashes. There is no new inference hidden in this analysis.
