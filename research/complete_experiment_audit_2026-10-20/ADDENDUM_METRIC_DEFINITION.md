# Addendum: target-after accuracy is not paired redirection

20 October 2026. A subsequent archive-only check clarifies a term reproduced from the historical discovery reports in this audit.

In D15/D21 and the inspected developmental readers, `redirected` counts target-after answers. It does not require a correct original answer. Consequently, statements such as A1701 source “redirection .625” must be read as **80/128 target-after correct**, not80 actual baseline-to-target switches. Only **2/128** had both original and counterfactual answers correct. D15 A1702 gives3/128, A1703 zero; E starts retain120–125/128 source paired successes and128/128 target paired successes.

The record contains20 developmental summary rows with zero reported query change but positive target-after accuracy. These cannot support an interpretation as causal switching. Full stagewise paired rates cannot be reconstructed from aggregate summaries alone.

See the [new design and measurement analysis](../role_general_acquisition_design_2026-10-20/DESIGN.md), [raw recheck](../role_general_acquisition_design_2026-10-20/ARCHIVE_RECHECK.json), [per-model table](../role_general_acquisition_design_2026-10-20/PAIRED_ENDPOINTS.csv), and [reproduction script](../role_general_acquisition_design_2026-10-20/archive_check.py).

This narrows the interpretation of D15/D17/D21/D23 and subsequent historical uses of this field; it does not rewrite their original numerical outputs or decisions. It does not automatically invalidate D5's training-support results or later Qwen studies with their own paired/native controls. The A/E discovery difference survives a stricter endpoint definition, but remains adaptive evidence on the same models.
