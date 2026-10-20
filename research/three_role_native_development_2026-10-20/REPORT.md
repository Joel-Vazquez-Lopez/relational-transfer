# Three-role Qwen native development screen: completed

20 October 2026. Frozen before inference at commit `44dca65389824eb9b66e4683046dd4feb174b780`. **108/108 development prompts completed,394 actual forwards across13 bounded workers,466 production checks passed and2,759 independent replay checks passed.** No held-out prompt, Llama inference, intervention, prompt search, calibration, training, model download or paid call occurred in this screen.

## Native competence

| Source form | Correct individual answers /36 | Complete three-question source cases /12 |
|---|---:|---:|
| Active |29|5|
| Passive |26|3|
| Goal-fronted |31|7|
| Total |**86/108**|**15/36**|

t01 scored44/54 and t02 scored42/54. All108 responses terminated within the limit and were one of the three source names; no off-format or truncated output was removed. The frozen leading-space whole-name probability ranking agreed with native correctness on all108 cases. That particular secondary scoring convention does not repair the wrong-role answers; it does not rule out broader answer preferences or question-interpretation effects.

The proposed task therefore provides useful correctly answered contrasts but does not establish reliable complete-task competence across the panel. Individual answers, three-question units and donor–recipient pairs are different units. Repeated questions/forms/names are not independent linguistic samples.

## Frozen admission decision

**28/162** six-answer transposition pairs meet the frozen native criterion, including18/108 cross-form pairs. The two development families each cover all three invariant roles and all three forms among eligible cross-form pairs. Thus the minimal decision is **BASELINE_READY_FOR_CAUSAL_PROTOCOL**. This means that the specified conditional contrasts can be constructed; it is not a positive causal result, a generic competence pass, or a claim about an entire model family.

All162 pair outcomes and the six constituent answer IDs are retained. Native failures remain in the full-panel summaries and raw records. A later intervention must disclose its conditional target population and retain the complete admission record.

## A descriptive behavioral lead

All22 wrong answers identify the **sent person**. In these templates that person appears second in active clauses, first in passive clauses and third in goal-fronted clauses. One fixed mention position therefore cannot describe all of these errors. However, a syntax-aware rule, question interpretation, lexical `sent` cues, graded answer preference and other mechanisms remain viable.

| Question role | Correct /36 | Wrong answers |
|---|---:|---|
| Sender |28|8 sent-person substitutions, all passive|
| Sent person |36|0|
| Destination person |22|14 sent-person substitutions across all three forms|

This characterization was made **after** the development outcomes. It was not a frozen prediction and is not an independently replicated behavioral law. All templates share one verb and only two AI-authored name triples. These data cannot attribute the change from older two-role performance specifically to adding a third role: wording, verb and stimuli differ. No p value or novelty claim is made. The descriptive confusion ledger retains every cell, including exceptions to any tempting stronger construction rule.

## Numerical and resource evidence

The native/wrapper full-vector and repeated-trace controls pass. All36 source prefixes are invariant across the later question choices under fixed physical length128, verified in production hashes and independently in the saved source-anchor states. Every scientific continuation preserves its own original-prompt-end logits within the frozen tolerance. All13 worker restarts pass the exposed native reference. No technical tolerance was relaxed.

The independent auditor's initial arithmetic comparison stopped at a1e-10-nat tolerance because two of324 candidate log probabilities differ slightly between sequential logaddexp reduction and a max-centered sum. Maximum discrepancy1.0870849359578827e-10; all classification signs agree and the smallest absolute margin is0.09375. Only the postcollection replay arithmetic tolerance changed to1e-8; original scores, scientific gates and collection code remain frozen. `AUDIT_NOTES.md` records the stop and correction.

Every actual pass was journaled before dispatch and saved before scoring. Baselines retain all29 stages at three person anchors, source period and assistant end; full source-prefix values were hashed rather than retained, while the selected vectors are available for independent comparison. This screen captures residual states, not attention/MLP writes or causal identity-patch controls.

## Preservation and next step

`ANSWERS.jsonl`/`ANSWER_LEDGER.csv` preserve all complete responses and token paths, `FORWARDS.jsonl` and `BUDGET.jsonl` every actual pass, `PAIR_LEDGER.json` every pair, and `UNEXPECTED_ANSWERS.json` every native or probability-ranking error. Frozen sources/inputs/runtime/model hashes, preflight fixes and all controls remain available.

All394 original NPZ files are losslessly preserved in20 tar shards, with per-member SHA256 checks and bundle checksums in `PAYLOAD_BUNDLES.json`. Expanded original NPZ files also remain local. No model weights are included.

The minimal gate permits preparing the separately frozen source-state intervention test. It does not authorize opening evaluation outcomes before an intervention and prospective predictions qualify. The current plan narrows the previously proposed site grid to one declared joint participant-state candidate at block14 for a small development test; this choice is made before any new causal outcome and is not a successful result. Identity, same-role syntax, neighboring-token and lexical-donor diagnostics must accompany it. If it fails, preserve that result and stop rather than selecting a new depth from the outcomes.
