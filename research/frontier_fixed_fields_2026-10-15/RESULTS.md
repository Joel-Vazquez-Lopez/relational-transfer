# Fixed fields, visible context and answering — bounded development result

15 October 2026. One untouched Qwen2.5-1.5B-Instruct checkpoint; 12 previously examined events from 12 distinct documents, selected by event-ID hash without outcome filtering. 540 generations including 12 exact extraction replays; no training, reserved evaluation, external credits or downloads. This is exploratory, not independent confirmation.

## Main finding

**We did not establish a residual extraction advantage over strong direct QA.** The model uses supplied fields, but visible conversational packaging can disrupt the answer contract. Even correct reference fields do not guarantee correct question-conditioned selection.

| Condition | Original QA-span exact | Provisional referent accuracy | Format compliance |
|---|---:|---:|---:|
| Format-matched direct QA | 11/24 | 14/24 | 22/24 |
| Two-pass direct self-review | 6/24 | 10/24 | 24/24 |
| Generated fields, fresh, no sentence | 23/48 | 27/48 | 45/48 |
| Generated fields, fresh, sentence included | 26/48 | 32/48 | 42/48 |
| Same fields as transcript, no sentence | 0/48 | 0/48 | 0/48 |
| Same fields as transcript, sentence included | 7/48 | 9/48 | 18/48 |
| Correct source-reference fields, fresh + sentence | 34/48 | 34/48 | 42/48 |
| Swapped source-reference fields, fresh + sentence | 26/48 | 26/48 | 42/48 |

Field conditions cross both field orders and both target questions; their 48 outputs are repeated measurements of 12 events, not twice the independent sample of the direct baseline. Reference fields are an oracle diagnostic, not an autonomous system. Full intact/swapped cells, role/order breakdowns and metrics are in ANALYSIS.json.

Fresh sentence-plus-generated-fields minus direct QA: **+8.33pp**, paired event-bootstrap interval **[-6.25,+22.92]**; event-averaged scores improve in 3 events, worsen in 2, tie in 7. Provisional referent contrast is also +8.33pp, interval [-10.42,+27.08]. Twelve-event uncertainty is large. These are descriptive intervals without multiplicity control.

The self-review baseline deteriorated, despite perfect operational compliance. Beating it is not evidence of a superior relational algorithm. It refers to the original question in earlier context rather than restating the sentence/question in its last turn; this may be a weak conversational baseline. No follow-up inference was launched to improve it after seeing outcomes.

## Stronger existing direct controls: sensitivity, not a selected new headline

All four already-run interface variants were scored on these same 12 events:

- Original direct QA: 7/24.
- Original instruction + Answer prefix: 11/24.
- Explicit span instruction without prefix: 14/24.
- Explicit span instruction + Answer prefix: 12/24.

Generated-field reader accuracy averaged over orders is equivalent to 13/24. Against the earlier explicit/no-prefix variant, the contrast is **-4.17pp [-20.83,+12.50]**. This comparison is post hoc and the interfaces differ; it is evidence of sensitivity to baseline specification, not confirmation that a chosen baseline is universally superior. All four are retained in SENSITIVITY.json. The cached fields with an oracle selector score 15/24 on this subset; that selector advantage is not an autonomous readout result.

## Actual field use, distinguished from correctness

With correct reference fields and the sentence, the model gives 34/48 correct answers. **All eight compliant but incorrect responses select the opposite field**, across six events. The other six outputs fail the operational phrase-compliance rule. Correct extraction alone therefore cannot explain every error in this condition. This does not establish where an internal role representation fails.

Swapping reference fields reduces accuracy by **16.67pp [8.33,+25.00]**. This shows field-sensitive answering even though the original sentence remains available. Following a swapped field is not credited as correctness.

For generated fields with the sentence in fresh context, requested-field following is 27/44 nondegenerate outputs with intact fields versus 14/44 with swapped fields. Four outputs have identical subject/object values and cannot distinguish field identity. Swapping distinct fields changes the answer in 8/44 paired comparisons. Reversing field order changes answers in 3/24 pairs. A simple always-first or always-last rule is insufficient, but position-sensitive mixtures remain possible.

## Transcript effect is a visible-context package effect

With the sentence present, transcript packaging reduces exact accuracy by **39.58pp [-56.25,-20.83]**, accompanied by compliance falling from 42/48 to 18/48. Outputs often turn questions into statements or use placeholders, e.g. “Someone was influenced.” Without the sentence, transcript outputs are all incorrect/noncompliant.

This changes visible instructions, token positions, turn boundaries and conversational attribution. It does not isolate hidden-state interference or prove that semantic information disappeared. Context length, instruction locality and source grounding remain confounded. The sentence×transcript risk interaction is **+8.33pp [-8.33,+25.00]**, unresolved. Its accuracy floor also limits interpretation.

The token-identical technical control recomputed the same sequence with full versus split cached prefill: maximum next-logit difference **0.0**, same next token. Twelve original extraction responses also replayed exactly. No difference is attributed to tokens having been generated rather than supplied.

## Compute and scoring limits

All arms fit the same per-event cold-query budget under a declared architecture-based forward-work proxy. The pipeline is charged its full extraction cost for each question; review is charged its first and second passes. Decode allowance is set before each call from the remaining ceiling. All 540 calls completed without budget or length termination.

Mean proxy work per answer: direct 0.264 trillion; self-review 0.623 trillion; fresh sentence-plus-generated-fields 0.563 trillion. The pipeline uses about 2.13x direct work and 0.90x review work. Equal ceilings are **not equal consumed compute**; this is budget matching under a proxy, not certified exact hardware FLOP matching. Exact token counts and CPU wall time are logged. The stronger prior single-pass variant costs about 0.253 trillion under the same proxy, well below the pipeline.

Source-span EM uses original human QA-SRL accepted answers. Referent accuracy additionally uses frozen AI-provisional aliases and article normalization; it is not independently verified Agent/Patient labeling. Grammar and semantic roles are not equated. Compliance is an operational nonempty contiguous phrase from available sentence/fields without the target verb, not a full linguistic noun-phrase judgment. Field-following is scored separately and excludes degenerate identical fields. All raw outputs are retained.

## Earlier interface gains versus harms

On the full prior 24-event factorial:

| Change | Newly correct | Newly incorrect | Net |
|---|---:|---:|---:|
| Answer prefix alone | 7 | 0 | +7 |
| Explicit instruction alone | 8 | 1 | +7 |
| Both | 8 | 1 | +7 |

Five prefix gains shorten clauses to the requested subject phrase; another changes a determiner/span, and one repairs a reflexive answer. Explicit instruction also repairs participant choices but introduces a genuine error: the contacted person changes from “Indonesian President Joko Widodo” to “Tony Abbott.” Equal totals are not equal rescues or uniformly improved relational understanding. INTERFACE_TRANSITIONS.csv preserves all 144 item-condition comparisons.

## Decision and next experiment

No evidence here establishes a new relational-computation advantage, question interference or multiple internal mechanisms. The narrow supported result is **partial use and imperfect question-conditioned selection of supplied fields, with a large visible-interface vulnerability**.

The next informative test is a small **query-to-role versus local answer-contract experiment**: hold correct reference fields fixed, cross the original QA-SRL question with an explicit grammatical-role query, and cross span instructions in the earlier system context versus the final user turn. Counterbalance field order and retain a direct-QA baseline that restates sentence and question in its final turn. A typed role query is an oracle diagnostic, not an autonomous repair.

If only local instructions rescue answers, prefer instruction/context adherence. If the explicit role query rescues the opposite-field errors while natural questions do not, query interpretation is the stronger lead. If failures persist with explicit roles and compliant spans, investigate label-to-field selection/position before changing representations or training. This test is proposed, not run. No new inference followed inspection of this study's outcomes.

## Reproduce

Run `analyze.py` for frozen-run scoring and `sensitivity.py` for explicitly post hoc prior-interface comparisons. Neither performs model inference. PROTOCOL.json fixes generation design/source/model hashes; the pre-inference budget-reference revision is transparently preserved in pre_inference_draft/. ANALYSIS_IMPLEMENTATION_RECORD.json records that analysis code was implemented during generation before inspecting condition scores. COMPLETE.json and CHECKSUMS.json provide integrity records. Original records remain unchanged.
