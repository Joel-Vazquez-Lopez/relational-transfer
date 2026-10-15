# COMPUT-010 — event-level falsification gate

Date: 15 October 2026. **Status:** executable local auditor and 9 passing local synthetic verification tests; no empirical replay of original Qwen predictions.

The original per-answer 384-row patch records are still not accessible in GitHub or Comput's workspace. The reported 368/384 participant-position fit remains **archival**, not independently reproduced.

[Complete executable package is available in the Comput conversation artifact: COMPUT_010_EVENT_LEVEL_FALSIFICATION.zip.] This GitHub workstream retains this report and the runnable baseline implementation below.

## Data contract
Supply authorized, unsealed JSONL containing event_id, group_id, predicate, sentence, participant_a, participant_b, question_role, construction, model_answer, gold_answer. Gold is for scoring only. Strictly count model outputs that exactly match one of the two participants as format-valid. Keep format validity, model correctness, model-to-heuristic agreement, and heuristic-to-gold correctness separate. Hold out participant groups when selecting the best rule; do not select a rule using the held-out responses. Report training ties instead of optimistic cherry-picking.

Five simple rules: first-mentioned, last-mentioned, first-for-actor else last, last-for-actor else first, and construction-conditioned first/last. The last baseline has an externally specified construction tag, thus it is **not yet an autonomous raw-text predictor**. Text-derived construction extraction is still an essential independent validation gate.

## Negative controls / limitations
- A hand-coded construction-conditioned positional router achieves 24/24 on a 3-pair, 4-form toy verification; always-first achieves 12/24. These are tests of the evaluator, not pretrained behavior.
- Missing raw model predictions preclude recomputing the archival 368/384 result or inspecting the 16 residual failures.
- The historical active-only intervention cases confound order with semantic roles.
- No model weights, paid inference, sealed data or training were used.

## Minimal next step
Obtain the permitted original `llm_role_internal_v2/RESULT.json` with linked stimulus/output records, verify source hashes, and transform unsealed per-answer rows into the audited JSONL schema. Independently recompute the archival agreement, then analyze held-out group residuals, order-role counterfactuals, and answer-format failures.

## Commands
`python3 -m unittest discover -s . -p 'test_*.py' -v`
`python3 comput010.py /path/to/unsealed_predictions.jsonl --output results.json`
