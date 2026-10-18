# Atlas pilot: authorized technical continuation

The user explicitly requested “fix it and test it then no?” after the documented serialization failure. The original attempt and its frozen sources remain intact in the sibling `linguistic_development_atlas_pilot_2026-10-18` directory and commit `878b715`.

## Amendment before additional inference

`AMENDMENT.json` records the directly authorized recovery. Its administrative sequencing/budget provisions supersede the original no-resumption/per-model-count provisions for this continuation only. All scientific inputs, scoring functions, instrumentation, analysis code, hypotheses, sites, gate thresholds and AI-only annotation decision are unchanged. `SCIENTIFIC_INVARIANCE.json` checks byte equality of those copied files and AST equality of scoring/input functions.

The two earlier calls count against the **132-call global ceiling**. The saved first Qwen main capture is reused without repeating it. The lost unwrapped validation is recomputed and the wrapped repeat is retained. Every newly attempted forward is saved to the shared durable budget journal before execution; validation logits are now persisted immediately. No automatic retry of a failed continuation phase is allowed.

Run both models' development panels first, followed by eligible Qwen held-out families, then eligible Llama held-out families in fixed order a03/a04. Require budget for an entire 16-call family. If both development gates pass, the arithmetic is **117 cumulative calls**, with Llama a04 left unrun because the remaining 15 calls cannot fit a whole family. Family choice is fixed before held-out outputs. If a model fails development, its held-out budget is unused and the other eligible model can complete both families. Never weaken the technical/competence gates to fit the budget.

## Fix and tests

`persist_result()` asserts input/scorer gold agreement and merges dictionaries using a literal. `test_integration.py` exercises the production persistence path for all 128 actual input rows using explicitly synthetic causal states, round-trips tensors/logits/JSON, checks duplicate/mismatch rejection, runs prefix controls, and rehearses the full JSON/CSV/Gram/six-figure analysis export. It also round-trips the actual previously saved Qwen capture offline. `INTEGRATION_TEST_RESULTS.json` records success; **zero model calls** were made by that test. The synthetic outputs are temporary and are not empirical model findings.

`LOCK.json` freezes this implementation before additional inference, explicitly acknowledging one exposed development cell and no exposed held-out cases. Models remain local; no paid calls or downloads. The original failure report remains an accurate record of the first attempt.

Commands (existing workspace interpreter):

```sh
research/stronger_model_development_2026-10-15/.venv/bin/python research/linguistic_development_atlas_continuation_2026-10-18/orchestrate.py
```

This runs fixed development/held-out phases sequentially, then the original descriptive analysis. It needs ordinary local Metal GPU access. The report must distinguish initial and continuation calls, reused capture, any skipped families, all errors and technical failures. A positive geometric interaction remains compatible with nonlinear marker/syntax-aware processing; no unique semantic mechanism or novelty claim is warranted by this pilot alone.
