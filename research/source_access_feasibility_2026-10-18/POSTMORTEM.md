# Scoring-design failure — owned by the investigator

The bounded run stopped after **11 attempted Qwen forwards**: eight distinct development baselines and three validation calls. It ran **zero substantive source-access interventions, zero evaluation calls and zero Llama calls**. The native-wrapper/dense-mask/repeat validation discrepancies were all exactly zero. This establishes those numerical comparisons only; native source-isolation and restoration controls were not reached.

## What failed

The frozen scorer reports 6/8 development answers passing its next-token endpoint. Both rejected baseline outputs were `N` when the expected person was Nina. Qwen's tokenizer encodes:

- `Nina` as `[45, 2210]`, decoded as `N` followed by `ina`.
- ` Nina` as `[63888]`, a different, single-token realization with a leading space.

The preflight allowed a person if *any* natural realization had one token. The runtime then required the full-vocabulary first-token argmax to match a single-token full-name alias. This combination is inadequate: it can reject the beginning of a perfectly valid multi-token answer. Helen has the same kind of issue in the unexecuted Qwen evaluation manifest.

The stored top logits make the problem clear: `N` was the highest token on both affected baselines. The continuation was not generated or scored. Therefore we cannot conclude either that Qwen completed Nina correctly or that it failed to identify Nina. Crediting the prefix as a correct full answer would be another scoring error.

This is a measurement-design defect, not evidence against the linguistic or source-access hypothesis. The result is **NO-GO for this scoring protocol**. The gate's machine name `STOP_DEVELOPMENT_COMPETENCE` is preserved for provenance; the scientific interpretation is scoring incompatibility, not demonstrated model incapacity.

The second lexical family, Emma/Jack, passed all four frozen endpoints. Selecting that family alone now would violate the frozen all-family rule, so it was not substituted.

## Responsibility and historical lesson

Earlier research already encountered whole-name versus first-token scoring failures. I should have enforced that lesson at preflight. Having one single-token alias does not make every likely answer realization single-token. The synthetic integration tests checked the declared scorer and persistence but did not test its adequacy for realistic multi-token answers. Their passing status is preserved and does not excuse this omission.

## What was preserved

All frozen sources, input/tokenizer manifests, pre-inference commit, 11-call journal, every raw next-token result, full-vocabulary logits, available native captures, validation results, both rejected baselines and complete-family outcomes remain intact. No prompt, name, alias, hypothesis, scoring rule, threshold or intervention location was changed after results. No hidden retry or additional continuation was run.

## Requirement for a future protocol, not a continuation of this run

Choose a scoring endpoint that handles all specified realizations **before** fresh model outcomes:

1. Prefer reuse of the existing verified full-name sequence scorer, with exact candidate conditioning and a separate actual-generation correctness check. Every candidate/continuation forward must count against a new explicit budget; probability of a name prefix is not probability of a terminated answer.
2. If a single-token endpoint is essential, statically reject names unless every accepted response realization, including the exact no-leading-space form, is genuinely one token in each tokenizer. Validate that restriction in preflight rather than discovering it through competence outcomes.
3. Add an adversarial integration fixture where the highest token is the first part of a correct multi-token name and a competing person's complete name is a single token. A next-token-only scorer must flag this as unassessable or reject the design before inference, never label it established linguistic incompetence.

The new, separate `scoring_guard_fix.py` implements the stricter static rejection rule without changing any frozen file. `SCORING_GUARD_TESTS.json` records its passing multi-token adversarial fixture and rejection of four unsupported model/name entries in the original manifest. It made zero model calls. This guard prevents repetition of the defect; it does not implement or validate a full-name sequence scorer.

No replacement study is executed or implicitly authorized by this postmortem. The completed attempt remains closed. The broader source-access idea is untested here; novelty remains unresolved and close prior attention-knockout work remains relevant.
