# External validation gate and a concrete baseline-correspondence finding

20 October 2026. This follows the completed local evidence audit. **No external empirical replication was possible from the audited artifacts. A separate static code audit identified a reproducible baseline-correspondence defect in Dai et al.'s released notebook.** These are different results and must remain separate.

## What was attempted

Before fetching repository metadata or inspecting outcome arrays, `PROTOCOL.md` fixed six candidate studies in relevance order, an availability gate and a stop condition. `AVAILABILITY_LOCK.json` preserves the local preinspection timestamp and source hashes. This was not an externally registered preregistration. No source-specific numerical analysis plan was activated, because no source passed availability.

We pinned all six repositories to commit SHAs, inspected their complete default-branch trees and release manifests, and fetched 565,086 bytes of selected source files. We also fetched the official 2,738,763-byte ACL software ZIP for Dai et al. and the official ROME data listing. No model weights, paid calls, downloaded code execution, training or Transformer forwards were used. Notebook code was inspected as text and AST; output values were not used as an empirical dataset.

### Availability outcome

| Prespecified study | What was actually available in the inspected artifacts | Gate |
|---|---|---|
| [Dai et al., binding](https://github.com/cl-tohoku/OI-Subspace) | Six input datasets and notebook with aggregate displays; no joined per-example baseline/intervention archive | Fail |
| [Zhang & Nanda, patching practices](https://arxiv.org/html/2309.16042v2) | Paper-linked IOI codebase; attribution and required per-example outputs unverified | Fail |
| [Fierro et al., multilingual recall](https://github.com/constanzafierro/multilingual_factual_memorization) | Collection code filters exact-match examples; required inference/causal output files absent from pinned tree | Fail |
| [Nikankin et al., arithmetic](https://github.com/technion-cs-nlp/llm-arithmetic-heuristics) | Code and figures; two relevant notebooks have no retained outputs; referenced local result arrays not in tree/releases | Fail |
| [Jenner et al., chess look-ahead](https://github.com/HumanCompatibleAI/leela-interp) | Code generates arrays locally; Figshare returned an empty response here, leaving its inventory unresolved | Unresolved access; fails current gate |
| [Meng et al., ROME tracing](https://github.com/kmeng01/rome) | Collection/aggregation code and input/statistics listings; default runner does not measure interventions for baseline errors | Fail |

`AVAILABILITY_LEDGER.json`, repository inventories and fetch manifests preserve the exact evidence and qualifications. This is **0/6 eligible sources under the inspected scope**, not proof that no reusable public data exists. Empty responses and artifact-attribution uncertainty are recorded as uncertainty. Repositories can change after their pinned commits; uninspected branches, external archives and author-held results may exist.

The gate matters scientifically: reweighting a correct-only archive cannot recover intervention outcomes that were never measured for incorrect cases. Input datasets and executable collection scripts do not substitute for those missing outcomes. Restricting a study to successful recall can still be appropriate for its stated conditional question; this audit does not show that those papers' conclusions are biased.

## A concrete finding in the closest binding precedent

The official [Dai, Heinzerling & Inui (2024) software supplement](https://aclanthology.org/attachments/2024.emnlp-main.967.software.zip), linked from the [paper page](https://aclanthology.org/2024.emnlp-main.967/), contains `Binding_ID_subspace_base_activation_patching.ipynb`. In zero-based cell 23, `resid_act_patch_pca_ld` loops over intervention steps and then over examples. It assigns the cached baseline vector only while the step is zero. The next step subtracts that single cached vector from every example. By then it contains the **last example's** baseline, not each recipient example's own baseline.

This is a source-inspection discovery, not a preregistered finding or an observed effect in the original models. `baseline_certificate.py` parses the notebook without executing any downloaded function. Its AST checks verify the unique baseline assignment, the surrounding loops/condition, and the subtraction's operand. Four fixed authored arithmetic examples then verify the consequence with 12 checks; all pass. The raw fixture inputs and outputs are saved in `BASELINE_CERTIFICATE.json` and are explicitly synthetic.

### Exact consequence

Let b_i be the candidate-logit vector for baseline example i, y_i(s) its vector at a nonzero intervention step s, and n the last example. The released loop computes

`reported_i(s) = y_i(s) - b_n`

instead of the paired quantity

`paired_i(s) = y_i(s) - b_i`.

Consequently,

`mean(reported(s)) - mean(paired(s)) = mean(b_i) - b_n`.

This offset is the same at every nonzero step, but the code's step-zero differences are all zero. Changing the final example can change the offset. Error bars also differ: subtracting one constant gives the dispersion of intervened logits, while subtracting per-example baselines gives the dispersion of paired changes.

For an authored three-example fixture with **zero true intervention effect**, the stale-reference computation reports mean changes `[3, 1.6667, -7]`. Reordering the same examples changes those values to `[-5, 3.6667, 4]`. The correct paired means are `[0, 0, 0]` in both orders. These numbers demonstrate the implementation defect; they are not estimates of its effect in Dai et al.'s data.

### What this does and does not affect

- Absolute mean logit changes relative to step zero and the plotted standard deviations can be wrong under this implementation.
- Differences of means between two nonzero steps cancel the constant reference offset; this particular defect does not change those contrasts.
- The label-proportion calculation uses the argmax of the intervened logits directly. It does not use the baseline subtraction, so that endpoint is unaffected by this particular issue.
- Actual empirical distortion cannot be recovered from aggregate figures without the appropriate per-example records.
- We have not established that this released notebook generated the final published figures. This finding does not invalidate the paper's binding-subspace claim or its independent evidence.

`PAIRED_BASELINE_FIX.md` gives a concrete correction keyed to the existing example counter, recommends stable IDs, and specifies the records needed for empirical verification. It is a proposal for review; no upstream file was modified and no author was contacted.

## What this means for our research

The strongest empirical result remains our completed 24-family local confirmation: readout-dependent construction admission changes the conditional causal comparison. This external search has **not** independently replicated that result. It would be misleading to call static code inspection an external model replication or to convert unavailable observations into a null result.

The new useful result is more specific: the released binding notebook has a baseline-pairing defect with a provable effect on particular summaries. That supports including an explicit **example correspondence audit** alongside readout definition, sample admission and denominator support in our methods. It does not create a new calibration theory or a linguistic mechanism. We have not established priority over any previous bug report or private correction.

Our next empirical claim should concern a fixed population with paired native and intervention measurements retained for both successes and failures. Where the scientific target is successful retrieval, report that conditional target directly. Where the target is a cross-model or cross-construction comparison, preserve shared support and construction-stratified results. These are measurement requirements, not evidence of a universal common path through models.

**Decision:** stop the external numerical reanalysis at its failed availability gate. Preserve the completed local result and this separate static implementation finding. Do not expand the candidate list based on a desire to find a positive filtering effect, silently run a new synthetic panel, or claim that the published studies have been overturned. A later external replication requires actual paired records or a separately scoped faithful collection.

## Reproduction and preservation

`inventory.py` retrieves pinned repository metadata; `fetch_sources.py` verifies Git blob hashes before saving selected source files; `fetch_supplements.py` retrieves bounded official attachments/listings. The initial ACL URL guessed without the `/attachments/` directory returned 404; the actual paper link was then followed successfully. `SUPPLEMENT_FETCH_INITIAL.json` preserves that failed attempt. The zero-byte Figshare response remains unresolved. No credentials are saved.

Run `python3 research/external_admission_audit_2026-10-20/baseline_certificate.py` after fetching the recorded official ZIP. It uses only Python's standard library, never imports the notebook, and writes the static certificate and complete synthetic fixtures. `verify.py` checks all saved provenance and results locally; all 165 checks pass.

Public preservation includes our scripts, protocol, manifests, schemas, certificate and report. Third-party source snapshots, extracted notebook text and the downloaded ZIP remain local; the manifests and official links support exact retrieval without republishing those materials. Their local preservation is distinct from the audit's public artifacts.
