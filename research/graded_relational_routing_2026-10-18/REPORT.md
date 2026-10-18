# Graded relational routing — complete bounded investigation

Completed 2026-10-19. Source protocol frozen in commit `8a338cf1d9a1103db82ed3129322fadfa4c0d7ca`, before new outputs. This continuation changes the endpoint explicitly; it does not revise the stopped carrier pilot or any earlier failure. No further inference is authorized by this report.

## Decision

**NO-GO for the prospectively specified general route signature.** Both runs and all fixed evaluation panels completed. Qwen's selected 14→21 source-release interval passed only 5/8 evaluation family/voice cells and 2/4 complete families. Llama had no qualifying development candidate. Neither is a demonstrated new linguistic mechanism. There are useful descriptive dependencies to preserve, including robust graded sensitivity despite imperfect native labels, heterogeneous source dependence, and checkpoint differences. These do not reverse the primary decision.

## What was measured

Six fresh name/verb families per checkpoint, two development and four evaluation. Each contains two role assignments × active/passive × two question orientations. Gold annotations are deterministic AI-generated simple-grammar labels, checked independently from the tokenization and scorer; these are synthetic sentences, not independently annotated natural text. No families were discarded for wrong answers.

Native response means full-vocabulary next-token argmax in the gold Yes/No alias class with a positive class margin. This is not a multi-token generation test. A single threshold per checkpoint was fixed from eight already exposed predecessor scores, then transferred without refitting. Calibrated classification must not be called native correct answering.

For each four-answer family/voice cell, s = logsumexp(Yes aliases) − logsumexp(No aliases); D=(s00−s01−s10+s11)/4. The contrast cancels constant response bias and additive role-only/query-only terms. Positive D and all four positive conditional ordinal contrasts establish this task's role-consistent graded sensitivity, not an abstract semantic representation. Normalized intervention retention is D_masked/D_native only above the frozen magnitude floor. Raw and normalized responses are both preserved.

Masks begin at blocks 0,7,14,21 (zero-based): closed sentence access; source-isolated period; period allowed to reread; translated neutral deletion. Neighbor-token control at14; no-op shams at28. See PROTOCOL.md for exact graph definitions. Post-sentence positions include formatting, question and assistant header. The current experiment captures embeddings and every block output at period and answer anchors, not all tokens or separate attention/MLP outputs.

## Full collection and independent verification

879 forwards/checkpoint, 1,758 total, under the frozen 1,800 budget. Per model:48 baselines,816 intervention calls,12 shams,3 validations. All succeeded; no recorded technical failures. Production controls:1,936/model, all passed. Independent saved-data replay:3,419 checks, all passed; all1,758 full-vocabulary vectors re-scored, payload hashes verified, annotations independently parsed, summary contrasts/selection/prediction chronology replayed, cut0 disconnections and pre-cut/live/sham controls checked. No new inference was used in analysis.

`analyze.py` is a post-run analysis script; it does not belong to the frozen intervention code. No frozen source was changed. Figures were visually inspected. Full-prefix arrays used only for controls were saved as hashes, so their original arrays cannot be recovered from those hashes; measured two-anchor traces and full output vectors remain available.

## Normal responses: answer-level and family-level outcomes

| Checkpoint | All native | All calibrated | Evaluation native | Evaluation calibrated | Evaluation graded-sensitive cells |
|---|---:|---:|---:|---:|---:|
| Qwen7B |36/48|48/48|25/32|32/32|8/8|
| Llama3B |36/48|47/48|25/32|31/32|8/8|

The four evaluation families are g03 Nora/Caleb followed, g04 Iris/Leon thanked, g05 Elena/Marcus visited, g06 Zoe/Adam praised.

| Family | Qwen native /8 | Qwen calibrated /8 | Llama native /8 | Llama calibrated /8 |
|---|---:|---:|---:|---:|
|g03|6|8|6|7|
|g04|7|8|8|8|
|g05|6|8|4|8|
|g06|6|8|7|8|

Thus 0/4 Qwen evaluation families and 1/4 Llama families have all eight native answers correct; calibrated complete-family counts are4/4 and3/4. These are classification outcomes, distinct from primary causal-family success. All errors, unsupported answers, intervention-induced answers and calibrated errors are retained in ANSWER_LEDGER.csv and FAILURES.json. No binomial significance is claimed from correlated template cells. Unfamiliar lexical items and names are not a genuinely held-out linguistic construction or population.

## Qwen prospective test, with every failing unit

Development selected the suffix interval14→21, after neither period-carriage signature met all development criteria. Prediction was persisted and development raw hash fixed before any evaluation output; the independent audit reproduced it.

Frozen evaluation requirements: native sensitivity valid; closed retention≤.25 at14 and≥.75 at21; neutral retention within.25 of1 at both. Complete family requires both voices.

| Family/voice | Native D | Closed14 | Closed21 | Neutral14 | Neutral21 | Result |
|---|---:|---:|---:|---:|---:|---|
|g03 active|5.688|.293|1.045|1.482|1.007|FAIL: early retention and neutral drift|
|g03 passive|19.406|.144|1.001|1.018|.998|pass|
|g04 active|11.172|.037|1.007|1.116|1.004|pass|
|g04 passive|15.660|.008|1.019|1.039|.999|pass|
|g05 active|7.223|.577|1.015|1.065|1.007|FAIL: substantial earlier retention|
|g05 passive|18.574|.256|1.002|1.001|1.000|FAIL: .256 exceeds .25; criterion unchanged|
|g06 active|5.148|.091|1.016|1.082|1.007|pass|
|g06 passive|20.680|.043|1.019|.999|.998|pass|

Ratios are rounded only for display; calculations use unrounded values. All8 upper-endpoint responses survive source closure at21, but the selected interval as a whole fails. Do not rename that component observation a primary success. Closed0 and source-isolated0 yield exact opposite-role nulls.

## Llama and the complete nonselected controls

No development candidate met the frozen cached/refresh/suffix criteria, so there is no positive prospective Llama mechanism result. All fixed evaluation conditions were nevertheless collected. Its closed14 evaluation retentions range1.056–1.281 versus Qwen .008–.577, a descriptive checkpoint difference consistent with earlier availability in the post-sentence state under these masks. It is not a family-wide timing law. Llama neutral7 ranges.979–1.641; even unrelated deletions can strongly amplify the target contrast. Picking held cut14 now would violate the protocol.

At14 Qwen source-isolated period retention ranges.134–1.177, live period .283–.935, neighbor .053–1.071; Llama source-isolated .976–1.055, live .991–1.038, neighbor1.058–1.284. This heterogeneity does not establish a unique punctuation carrier or monotonically helpful rereading. Every individual value, including negative and amplified effects, is in INTERVENTION_LEDGER.csv. CAUSAL_PROFILES shows every evaluation family and unweighted means without clipping or timing alignment. NATIVE_DEVELOPMENT shows the full residual interaction norm; rising norms are descriptive and coordinate/magnitude dependent.

## What survives and what does not

Supported within this panel: (a) both checkpoints distinguish the controlled relationship in graded final scores in all8 evaluation cells despite wrong raw labels; (b) the fixed external threshold transfers well; (c) original-sentence access before different cut depths matters differently; (d) once later source reads are removed, already existing post-sentence states can support the final contrast. None implies that the original model ceases reading, that a semantic computation is finished, that information is localized to the answer token, or that any specific component stores a binding variable.

Competing explanations still include syntax/marker-aware lexical copying, distributed preparation of the answer, and alternative routes recruited by attention renormalization. Closed/frozen/live masks alter attention distributions, not only payload access. Equal edge counts do not match removed attention mass, positions or representational content. Period source isolation is not numerical freezing or period-only sufficiency. Role and question contrasts do not cancel arbitrary contextual biases. Active/passive changes order, markers and length together. One locally quantized checkpoint per family cannot establish an architectural family law. No causal interchange of semantic variables, head-specific route or attention/MLP decomposition was tested here.

Calibration is established prior work (Zhao et al.,2021, https://proceedings.mlr.press/v139/zhao21c.html); graded metrics and intervention choices affect conclusions (Zhang & Nanda, https://arxiv.org/abs/2309.16042). Attention routing/knockout is established (Geva et al.,2023, https://aclanthology.org/2023.emnlp-main.751/). Dai et al.(2024, https://aclanthology.org/2024.emnlp-main.967/) already causally study binding information; our assay does not identify a distinct binding mechanism. Mixing Mechanisms (https://arxiv.org/html/2510.06182v2) distinguishes pointers, lexical/positional routes and retrieved answers with counterfactual and knockout controls; its discussion of neighboring-token relays is particularly relevant. Repeated source lookup is also already studied in Language Models Use Lookbacks to Track Beliefs (https://arxiv.org/html/2505.14685v3). We claim neither a new calibration method nor discovery of the general availability/use/readout distinction. A new mechanism or paper-level novelty is not established.

## Preservation and reproduction

PROTOCOL/LOCK/input/calibration hashes preserve pre-output commitments; BUDGET, run logs, per-model RAW.jsonl and every CALL_*.npz preserve the complete local execution. `analyze.py` reproduces the audit and figures from those local payloads without loading a model. Its full replay requires the local arrays. All call-wise score records, errors, control outcomes, manifests, scripts and reports are committed; PUBLISHED_NATIVE_qwen7b.npz and PUBLISHED_NATIVE_llama3b.npz preserves baseline trajectories remotely. The approximately970MiB of all-call full-vocabulary/trace NPZ payloads remain local, indexed by complete SHA256 manifest; they are not claimed to be uploaded to GitHub. This limits remote full-vector replay, although the score-level results and baseline trajectory figures can be reconstructed remotely. No old scientific outputs were deleted. Runtime-storage and immutable-payload deduplication receipts document lossless disk safeguards during the run.

The bounded experiment is complete. The Gallistel follow-up is a separate theoretical/design review; it will not dispatch inference.
