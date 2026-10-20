# One prospective causal development candidate

## Scope and frozen choice
The completed native screen (86/108; 15/36 complete; 28 eligible pairs) passed its minimal development feasibility gate. This stage tests ONE joint participant residual replacement at the outgoing boundary of Qwen block 14 (index 13), at the last subtoken of each of the three person mentions. Selection of this site narrows the previously unexecuted six-candidate proposal before any causal outcome. No alternative block, period, attention, MLP or site is tried if it fails. No held-out or Llama inference occurs in this stage.

All annotations are AI-only exact-template grammatical labels. The baseline error analysis and selection are exposed development information. This is candidate discovery and adversarial falsification, not confirmatory generalization.

Select the lexicographically first native-eligible cross-form transposition pair for each development name family and each invariant role. Sort key: recipient assignment, donor assignment, recipient form literal string, donor form literal string. Six units; each is tested in both directions and under all three future questions. All six native answers of each main pair must have been correct. Directions/questions from a pair are dependent and never counted as independent documents.

## Conditions and predictions
Four conditions per direction/question (144 native generations total):

1. **main**: three donor person states replace recipient person states in surface mention order. No semantic reordering of vectors.
2. **same_role_voice**: donor has recipient assignment but main donor form. Role-table prediction preserves all recipient answers; positional prediction follows donor mention names interpreted in recipient syntax. These control sources are not required to have correct native answers: native failure does not demonstrate source information absence. Their native outcomes are recorded, including failures.
3. **neighbor**: same main donor, but replace three preceding non-person tokens. For each person span take the nearest preceding token outside ALL person-name spans. All three sites must be distinct. This is a matched locality diagnostic, not proof all other sites are irrelevant.
4. **cross_lexical**: other development family, main donor assignment/form. This diagnoses literal donor content versus recipient lexical addressing. Foreign names may be absent from the recipient and are reported as valid diagnostic outputs. It is not an inert unrelated-text sham and is not a qualification gate.

`JOBS.json` freezes literal predictions for every condition:

- role_table: donor assignment queried by requested grammatical role;
- mention_order: donor names in surface order addressed using recipient grammatical-to-mention mapping;
- fixed_one_answer: donor name occupying the transposition's invariant role, repeated for every question;
- syntax_aware_parser and complete_future_answer_table: identical predictions to role_table. This experiment cannot distinguish those implementations. It cannot establish a unique semantic mechanism.

The input inventory contains three cyclic source orders. The frozen deterministic selection yields active/goal-fronted pairs in all six units; both directions are tested. Passive sources are recaptured for validation only, not tested causally here. This limits syntax generalization. Role-table and mention-order predictions differ on every main answer. The invariant query supplies a same-answer/different-relation comparison; two changed roles prevent one fixed name from satisfying all future demands.

## Capture and technical gates
Before edits, recapture all 36 development source contexts once with canonical sender question. Save 29 stages x 8 sites (three person anchors, three neighbors, period, assistant end), plus full-vector native logits. Compare each to the already saved native source: full-logit maximum error <=0.05, identical argmax, shared-anchor maximum error <=1e-6, all 28 source-prefix hashes identical. Cache each donor once and reuse across future questions, never choose query-specific donor states.

Identity replacement on every distinct selected recipient prompt must preserve complete full-vocabulary logits (<=0.05 and identical argmax), all shared traces (<=1e-6), and every source-prefix hash. This validates the implementation and same-source sham; it does not prove the transformed donor is on distribution. Before-block-14 recipient traces/hashes must be unchanged. At block 14 the three replaced vectors must equal saved donor vectors (<=1e-6). Patched source-prefix hashes must remain identical across future questions. Every generation continuation must reproduce its own intervened original-prefix full logits within0.05. Stop immediately on any technical failure, without automatic model retry.

Model/prompt/native full-name scoring match the frozen native screen: fixed shape 1x128, greedy <=12 decisions, EOS required, strip only whitespace, no candidate restriction for generation. Secondary endpoint is leading-space full-name-prefix log probability (not calibrated native answer posterior; excludes EOS). Cross-lexical secondary scores cover only recipient names and cannot estimate donor-name mass.

Every forward is journaled before dispatch and its raw full vectors saved before scoring. Resource limits: disk reserve4GiB, swap<2GiB, MLX6GiB guideline, cache256MiB,32 actual forwards per worker including exposed restart reference,90s per forward,45-minute dispatch-start window, no retry. `LOCK.json` fixes a conservative call cap from per-job prefix bounds plus all recaptures/identities and restart controls. No paid APIs, downloads, training, alternative model or held-out prompts.

## Prespecified decisions and stopping
The strict joint account endpoint includes main and same_role_voice:72 answers. Each account must predict ALL72 to qualify. Report all individual answers, all six bidirectional six-answer units, and each three-answer direction separately. No fitted threshold or p-value.

A **role candidate** qualifies for a separately frozen held-out protocol only if all technical controls pass, all36 main answers and all36 same-role answers match role_table, and neighbor full-role transfer occurs on at most2/6 complete bidirectional units. This is a development operational gate, not a universal necessary criterion for linguistic computation. Cross-lexical outcomes are diagnostic only. A perfect joint mention_order account with valid technical controls yields NARROW positional/answer-source steering. Otherwise NO-GO for this chosen candidate. Failure ends this stage with no adaptive depth/site/prompt/scoring change and untouched evaluation families. A positive role gate would still retain syntax-aware parser/complete answer-table equivalence and require prospective evaluation before any generalization claim.

Independent arithmetic replay uses float64 full-vocabulary normalization with tolerance1e-8, fixed before this causal collection. Report all technical errors, surprising outputs, failed controls and limitations. These are two small development lexical families and one verb/construction task; no novelty or population significance claim.
