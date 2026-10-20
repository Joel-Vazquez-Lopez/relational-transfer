# Role versus mention position in causal source access

New authorized investigation,20 October2026. The previous failed gates, native stop, claim corrections and unused t03/t04 remain unchanged. One fixed local Llama3.2-3B4bit study. No Qwen, paid API, model download or training. No guarantee of a positive result or novelty.

## Scientific target

Does a query's sensitivity to replacing source states follow the queried grammatical role across changes in construction and mention position, beyond generic depth, position and native-output predictors? This measures intervention-dependent access to information, not when a semantic variable is constructed, an internal matching algorithm or universal model-family stages.

The four source names are all people. Three fill sender/sent-person/destination roles; the fourth waits nearby. In each counterfactual, exactly one queried participant is exchanged with the unrelated person; the other two queried answers stay fixed. Unlike three-for-three exhaustive decoding, those two invariant answers cannot determine the changed answer. No constrained assignment decoder is used in the primary analysis.

## Fixed factorial and linguistic controls

Two new quartets: Grace/Helen/Jack/Kate and Nora/Luke/Paul/Mark. Four cyclic name rotations. Three forms:

- A sent B to C.
- B was sent to C by A.
- To C, A sent B.

These cycle every role through all three event mention positions. Append `D waited nearby.` to every passage; this participant is always last, an explicit limitation. Each quartet uses a different question paraphrase set, fixed before outcomes. This deliberately tests combined lexical/paraphrase transfer; it does not estimate their separate effects. Every participant appears once and occupies one complete tokenizer token. The four rotations balance individual name identity across roles and distractor status, not all ordered name pairs.

Per quartet/rotation/form: base0123, swap each event role with slot3, plus a same-answer sham changing only `waited` to `stood`. Five sources × three role queries gives360 native prompts:288 primary-panel plus72 sham. AI-only authored and independently parsed; grammatical task roles, not human-reviewed corpus annotations. Every case is retained. Native and edited full responses use unconstrained vocabulary-wide argmax, ≤4 token decisions and EOS, whitespace strip only. Unknown/truncated/incorrect responses remain failures.

Q1 questions: `Who sent someone to another person?`; `Who was sent to another person?`; `Who was someone sent to?`. Q2: `Which person did the sending?`; `Which person was sent by someone?`; `To which person was someone sent?`. No failed `recipient` label. Questions vary in length; actual positions and question lengths enter the nuisance competitor. No wording changes after outputs.

## Native prerequisite and collection stop

Before interventions: at least90% of the288 primary native responses must be complete correct first-token-name responses; each quartet×form must have at least80% correct. Each quartet×form×changed-role cell needs at least3 of4 rotations with both target-query endpoints complete, correct, and native B-minus-A logit separation≥1. These eligible contrasts are determined solely from native outcomes. Record all failures and supports; never replace a pair. Sham native performance is reported separately. Any failed prerequisite stops before causal collection. All native prompts are collected unless a resource/technical guard fails.

## Fixed intervention and controls

Native full residual banks at embedding and outgoing blocks0/7/14/21/28, plus all29 readout states, use physical1×128 right-padded shape and actual prompt end. At cut7/14/21 partition the prompt into P (everything through the final source period) and Q (question and assistant tokens through the original readout). Within each same-form pair, all tokens/positions match except two whole name tokens. Record native AA and BB. Compute the two unique hybrids BA=(P_B,Q_A), AB=(P_A,Q_B), both in the A input-token context. Original-prompt states only are replaced; new generated positions retain their native A-context computation. This affects interpretation of continuation equivalence; only original-prompt full-vocabulary equality is an architectural positive control.

For all72 role-isolated edges, evaluate the changed query at all three cuts, both hybrids:432 first-token measurements. At14, also evaluate both invariant queries:288 more first-token measurements. Generate complete responses for all432 cut14 hybrids. At7/21 the endpoint is explicitly the next-token vector, not a claimed full answer. For all72 sham prompts, replace only P at14 and generate the complete response. No semantic projection, learned neuron map, role-based alignment, fragment replacement or adaptive site search.

Technical controls use rotation0/edge0 for all six quartet×form strata and all three queries:72 architecture-forced endpoint forwards (two hybrids at0/28);108 identity/full-joint donor forwards (three intermediate cuts). Full-vocabulary max absolute error≤.05, repeat states exact, replacement float32 hashes exact, untouched generated/padded tail hashes equal, same source banks invariant across future questions. All pass before scientific edits. Shape/end/prefix invariance and restart references required throughout. Endpoints are controls, not discoveries.

## Prespecified primary measurement and predictions

For each edge's changed-role query, define m as B-name logit minus A-name logit. Let S=(m_BA−m_AB)/(m_BB−m_AA). Denominator≥1 is required. No clipping; nonlinear S can lie outside[−1,1]. Pure source dominance gives+1 and pure suffix dominance−1. This is an analyst-defined causal contrast of output scores, not an internal probability or amount of semantic information. Native eligibility additionally requires correct full responses at both endpoints.

Compare four fixed predictors of the3-cut S profile:

1. **Role:** training mean S for the queried sender/sent-person/destination role at each cut.
2. **Mention order:** training mean S for that role's event mention position at each cut.
3. **Generic schedule:** training mean S at each cut.
4. **Native-output/position nuisance:** per-cut ridge regression,lambda1, intercept unpenalized, training-only standardization. Features are native m_AA,m_BB, each endpoint's four-candidate softmax entropy, absolute queried-person token index, distance from that mention to source period, distance from source period to readout, and event mention-position one-hot. The candidate entropy is over the explicitly declared four possible next-name tokens, conditioned on that prompt; it is an external closed-set statistic, not a posterior over interpretations.

Cross-fitting is fixed: for each of six held-out quartet×form folds, train on the other quartet's other two forms and test on that held-out quartet/form. Thus every tested case has unseen names, question paraphrase and construction in that fit. Four rotations per role/form are the repeated cases; there are only two lexical quartets, not independent layer/answer replicates. Fit training means/regressions only on native-eligible contrasts. Do not tune parameters or choose a favorable fold/depth.

**ROLE_ACCESS_PILOT** requires: all technical/native gates; ≥90% terminated known-name cut14 scientific generations; ≥90% gold-correct invariant answers among native-competent invariant endpoints; ≥90% correct sham responses among correct native sham endpoints; at least75% of possible invariant/sham comparisons must have both endpoints native-competent; pooled held-out role MSE≤.75×the lowest pooled rival MSE; role error strictly below each rival within each of the six held-out folds; and pooled within-cut S residual standard deviation≥.15 (subtract each cut's pooled mean). Thresholds are descriptive pilot criteria, not a population significance test. No p-value or family-wide mechanism claim.

If native gates fail: **INCONCLUSIVE_NATIVE**; if technical/resource guard fails: **STOP_TECHNICAL_OR_RESOURCE**; if language/sham/invariant gates fail: **INCONCLUSIVE_INTERVENTION**; otherwise failing the role criterion is **NO_SUPPORT_FOR_ROLE_INDEXED_ACCESS**. An answer-based rival win is evidence for that rival on this panel. Do not call it a new internal algorithm. Plot all native readout response magnitudes and all three causal cuts without treating descriptive geometry or a secondary depth as primary success.

## Resources and preservation

Max4054 actual forwards including restart references,32/worker,90s/forward. Native and causal stages each≤45min. Normal memory pressure, free disk>8GiB, swap<4GiB and growth<1GiB; MLX3GiB guideline and256MiB cache. Stops are final, no automatic retry or prompt repair. All scripts, inputs, model/backend hashes, raw vocabulary vectors/states, controls, full available generations, native exclusions, predictor fits, failures and reports preserved. Freeze and verify before model inference. Commit to the existing research branch and link Issue35. After outcome, stop; no extra adaptive inference.

## Prior-work boundary

Aljaafari et al.(ACL2026), `Emergence and Localisation of Semantic Role Circuits in LLMs`, studies role-conditioned noun continuation, training-time development and cross-model circuits. Our panel asks about participant identity among same-type people with isolated role changes; it is not a replication or refutation of their circuits. Its different task does not by itself establish novelty. Dai et al.(2024), Feng/Steinhardt(2024), Mixing Mechanisms, Lookbacks(2025/26), calibration and causal-abstraction work remain strong precedents. A successful result would support a narrow, out-of-sample functional access profile under this intervention—not a unique representation, semantic circuit, complete computational map or ready paper.
