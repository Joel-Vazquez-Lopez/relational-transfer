# Prospective Qwen study: role-contrast source access

20 October2026. This version supersedes the unexecuted hard-accuracy-gated Qwen design **before any new Qwen scientific output**. The archived version and SUPERSESSION_BEFORE_INFERENCE.json preserve chronology. It does not repair or relabel the failed Llama experiment. Model selection and hypothesis development used exposed Llama results; only this Qwen execution/prediction is prospective. One existing Qwen2.5-7B-Instruct4bit checkpoint, no downloads, paid calls or neural weight training.

## Why the question changes

The original Observatory asks about information availability, causal contribution and behavioral expression as distinct functions. Requiring correct hard answers excludes the very cases in which useful relative evidence might coexist with a biased answer. We will report native accuracy fully, but test a separately validated graded response. Calibration and availability-versus-use distinctions are established, not new theories.

The new statistic is a **joint contrast across three separately prompted questions**, not a native joint posterior, a quantity the model is claimed to compute or proof that each queried row independently contains its relation. Native gold answers may be wrong. A direct target-query response is required in addition to the joint contrast, which prevents a literally constant target row from qualifying solely through other questions. Shared source-dependent bias plus negative evidence in other questions can still confound allocation of information to a particular query; no unique internal representation is identified.

## Fixed data

Use the exact360 inputs/72 role-isolated edges of the initial Qwen preparation. Two name quartets, four rotations, active/passive/goal-fronted forms, base and three swaps with the fourth person, plus the waited→stood sham. All people appear once and all names are single tokens; lexical word sets are unchanged within each role-swap pair. Only one queried answer changes; the other two stay fixed. Two question wordings are tied to the two quartets, so transfer tests their combined shift. Individual identity is rotated, but not every ordered pair is covered. The distractor is always last. AI-only authored/template-parsed, one verb and one checkpoint. No human/corpus claim.

Native collection:288 primary plus72 sham full responses, maximum four token decisions, vocabulary-wide greedy decoding, EOS required, whitespace strip only. No forced candidate choice or score-based exclusions during collection. Save every full vocabulary vector, the five full prompt residual banks at0/7/14/21/28 and all29 native readout states.

## Measurement derived on exposed data

Let source A/B exchange the participant in role r with an unrelated fourth person. Let a/b be that old/new participant. For each of the three query prompts q, define m_q(X)=logit_X(b)−logit_X(a). Define C_r(m)=m_r−mean_{q≠r}(m_q). The native relational-response interaction is

`R = C_r(m(B)) − C_r(m(A))`.

It is a standard factorial/triple contrast. It cancels arbitrary source×candidate offsets shared across queries, arbitrary query×candidate offsets fixed across sources, and source×query scalar offsets shared across candidates. Neither gold answer constraints nor matching are used. A pure query-independent name copier givesR=0. A parser or complete future-answer lookup can giveR>0; they remain competitors to claims about semantic representations.

The exposed Llama audit foundR>0 for62/72 edges andR≥1 for56/72, after the166/288 hard-accuracy gate failed. These are discovery data, not confirmation. This Qwen test freezes the following new validation rule before its outcomes.

## Native graded-response prerequisite

At least90% of288 native responses must terminate with exactly one full leading-space name token, and at least80% must do so in every quartet×form stratum. At least90% of72 edges must have bothR>0 and the direct target-query change Δm_r>0. Each quartet×form×changed-role cell must have at least3/4 rotations withR≥1, Δm_r≥1, and all six endpoint responses properly formatted. These native-only eligible edges define the primary conditional population. Retain every rejected case and unnormalized effect.

Native correctness is not this gate. The gate is also not entirely bias-invariant: the direct target-response guard can fail under sufficiently strong source-dependent bias. It defines a stated reliable-response subset, not an estimate of all linguistic computation. If any prerequisite fails: **INCONCLUSIVE_RELATIONAL_RESPONSE**, stop before interventions, report native answers and allR values. No lower threshold, repaired wording or replacement case.

## Causal panel

Same P/Q split as the archived design: P through the final passage period, Q all later question/assistant positions through original readout. At each declared outgoing cut7/14/21, combine native residual banks into BA=(P_B,Q_A) and AB=(P_A,Q_B), both in the A input-token context. Every one of the72 edges now evaluates **all three questions at every cut**, both hybrids:1,296 next-token measurements. Generate all432 cut14 responses and72 prefix-only waited→stood sham responses fully. At7/21 do not invent full generations from first-token data. Original prompt slots only are replaced; newly generated positions retain A-context lower-layer computation.

Native AA/BB provide m-vectors; hybrids provide m_BA/m_AB. Define

`S_R = [C_r(m_BA) − C_r(m_AB)] / R`.

R≥1 is the fixed native denominator guard. No clipping. Pure source dominance gives+1, pure suffix dominance−1; nonlinear values outside this interval are allowed. This is causal sensitivity of a declared multi-query output contrast, not a fraction of semantics or a construction time for role r. Always retain the three separate m responses and actual generated answers so the joint contrast cannot hide countervailing changes.

All archived technical controls remain:72 architecture-forced0/28 endpoints and108 identity/full-joint donor controls, source-prefix future-query invariance, exact realized replacement hashes, unchanged generated/pad tails, original-prompt continuation invariance, unwrapped/wrapped repeat, pre-exposed worker references. Full-vocabulary tolerance≤.05; exact source/repeat hashes where declared. All pass before scientific edits. Endpoints are controls, not discoveries.

## Competing predictions and fixed cross-validation

Use the same six held-out quartet×form folds: train on the other quartet's other two forms; predict that held-out quartet/form at each of three cuts. Role/mention/generic predictors are training means grouped by queried role, event mention position or cut alone. The nuisance rival is per-cut ridge regression,lambda1, intercept unpenalized, training-only standardization, with native C_A,C_B, raw target m_A,m_B, four-candidate native entropy at each endpoint, actual person-token index, distance to source period, source-to-readout distance and mention-position one-hot. The entropies refer only to the declared four name-token distribution conditioned on each prompt; they are not interpretation probabilities. No parameter search or fitted representation alignment.

Role-indexed prediction expects better unseen-form/lexical/paraphrase accuracy than mention-position and generic schedules. Native-output readiness predicts the nuisance rival will match or outperform it. Source/question address incompatibility can make hybrids incoherent; actual native switches/invariant failures remain material evidence. A role predictor win does not rule out a syntax-aware copier or explain why a particular internal component carries the response.

## Sham validity and scientific decision

For each edge's same a/b contrast, form a three-query m-vector from the waited→stood prefix-sham at14. Let `R_sham=C_r(m_sham)−C_r(m_A)`. Among native-eligible edges, at least90% must have |R_sham|/R≤.25. At least90% of scientific cut14 and sham full generations must terminate with a complete source name. Native-gold switch counts, unchanged-role correctness, complete three-query hybrids and conditional hard-answer preservation are reported separately; they do not become a substitute success endpoint or proof that wrong answers have been fixed.

**ROLE_CONTRAST_ACCESS_PILOT** requires native and numerical gates; the sham/format gates; role pooled held-out MSE≤.75×the lowest pooled rival MSE; role MSE strictly below every rival in each of six held-out folds; within-cut S_R residual standard deviation≥.15. Otherwise: **INCONCLUSIVE_INTERVENTION** if validity fails; **NO_SUPPORT_FOR_ROLE_INDEXED_ACCESS** if validity passes but role prediction fails. An output/position/generic rival win is retained. No p-values, universal process, semantic subspace, model-family law or publishable novelty claim.

These are deliberately demanding descriptive pilot criteria on two lexical quartets. Shared questions, rotations and layers are dependent. All raw and native-excluded edges remain available. A positive conclusion is conditional on the explicit native-response gate and this intervention, not a property of all sentences or all models.

## Resources and stopping

Serial sequential-layer Qwen only; failed batch optimization retained and excluded. Each forward≤90s,32/worker,4650 total including references; native and causal stages≤90min each. Normal system pressure, disk>8GiB, swap<4GiB and growth<1GiB, MLX3GiB guideline/256MiB cache. No numerical tolerance changes or automatic retries. If native, technical, resource or validity gates fail, preserve and stop. No extra model or adaptive inference after this counterpart. Complete scripts, raw vocabulary/state payloads, controls, exclusions, sham effects, fits and methodological limits committed to the existing branch and linked Issue35.
