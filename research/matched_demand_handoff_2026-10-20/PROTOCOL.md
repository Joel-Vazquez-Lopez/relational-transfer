# Matched-demand source/suffix recombination pilot

20 October 2026. New, separately authorized reformulation after the failed participant-last-subtoken study. Previous failures and t03/t04 remain intact. No Qwen, paid API, training, download or model selection. AI-only authored and template-checked annotations.

## Question and scope

At a fixed intermediate block, does the causal balance between source states and subsequent question/assistant states differ between grammatical-role and mention-order questions, with the source, correct name, question length and answer location matched? This is a functional intervention comparison in one quantized Llama 3.2 3B checkpoint, not identification of a unique linguistic algorithm or a cross-model mechanism.

The generic source-to-question handoff and joint source/question patch are already precedented by [Language Models Use Lookbacks to Track Beliefs](https://arxiv.org/html/2505.14685v3). [SCIT](https://arxiv.org/html/2608.27265v1) also tests declared prefix/suffix causal carriers in latent reasoning caches. We do not claim that these intervention ideas or broad developmental stages are new. [Mixing Mechanisms](https://arxiv.org/html/2510.06182v2), [Dai et al.](https://aclanthology.org/2024.emnlp-main.967/) and [Feng and Steinhardt](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9d1b7fc578c0d2d6431fc26d736ecaf3-Abstract-Conference.html) are relevant binding precedents. The narrower matched-demand contrast is the question of this pilot, not an established novelty claim.

## Fixed inputs

Two fresh name triples (Alice/Bob/Carol; Dave/Erin/Frank), all six role permutations, active and goal-fronted forms, one verb `sent`. Six questions: three grammatical roles and three mention positions. Each question is exactly five Llama tokens; each name is one token both in its source mention and as a leading-space answer. All six questions have identical system instructions and answer positions within a form. This is 144 native prompts and 24 source cases. No human annotation or corpus generalization.

The three fixed transposition edges from assignment 012 are 012↔021, 012↔102 and 012↔210. Both forms and both name triples are used without selecting pairs from outcomes. Other permutations are included in the native screen to preserve the full six-assignment comparison. There is one question wording per demand/role; residual lexical differences between questions remain a competitor.

## Intervention

At cuts 0,7,14,21,28 (embedding output or outgoing block), partition the actual prompt into prefix P through the source period, and suffix Q comprising every later question/header/assistant position through the original answer-readout token. No source reordering, learned map, residual projection or token matching is used. Within a same-form pair, A/B token arrays differ only at complete, single-token person mentions and have identical lengths. The entire residual state of each selected position is replaced.

Native AA and BB are recorded. The two unique hybrids are BA=(P_B,Q_A) and AB=(P_A,Q_B), both computed in the A token context. Reversing recipient orientation would produce the same two original-prompt hybrid tensors: it is not an independent observation. Original-prompt continuation slots beyond the original answer boundary are never replaced. Padding is after the actual end, physical shape 1×128, no KV cache. Capture only five declared cuts.

The 432 scientific generations comprise 2 families×2 forms×3 edges×6 questions×2 hybrids×3 intermediate cuts. All are collected, including native failures. Exactly 96 endpoint controls (first edge, all six questions, both hybrids, cuts0/28) and 144 intermediate controls (first edge, all questions, three cuts, identity and complete joint swap) precede scientific hybrids. Joint swap must match the whole native donor vocabulary vector within absolute .05; identity and forced endpoints likewise. Wrapper/repeat controls, worker restart references, future-question source invariance and exact float32 vector hashes are required. Any technical/resource failure stops; no retry or change after results.

Cut0 source determines original-prompt logits and cut28 suffix determines them by architecture. These endpoints are controls, not empirical discoveries. The forced equivalences concern the original prompt's full next-token vector, not an unproved equivalence of future generated-token lower-layer states.

## Measurements and predictions fixed before inference

For an edge and a changed role r, match its role question to the ordinal question referring to that same surface position in that form. Thus both A and B expected names match across demands. Let m_XY be the raw next-token logit for the B name minus that for the A name, under native/hybrid XY. Single-token candidate names avoid word-fragment scores. Define source authority S=(m_BA−m_AB)/(m_BB−m_AA), only when the native denominator is at least1 logit. No clipping or alternative denominator. Source dominance gives +1 and suffix dominance −1 under an ideal additive selector. Values outside that range are possible nonlinear responses. This is an analyst-defined causal contrast, not a probability, information measure or quantity claimed to be computed by the model.

Primary contrast is Δ=S_role−S_ordinal at block14. Blocks7 and21 are contextual secondary profiles, not alternative primary endpoints. Each of 12 fixed edges contributes two changed-role contrasts, giving24 paired contrasts at a cut. Invariant questions remain diagnostic and are not counted as switches. Report individual next-token preferences, unconstrained full generations (maximum four next-token decisions, EOS required, whitespace strip only), complete three-question hybrid role/ordinal assignments and complete bidirectional edge units separately. Incomplete and unknown responses remain failures; no repair or exclusion.

H1 generic schedule: S changes across depth similarly across demands; Δ14 is small or inconsistent. H2 prolonged role dependence: positive Δ14 generalizes across both forms and both name triples. H3 compatibility/nonlinear dominance: hybrids retain one baseline assignment irrespective of which bank supplied it, or become incoherent; joint swap still succeeds. H3 can coexist with a demand effect. No unique syntax-aware parser, copying algorithm or output-preparation circuit is identified by this panel.

Technical pass is required. Scientific interpretability requires at least90% native full-generation correctness, at least95% native correct first-token answers, and at least20/24 eligible normalized contrasts at14, with at least4/6 per family×form stratum. Eligibility requires all four matched native role/ordinal answers to be complete and correct, and both denominators≥1. Failed baseline cases are still reported and intervened on; all unnormalized contrasts remain available. This admission boundary and its selection limitations are explicit.

A predeclared **DEMAND_DEPENDENT_PILOT** requires those gates, mean Δ14≥0.20 in each of four family×form strata, and positive Δ14 in at least80% of eligible contrasts. This is a deliberately demanding descriptive pilot criterion, not a population significance test; dependent items are not independent replicates. If competence/eligibility fails: **INCONCLUSIVE_COMPETENCE**. Otherwise, failing the criterion yields **NO_SUPPORT_FOR_PROLONGED_ROLE_DEPENDENCE**, even if a secondary curve looks interesting. No endpoint, hypothesis, score or site will be changed to rescue it. A negative result does not establish identical circuits.

## Resource and stop rules

At most2630 actual model forwards including restarts,32 per worker,90s/forward,45min each for native and intervention stages. Normal macOS memory pressure, >8GiB free disk, <4GiB swap and <1GiB growth from controller start; MLX active limit3GiB and cache256MiB. Stop on a failed gate, preserve partial records; no automatic rerun, extra depth, new wording, new model, significance chasing or adaptive experiment. Native failure does not trigger a prompt repair.

## Limits

Only two lexical triples, one verb, two templated constructions, one checkpoint and one wording per question. Quantization and format may affect results. Prefix/suffix mixtures are counterfactual states, possibly outside native support. Whole-state regions contain lexical, positional, syntactic and task signals together. Any matched-demand difference may include distributed answer preparation or question-specific lexical effects. A positive pilot would justify an independently designed multi-verb/paraphrase replication, not establish a new semantic representation or publishable mechanism. Complete all raw and methodological records before stopping.
