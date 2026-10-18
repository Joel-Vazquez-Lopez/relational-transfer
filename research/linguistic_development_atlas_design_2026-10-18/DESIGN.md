# Linguistic Development Atlas: an argument-boundary pilot

**18 October 2026 · DESIGN FOR REVIEW · no new model inference authorized or executed.**

## Recommendation

Study **when a model's response to the same name permutation becomes sensitive to argument grouping**. Start with a small coordination × name-permutation panel, measured throughout the sentence and question, from embeddings through all 28 blocks of local Qwen 7B and Llama 3B. This addresses the missing lexical-permutation control in the latest benchmark. It does not attempt a complete theory of language, identify a unique semantic circuit, or require cross-model paths to coincide.

The preferred initial observable is an interaction field over layers and token positions. It asks whether *the effect of an operation depends on the linguistic context in which it occurs*. A nonzero field is a useful constraint on additive accounts; its linguistic interpretation must survive stronger syntax/marker/output accounts. A second-order contrast is ordinary factorial analysis, not a new mathematical discovery.

This is a concrete proposal for joint review, not a preregistration of an approved run. Stimuli and resource estimates are executable; the model runner and final analysis lock remain to be implemented after design review and before any model outputs. No previous failed gate has been relaxed.

## 1. What the saved record can already answer

### Preserved observations

| Record | What is supported | Boundary that remains |
|---|---|---|
| `7cfb406`, query panel | Qwen 7B: Agent 24/32, first-name 32/32; late query difference +1.010, positive in all 8 families; 3/8 fully competent | Output scalar predicts 16/16 final geometric signs; copying counterexample survives. This is task-conditioned response geometry. |
| Same panel, Llama | Agent 27/32, first-name 32/32; query difference +0.868 | Historical pre-question discrepancy .003827 exceeds .001. Failed gate stays failed; no clean replication. |
| Component reuse pilot | Saved attention/MLP/outgoing writes at six anchors, 12 development sentences; reconstruction passed on Qwen | 161/192 answer checks passed; competence stopped the study, **zero patches**. No reuse result. |
| Prefix-shape audit | Some identical causal prefixes differ when total input crosses the 64/65 quantized-kernel boundary; fixed shape eliminates the tested discrepancy | Numerical implementation effect, not evidence of future-question information or a linguistic family difference. |
| Latest specificity benchmark, `f59a798` | Fixed-shape long-lag role/role vs role/unrelated Gram gaps +.0961 Qwen, +.1524 Llama; leave-one-family-out gaps all positive | Role swap preserves vocabulary whereas unrelated contrasts do not. Lexical permutation remains a live competitor. Cross-model operation matching was weak. |

These are historical results, not rerun findings. All five state archives inventoried in `SAVED_DATA_AUDIT.json` are local; the two original query-state byte hashes match their recorded summaries. Headers document widths, stages, sites and precision. Full tensor archives are not all uploaded: the latest remote benchmark preserves descriptors and raw text ledgers, with full states local. Float16 query archives cannot support new claims about tiny numerical effects. The old component archive has no clean Llama counterpart.

### New saved-data/design audit in this delivery

1. **Exact alias certificate.** Let R=+1 mean A is Agent, V=+1 mean active voice, and Q=+1 mean the Agent question. In the old two-person design:

   `first-person identity O = R V`

   `expected answer identity Y = (R + RQ + RV − RVQ) / 2`.

   The eight factorial columns have rank 8; adding O and Y leaves rank 8. Thus the role×voice interaction is also the first-identity variable, and combinations of task interactions reproduce the answer code exactly. Repeating this design with more names does not remove that alias. This is an algebraic fact about the stimuli, not evidence that a particular model uses a particular code.

2. **Pair coverage is available but insufficient.** Each model's saved eight-family panel contains 48 within-family pairs with the same expected answer and different Agent identity. Of these, 16 keep the question unchanged: they use the first-person question and change voice. Both answers pass the original scorer in 40/48 Qwen pairs and 43/48 Llama pairs. These are overlapping pairs, not independent observations. They do not simultaneously match question, voice, first position, and relation. `OLD_PAIR_COVERAGE.csv` lists every pair.

3. **Questions answerable without new inference:** whether saved contrasts align with expected/actual answers; sensitivity to incorrect families; which positions are causally eligible for each difference; native versus fixed-shape results on the latest fixtures; component cancellation on the exposed Qwen development captures. These can characterize observations but cannot fill missing factorial cells. The exploratory laboratory owns broad curve/motif exploration; the present contribution is identifiability, provenance and a discriminating design.

4. **Questions not answerable from these archives:** a vocabulary-matched, role-preserving permutation baseline; independent role/voice/position effects in the old two-person design; question-conditioned attention/MLP development across both models; family-level causal equivalence. Those require new data. No replot or fitted alignment can manufacture them.

## 2. The smallest preferred new linguistic contrast

Four sentence cells for one family:

| Argument frame K | P=original | P=swap Boris/Clara |
|---|---|---|
| Within one role | Ada recommended **Boris and Clara** to Neil. | Ada recommended **Clara and Boris** to Neil. |
| Across two roles | Ada recommended **Boris to Clara** and Neil. | Ada recommended **Clara to Boris** and Neil. |

Intended grouping in the second row is `Ada recommended [Boris] [to Clara and Neil]`, with the analogous swap. Each sentence has the same word multiset. K exchanges only *and* and *to*. P exchanges the same names in the same two ordinal name slots. The actual tokenized prompts preserve their positions in **both** existing tokenizers; changing K changes exactly two token IDs. All names that move occupy one space-prefixed token each. These are tokenizer observations only.

Within-frame P preserves Theme membership {Boris, Clara}; across-frame P moves Boris and Clara between Theme and Goal. Agent Ada and final person Neil remain fixed. The distinction concerns participant-role membership in ordinary affirmative readings. It does not assert that coordination order is pragmatically meaningless or that event structure is identical in every semantic theory.

### Source annotations and uncertainty

These are **constructed English sentences**, not a corpus sample or an independently validated gold dataset. Sixteen unique sentence strings have explicit Agent, Theme and Goal-member annotations in `DRAFT_STIMULI.jsonl` (64 sentence/question rows). Two verbs, *recommend* and *refer*, share this surface frame but may differ in event interpretation. Goal here means the recipient/target argument of the *to* phrase, not a universal claim about the thematic-role inventory.

Coordination scope, distributive readings, recommendation/referral pragmatics and the naturalness of the questions need review. Before freezing a run, require two independent annotation passes, at least one by a human comfortable with English syntax; record disagreements. Reject ambiguous items before viewing model outputs. Do not repair/rephrase them after results. This review is **not completed**. The table's bolding is explanatory only; the actual prompts have no markup. If reviewers cannot agree that the relevant membership questions have unambiguous answers, this proposal stops at design.

### Four question demands, held fixed across K and P within each family

| Demand | Example | Within P0/P1 | Across P0/P1 | Purpose |
|---|---|---|---|---|
| Fixed Agent | Did Ada recommend someone? | Yes/Yes | Yes/Yes | Different full relation configuration, same query and answer |
| Theme membership | Was Boris recommended to someone? | Yes/Yes | Yes/No | Whether the moved entity is the recommended person |
| Goal membership | Was someone recommended to Boris? | No/No | No/Yes | Complementary requested relation |
| Mention order | In the sentence, was Boris mentioned before Clara? | Yes/No | Yes/No | Same answer-switch pattern in both frames; within-frame roles stay fixed |

For two families the fixed-Agent question instead targets the final person and is constantly **No**. Development and held-out partitions each contain one constant-Yes and one constant-No family, and 16 Yes/16 No gold answers. The complete panel has 32 Yes/32 No. This avoids a constant-Yes-only control and the earlier arbitrary X/Y decoding problem. It does not prove that binary verification is as demanding as open participant retrieval.

The system instruction explicitly asks about roles assigned by the provided sentence and observable mention order. A No denotes absence of the queried assignment from this sentence; it does not assert that the person never performed that action elsewhere. This is a closed-sentence extraction/verification task, not a test of logical negation or real-world falsity.

**Scope:** this core tests grouping × name permutation × question demand, with entity renaming across families. Voice is held active. Do not describe it as independently manipulating every linguistic factor in the programme. See Section 8 for a voice/position extension.

## 3. Measure the interaction throughout development

For each model, family, question, layer l and aligned site t, retain native hidden vectors h and define:

`δwithin = h(within, swap) − h(within, original)`

`δacross = h(across, swap) − h(across, original)`

`I(l,t,q) = δacross − δwithin`.

Equivalently, I is the difference between the two frame responses under P. These are algebraically dependent descriptions, not independent replications. The atlas retains the labeled main responses, their magnitudes/directions, I, and their layerwise changes. Do not collapse the record to one similarity score.

### An exact negative control supplied by the language

Before the first *and/to* cue, the two K conditions have **identical causal token prefixes**, separately for P0 and P1. Therefore δwithin=δacross and I=0 at every block at those positions in an exact causal implementation. The future question also cannot change sentence-prefix states. These are technical causal-mask nulls, not claims about when the model begins to understand a relation. Prefix differences between P0/P1 themselves are expected at moved names; it is the **interaction** that must be zero.

After the cue, contextual computation can create I at unchanged later tokens and at the question. Report all positions, including the cue itself; do not call a difference at different *and/to* tokens proof of semantics. At a fixed unchanged token with unchanged embedding, a nonzero residual interaction must arise contextually. Contextual syntax/position computation remains sufficient.

### Primary record and descriptive summaries

- Embedding and all 28 outgoing residual states at every sentence token, question token and original prompt-end token. Keep token IDs, offsets, true indices and word spans. Retain all subtokens; last-subtoken summaries are secondary.
- Preserve `||δwithin||`, `||δacross||`, `||I||` and their full vectors. Also show `||I||/(||δwithin||+||δacross||)` with the denominator. A zero denominator is undefined, not a perfect match.
- Full layer × token maps per family; line plots at pre-cue, post-cue, sentence-period and prompt-end anchors. Embeddings are stage 0. Keep both physical block number and relative depth. Native readout normalization/logits are a separate final operation, not an extra linguistic layer.
- Save within-model Gram relationships among the labeled response vectors to retain direction relationships. Compare these coordinate-free summaries across models only after inspecting raw values. Orthogonal invariance is useful; it does not establish invariance under arbitrary invertible reparameterization or identify shared algorithms.
- Pre-question maps must agree across all four demands. At the final prompt-end, compare I under Agent and Order (output-label-controlled) with Theme/Goal (label changes differ across K). Query wording and end positions differ across demands, so a cross-question difference alone does not isolate an abstract task variable.
- Do not fit time warping, a projection, a layer window or a new similarity measure on held-out outcomes. No dichotomous “semantic onset” will be selected from peaks. Show the complete development and the exact causally ineligible region.

### Within-layer contributions

Capture the actual native residual attention and MLP **writes**, preserving sequential attention→MLP order. For each block, check `out ≈ in + attention_write + mlp_write`. Compute the same factorial I separately on writes; their vector sum accounts for the change in the residual interaction, up to measured rounding error. Positive magnitudes can cancel: never add norms as though they were signed contributions.

Reuse the existing wrapper pattern after verifying it against each model's implementation. Observed write accounting is not a causal attribution to a head, nor evidence that attention weights describe information flow. A causal patch stage is **not** part of this request. If component reconstruction fails, retain valid residual observations and mark component attribution unavailable; do not retrofit a different decomposition after held-out outcomes.

## 4. Competing accounts and prospective discriminators

| Account, explicitly restricted | Predicted observation | What would challenge it | What a success still would not exclude |
|---|---|---|---|
| Context-independent lexical/position permutation with additive frame and question effects | Same P-response in both frames; I=0 | Reproducible post-cue I above measured technical null, at unchanged tokens | Nonlinear positional/marker computation; norm/normalization effects |
| Fixed writer of the **answer label alone**, possibly distributed across many dimensions | I=0 for Agent and Order; Theme/Goal I can differ | Valid I in Agent/Order or pre-question I | Confidence-dependent output preparation; unqueried-answer preparation; richer contextual writers |
| Generic, input-independent depth schedule alone | Shared depth rhythm without condition-specific content discrimination | Label-specific grouping interaction replicated across new names beyond shared rhythm | A generic clock modulating input-dependent features; observing I does not remove schedule effects |
| Constituent-sensitive composition or constituent-gated retrieval | Can produce I only after a distinguishing cue; may preserve it at sentence end and reweight it under questions | No robust I, only answer-switch-aligned final effects, or failure on new names | Sensitivity can be nonlinear local syntax, not a semantic abstraction |
| Strong syntax-aware/compound-key copying | Reads *and/to*, selects positional name slots, routes requested membership/order | This pilot's behavior alone has **no discriminator**; broader transfer or a separately discriminating causal design is required | Remains a serious survivor even after a positive pilot |

`PREDICTION_WITNESSES.json` makes the restricted output account explicit. Encode Yes as +1 and No as −1 and let the state be that scalar times any fixed distributed vector. Its I is **0 Agent, −2 Theme, +2 Goal, 0 Order**. Thus Theme/Goal interactions alone are uninformative against this rival. A literal surface-string program implemented in `design_audit.py` answers **64/64 draft rows** correctly using marker/position keys. This is a synthetic rival result, **not an LLM result**.

Same answer does not mean same logits/confidence/internal output code. Save full-vocabulary logits, actual argmax tokens, Yes/No probabilities and margins, and compare those patterns descriptively with I. Removing one readout direction does not eliminate distributed output preparation. Pre-question I eliminates a writer operating only after the displayed question, but a system preparing possible answers in advance can still explain it.

**Critical limitation:** K changes argument cardinality, cue position and local syntax together. It is a controlled linguistic package, not an isolated semantic switch. A lexical *and/to* gate can reproduce conditional sensitivity. No positive result here licenses “unique event selector,” “universal semantic path,” “same circuit across families,” or “novel binding mechanism.”

Nonzero I alone is a weak finding: ordinary nonlinear transformations and normalization can create interactions even without useful linguistic composition. The substantive object is its reproducible **distribution over eligible tokens, layers, demands and component writes**, interpreted alongside output behavior and the lexical-marker rival. This pilot is a measurement/linguistic feasibility test, not a decisive semantic-versus-copying experiment. If the review requires that stronger distinction immediately, **do not run this pilot as its substitute**; no validated discriminator for the strong copier is supplied here.

## 5. Pilot size, technical gates and execution order

### Proposed budget, after review only

- Existing local quantized Qwen2.5-7B-Instruct and Llama3.2-3B-Instruct, sequential loading, existing MLX environment. No training, paid calls, downloads or additional model families.
- Four lexical families = two verbs × two disjoint name sets. One family per verb for development, the other for held-out names. Verbs are shared across splits: this is **not** verb generalization.
- 4 families × 2 frames × 2 permutations × 4 demands = **64 unique prompts/model, 128 main forward passes**. Two extra passes/model for wrapped/unwrapped and repeat checks: **132 maximum forwards**, including validation. No generated rollouts, patches or adaptive extra items.
- Local tokenizer audit: prompt lengths **69–74 Qwen, 91–96 Llama** under the explicit role-verification system instruction. Every run has batch 1 and total length 128, with dummy tokens strictly after the original prompt end and causal masking. Logits are read at the **original prompt end**, not at the dummy suffix. These are conditional next-token predictions, not full generated completions. Same-shape suffix invariance is checked across existing main prompts; gate failures stop interpretation.
- Existing Yes/No single-token aliases (case and leading-space variants) are listed in `TOKENIZER_AUDIT.json`. Compute logsumexp over deduplicated aliases per class; gold margin must be positive. Also require the full-vocabulary argmax to be a gold-class alias for the primary competence check. Report every other argmax as an unexpected answer, including punctuation or refusals. No forced-choice-only score will be described as unconstrained correctness.
- Float32 outgoing states plus attention/MLP writes at all retained sites: upper bounds **1,228,308,480 bytes Qwen**, **1,052,835,840 bytes Llama**, about 2.12 GiB combined before compression, plus logits/metadata. Stream one prompt at a time. Runtime is not benchmarked for this new instrumentation; no wall-time promise.

The workspace filesystem had approximately 8.9 GiB available at design time. Recheck space before approval/execution and allow temporary-file overhead; no cleanup of existing research artifacts is proposed.

### Before any held-out result is viewed

1. Jointly review this proposal and linguistic annotations; resolve source uncertainty. Record approval and final stimuli separately from these drafts.
2. Implement the runner and exact plots/analysis against the locked design; archive code hashes, installed versions, model/config/tokenizer/chat-template hashes and chosen dummy token ID. Validate token mappings and all stopping logic without model execution first. Use the existing full source manifest as a starting point, recheck rather than assume weights unchanged.
3. On development prompts, use the two extra forwards/model: one unwrapped evaluation of the first development cell and one wrapped exact repeat; its main wrapped capture supplies the comparator. Require max absolute logit deviation ≤0.05 nat and relative retained-state repeat discrepancy ≤1e−6. All main prompts have the same total shape. The suffix of other questions supplies content controls without extra runs.
4. On all development captures require finite arrays; pre-cue I and same-sentence pre-question state discrepancy ≤1e−6 relative to the largest relevant state norm; native block reconstruction max relative error ≤.002 (`||out−in−attn−mlp|| / max(||out||,1e−12)`). If prefix invariance fails, **stop before held-out inference**. Do not increase tolerances after the fact. Reconstruction failure stops component claims; residual observations can continue only if their own checks pass.
5. Save every development answer and map. Each family has 16 answer checks. Report family-level all-correct status and individual accuracy separately. If neither development family is fully competent for a model, **do not run its held-out panel**; report feasibility failure and all exposed data. This gate limits mechanistic interpretation of misunderstood questions and costs. Do not replace families, reword questions or change readout rules within this pilot.
6. Freeze the implementation, primary maps, controls, eligibility decisions and written predictions before releasing held-out outputs. No predictor/window fitting is proposed for this tiny pilot. Run eligible held-out panels once, preserving all mistakes. No retrospective “competent only” main result. If only one model qualifies, report a model-specific observation, not a replication.

The proposed two-validation-pass bound cannot validate a newly altered runtime exhaustively. Unexpected failures end this pilot and become a separate report, not permission for additional unbounded debugging inference.

## 6. Decisions and honest interpretation

This feasibility pilot is deliberately small: four item families, only two held out. Layers, tokens, correlated contrasts and answer rows are not independent replications. No population p-value, family-level conclusion or paper claim follows from this sample. Preserve all individual curves and failure ledgers.

| Outcome | Decision for this pilot |
|---|---|
| Technical null fails | Invalid measurement for the intended comparison; stop the affected model before held out. |
| Development competence fails | Feasibility failure; preserve exposed maps, no silent semantics interpretation or adaptive repair. |
| Effects restricted to Theme/Goal final readout and track label/margin changes | Output-compatible observation; no evidence beyond output preparation. |
| I appears at valid post-cue sentence positions and/or under matched-label Agent/Order demand, with comparable motifs on both held-out name families | Support for context/argument-frame-sensitive permutation responses over the **restricted additive** and **fixed-label-writer** accounts. Syntax/marker/copying rivals remain. Describe models separately. |
| One model differs or effects vary with verb/name | Valuable model/item difference, not a failed universal mechanism or evidence of family specificity. |
| I stays negligible | Challenges this operational hypothesis under these conditions; does not prove an absence of relational representations elsewhere. |

A positive map does not yet tell us whether a feature is constructed, carried forward, reread or used for the answer. Attention/MLP write accounting distinguishes where the observed interaction changes; causal use requires a later, separately reviewed intervention with rival-specific behavioral predictions. There is **no immediate causal experiment in the proposed budget**.

## 7. Cross-model comparison without assuming shared hidden spaces

Compare the *labeled experimental relationships*: does the interaction respect causal cue availability, persist at unchanged later sites, cancel within blocks, and change with demand? Report absolute norms, normalized magnitudes, direction relationships and blockwise changes separately. Qwen and Llama can differ in each without invalidating the programme.

Both selected models have 28 blocks, which simplifies display but does not establish homologous layers. Start with parallel raw maps and an event-relative token axis (before cue, cue, later arguments, period, question); no learned time alignment. A later replicated set could test whether a low-complexity mapping predicts held-out motifs better than a depth-only baseline. Such fitting is underpowered here and is not a primary result. One model per family confounds family, width, training and instruction tuning; **family dependence cannot be established**.

The wider atlas may need a collection of magnitudes, operator relationships, token-availability masks, interactions and readout relations rather than one scalar distance. This proposal tests whether there is content worth comparing before inventing a more flexible metric.

## 8. Alternative directions to preserve

### A. Voice × argument order × role assignment

Use a three-person frame with a fixed Agent and independently exchange Theme/Goal names. Cross active/passive with Goal-PP fronting:

1. Ada referred Boris to Clara.
2. To Clara, Ada referred Boris.
3. Boris was referred to Clara by Ada.
4. To Clara, Boris was referred by Ada.

Then exchange Boris/Clara in each. This supplies role-preserving reorderings and same-Agent-answer/different-relations controls. It is broader and can test voice/fronting systematically, but PP fronting is marked, voice changes token inventory, token positions shift, and competence has already been fragile for other constructions. Human naturalness review and a separate resource/design approval are needed. A two-person voice×order design is not automatically independent: surface first identity can still equal a role×voice interaction. Calculate the design rank before running it.

### B. Cue-arrival and delayed disambiguation

Construct prefixes with the same early evidence and later disambiguating attachment/voice cues. A causal Transformer imposes exact information-availability boundaries. The question is where later eligible positions begin to differ and how earlier unresolved alternatives persist in later representations. Earlier token states do not get revised by later text in a single forward pass. Distinguish processing across token positions from depth development and from training time. The present coordination design is a first bounded version.

### C. Native causal use, only after a discriminator exists

Ask whether a validated interaction can predict a native attention/MLP intervention's **differential** effect on Theme/Goal versus matched-label controls, with identity, neighbor, sham, cross-document and output-source controls. Existing failed event-period/individual-write gates remain failed. Do not pick a new causal site simply because a plot peaks there. If a structure-keyed copier and the intended account predict the same intervention outcomes, do not run it as a mechanism discriminator.

### D. Richer mathematical objects, only when warranted

Labeled response Gram tensors, interaction fields, distances among operator responses and component cancellation are candidates, with explicit invariances and known counterexamples. Closed loops of raw state differences telescope to zero when transformations return to the identical input; calling that “holonomy” does not create evidence of path-dependent model computation. Model known toy systems first. Learned dynamics comparisons are possible later, but transformer blocks have different parameters: depth is not automatically the time of an autonomous recurrent dynamical system.

## 9. Prior research and novelty boundary

The comparisons below use primary paper summaries as a scoped literature check, not an exhaustive novelty audit.

- [Dai, Heinzerling & Inui (2024), *Representational Analysis of Binding in Language Models*](https://aclanthology.org/2024.emnlp-main.967/): low-rank binding-ID information and activation edits that redirect attribute retrieval are already prior work. Our earlier answer redirection cannot itself establish novelty over that result. This proposed pilot studies argument-grouping-conditioned developmental responses; it does not demonstrate a new binding code or improve Dai's causal mechanism evidence.
- [Hewitt & Manning (2019), *A Structural Probe for Finding Syntax in Word Representations*](https://aclanthology.org/N19-1419/): syntax-related geometry in hidden representations is established. Finding a geometric grouping distinction alone is not a new foundational claim.
- [Tenney, Das & Pavlick (2019), *BERT Rediscovers the Classical NLP Pipeline*](https://aclanthology.org/P19-1452/): layerwise linguistic organization and changes in the prominence of linguistic information have substantial precedent. A decoder's controlled factorial interaction map asks a more specific question; it must earn its contribution empirically.
- [Denning et al. (2025), *Do Large Language Models know who did what to whom?*](https://arxiv.org/abs/2504.16884): overall sentence geometry and particular attention heads can differ in their thematic-role sensitivity. This motivates retaining multiple sites/components rather than treating a residual cosine as the sole test. Their attention findings do not prove causal use in our models.
- [Ostrow et al. (2023), *Beyond Geometry… Dynamical Similarity Analysis*](https://arxiv.org/abs/2306.10168): comparing computation beyond static geometry is already an active research direction. It does not automatically justify fitting recurrent dynamics across heterogeneous Transformer layers.

A plausible future contribution would be a reproducible, linguistically reviewed method and replicated result about **where and how context changes the developmental effect of an otherwise matched operation**, with strong lexical/output competitors. This pilot can establish feasibility and expose failures. No “first ever,” unknown mechanism, or paper-level discovery is claimed.

## 10. Coordination, reproducibility and next review

Issue [#35](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/35) remains Joel/Armaggedon's exploratory laboratory: broad linguistic theory, multi-site observation and mathematical ideas. Its separate branch is untouched. Codex's existing research branch holds this identifiability audit, proposed stimuli, prediction witnesses and the constrained future measurement plan. Share pointers and unresolved questions on #35; avoid duplicating its planned motif exploration.

Reproduce the design-only audit from the workspace root:

```sh
research/stronger_model_development_2026-10-15/.venv/bin/python research/linguistic_development_atlas_design_2026-10-18/design_audit.py --tokenize
```

The script reads local saved artifacts, computes the design certificate and pair coverage, generates draft annotations/prompts, and tokenizes with local files only. It never imports MLX, loads model weights, calls a paid API or executes a forward pass. `SAVED_DATA_AUDIT.json` identifies exact source bytes; `MANIFEST.json` hashes this delivery. `DRAFT_TOKENIZATION.jsonl` preserves actual chat strings and token/site mappings. `PREDICTION_WITNESSES.json` is explicitly synthetic. No original result files were edited.

**Review requested:** prefer the grouping/permutation pilot or the broader voice/fronting direction; settle the annotation ambiguities and whether this limited target is worth 132 bounded forwards. After that review, freeze a prospective executable protocol before inference. This delivery stops at scientific design as requested.
