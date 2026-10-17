# MIRA-006 — Irrelevant-name interference: three rival causal stories

**17 October 2026 | DESIGN ONLY | No new model calls, no human-gold validation.** Complements MIRA-004 and coordinator #33; does not supersede the original >=8-error replication failure or the research pause through Oct 22.

## Observation motivating the design
Coordinator #33 reports another investigator's corrected Qwen7B development panel: 32/32 isolated, 64/64 bare two-event, 57/64 with irrelevant named bystanders. All seven errors were wrong-event participant copies on the second target, not bystander copies. Raw files not independently replayed; preregistered >=8 error threshold **FAILED**. This is a descriptive pattern, not an established mechanism.

## Three explicit competing models

**P: Positional accessibility.** Added names/tokens change distance, ordinal indexing or source-recency competition. Prediction: errors track target ordinal, token offsets, insertion location and distractor proximity, including with semantically inert tokens of comparable length. A second-event asymmetry should persist when event-role overlap is absent.

**E: Event-address competition.** Distractor perturbs selection among event-indexed participant sets. Prediction: with token length and event/source positions matched, wrong-event borrowing should rise with event-type/role/participant overlap; errors should concentrate on the other event's *same role* filler rather than an unrelated name or the other role. Reversing event order may reverse the borrowed source while preserving overlap dependence. This remains a *functional* explanation compatible with syntax-aware copying; it does not identify a neural event address.

**Q: Query-conditioned reconstruction.** The query determines which local extraction computation is performed; distractor changes the chosen extraction strategy rather than an event key. Prediction: under fixed passage, question paraphrase, explicit target pointer, and answer-field order can change errors with stable source events. If event pointer removes errors but a syntactically equivalent paraphrase reinstates them, query-side routing remains viable. Native-state mediation not established.

## Controlled interventions: identify contrasts before new inference

1. **Insertion location:** same bystander list at passage beginning, between events, and after both events; match source token counts *at target-relative positions where possible*. If impossible, estimate distance as an explicit covariate rather than claim perfect matching.
2. **Entity vs length:** replace named bystanders with matched-token non-name nouns, neutral clauses and name-only lists; verify capitalization/tokenization and discourse coherence. Pure token count is not equal to attentional salience.
3. **Overlap:** two events same lemma and role signature vs distinct lemma/roles, with disjoint fillers; match question, target, length, target ordinal and local syntax. Do not assume natural source examples satisfy all factors.
4. **Order:** swap full event clauses while preserving referents, tense and discourse truth conditions; annotate any change of pragmatic salience.
5. **Query:** direct original human-written QA, neutral paraphrase, calibrated event pointer, and joint answer; pre-register which comparisons are comparable. A pointer is an interface intervention, not evidence of pre-existing latent addressing.
6. **Provenance of errors:** score OTHER_EVENT_SAME_ROLE, OTHER_EVENT_OTHER_ROLE, BYSTANDER, SAME_EVENT_WRONG_ROLE, UNRELATED, INVALID, AMBIGUOUS. Same answer string shared across events => AMBIGUOUS without independent occurrence evidence.

## Minimal prediction table
| Contrast | P | E | Q |
|---|---|---|---|
| Names vs token-matched nonnames | Similar if offsets dominate; name salience may differ | Names can raise event competition only if integrated into event parsing | Query strategy may be name-sensitive |
| Overlap HIGH vs LOW, matched position | Small after strong position controls | Larger wrong-event same-role borrowing | Could vary via syntax/lexical parser |
| Swap event order | Errors follow ordinal/distance | Borrowing follows competing event roles; may interact with order | Query-dependent |
| Change question only, passage fixed | Little beyond target cue | Stable if same event address resolved | Large, systematic sensitivity |
| Calibrated oracle event pointer | May help | Should help if event selection bottleneck | May help by simplifying parsing too |

**Crucial non-identifiability:** these predictions are not exclusive for unrestricted learned models. A competent mixed syntax+position+lexical copier can emulate P/E/Q on finite data. Require explicit restricted rivals, heldout preregistered contrasts, and if claiming native mechanisms, isolated path-specific interventions. Behavioral data alone cannot prove event-index storage.

## Design acceptance criteria
- Independent human review of source rights, event identity, gold roles and coreference; preserve all rejected candidates.
- Freeze exact raw prompt serialization, tokenizer lengths, model versions, decoding and scorer.
- Competence gate for both target events individually and jointly *in the same context*, not only after-the-fact isolated tests.
- Fit rival predictions only on development documents; evaluate on source-disjoint heldout documents and >=2 model families.
- Cluster uncertainty by document/event family, and keep effect sizes, exact counts and negative outcomes.
- Primary error provenance endpoint predeclared; don't re-use the 64 exposed Qwen pairs for confirmatory replication.
- Stop if no valid natural same-document multi-event human gold, or if joint competent shortcut explains heldout provenance.

## Concrete next work permitted during pause
Construct a **zero-inference counterfactual matrix validator** for event-order and distractor-placement permutations, including explicit impossibility markers for unmatched token lengths and semantic changes. Seek independent critique of this design, not an experiment approval. Coordinator #31 remains binding.
