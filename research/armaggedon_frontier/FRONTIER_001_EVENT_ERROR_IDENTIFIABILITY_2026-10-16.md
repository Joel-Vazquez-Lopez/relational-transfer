# FRONTIER-001 — Event-coherent errors versus fieldwise retrieval: discriminability before inference

**Scientific coordinator:** Armaggedon | **Written:** 2026-10-16 | **Scientific status:** ACTIVE HYPOTHESIS / READ-ONLY DEVELOPMENT ONLY | **Experiment status:** NOT APPROVED / NOT RUN | **Research lead:** Joel

> **Resumption principle:** We are investigating whether this is worth making the programme's next scientific frontier, **not** asserting a mechanism, an ACL novelty claim, or authorization for a new model run. The original relational-binding thesis is disposable if a better question emerges.

## Zero: the conclusion we can already prove analytically

Let `B_r=1` when a response to role `r` is observably the unique gold filler for **the other annotated event**, and `B_r=0` for the target event's unique filler. The roles must be genuinely annotated for both events and distinguishable; otherwise `B_r` is unobservable.

**Restricted shared event selector:** draw `S~Bernoulli(p)`; choose one event and emit both roles from it: `(B_1,B_2)=(S,S)`. Then `P(B_1=B_2=1)=p`, `P(00)=1-p`, off-diagonals zero, `Cov(B_1,B_2)=p(1-p)`.

**Competent shared-record lexical/position/salience copier:** draw `C~Bernoulli(p)` from a shared cue and copy both role fillers from the source record it selects: `(B_1,B_2)=(C,C)`. Its **entire behavioral distribution is identical** to the first model on such instances. A learner that conditionally reads the target query and reconstructs event roles can likewise be functionally equivalent. No number of ordinary behavioral examples uniquely separates unrestricted implementations that compute the same function.

**Conditionally independent role readers:** `B_1 ⟂ B_2 | Z` does *not* imply marginal independence if difficulty/source salience `Z` varies across worlds. By total covariance,
`Cov(B_1,B_2)=E[Cov(B_1,B_2|Z)]+Cov(E[B_1|Z],E[B_2|Z])`.
The first term can be zero while the second is positive. Thus a pooled excess of coherent borrowing over the product of pooled error rates can arise with **no shared latent event selector at the per-item readout stage**.

**Output serialization:** Joint autoregressive generation can create coherent field errors by prefix copying, forced complement, or formatting. A source-sensitive, query-conditioned record copier is not necessarily “nonrelational”; do not draw an arbitrary semantic/nonsemantic line between functionally identical algorithms.

**STOP condition zero:** a hypothesis distinguished *only* by coherent wrong-event pairs, marginal covariance, answer flips or a positive intervention interaction is **NOT IDENTIFIABLE** as a native event-binding mechanism. Do not run new inference on that justification.

## 1. Exact science question and scope

**Question:** In *authentic human language* with at least two gold-annotated co-occurring same-type events, when a pretrained LM is wrong on participant roles, does a restricted **common event-choice error process** add held-out predictive information beyond a competent **joint lexical/syntax/position/source-salience copier**, rolewise retrieval with shared observed covariates, and output-generation coupling? If yes, is there a second, independently specified native causal prediction that can distinguish a subset of candidate implementations?

This is a **provisional behavioral predictive comparison**, not the discovery of a unique semantic binding variable.

**Scientific distinction from existing work:** We would measure calibrated joint **error provenance and source-event identity** conditional on event annotation and compare predictive rivals on source-disjoint documents, rather than only optimizing aggregate event argument F1, showing multi-event context helps, improving consistency, or reporting positive patch sensitivity. **That distinction might still already exist in full papers; novelty remains unverified.**

Keep Nova's original question→all human-accepted-span QA-SRL **P0** separate. QA-SRL original questions do not automatically annotate two complete same-type events, two roles each, or gold event identity. A dual-question QA-SRL hit is not eligible by default.

## 2. Candidate rival policies; their limits must be explicit

| Code | Operational rival | Distinctive *restricted* prediction | Insufficient evidence |
| --- | --- | --- | --- |
| `E` | shared target-event chooser then two role-conditioned lookups | one shared mistaken event can produce a full coherent distractor tuple under independent role requests | coherence alone, because `C` matches it |
| `C` | *strong* query-aware syntax+lexical+position+coreference/record copier, optionally with a common salience factor | may produce the **same** correct and wrong tuples as E; frozen restricted cue variants can fail on cue-preserving target changes | calling it “a shortcut” without matching competence |
| `F` | independent role-specific selectors, conditioned on shared observed source features `Z` | residual pair dependence **conditional on Z and source-world clusters**, not pooled marginal covariance, differentiates *restricted* F | joint-error excess over product of **pooled** errors |
| `T` | generation-prefix/format templater on top of a question-conditioned answer policy | extra pair coupling under joint-field serialization and response-order changes, after direct-question competence matching | AP/PA output formatting collapse |
| `O` | oracle original human event/role schema and competent local parser | upper bound/diagnostic of gold observability and task competence | oracle-provided role labels as model inputs or inferred LLM mechanism |

Rivals `E` and an unrestricted `C` may remain observationally equivalent. In that case report **an equivalence class**, never claim that an internal semantic selector was discovered.

## 3. Candidate *witness* families: differences are against restricted rivals only

| Witness / authorized future stimulus condition | Can falsify | Cannot distinguish / major confound |
| --- | --- | --- |
| `W1` source document fixed, **target annotated trigger switched**, same question role and output contract; measure which gold event's fillers follow the query | *Query-blind* global first/last/salience copier that never reads target trigger | a competent query-aware grammar/compound-key copier can behave exactly like E |
| `W2` target trigger and target gold fixed, competitor's lexical repetition, proximity or presentation prominence altered **without changing original target event truth conditions**, independently human-validated | a *frozen, explicitly cue-limited* copier if its predictions must track the changed cue | grammar/coreference shifts, annotation drift, and a more flexible salience model |
| `W3` crossing two roles for the same target/competitor events with distinct accepted fillers; compare errors **conditional on measured case covariates/source clusters** | restricted factorized F under actual conditional-independence hypothesis | shared-Z, selection bias, joint syntax reader, and E versus C |
| `W4` same eligible event-role gold under separate role calls, both joint field orders and a budget-/format-calibrated interface | pure joint-prefix templater if coherence appears **only** when the first output field is already present | source-sensitive salience and autoregressive dependence still cannot identify a native cache |
| `W5` if W1–W4 survives and native model is competent: matched on-manifold source-event and participant-swapped **activation/path** interventions with site/architecture/semantic-gold matched shams | specified native route dependence and *some* restricted internal alternatives | generic copier executing same function, off-manifold donor artifacts, novel binding variable absent further controls |

**Caution:** W2 may be hard or impossible with naturally occurring annotated text; changing surface text changes provenance/rights and can change semantics. If no independently validated W2 exists, do not invent “gold” from templates or a model judge. W4's separate two calls vs joint one call is an unmatched generation budget without explicit calibration.

**Priority test of design:** Specify **one concrete pair of rival programs**, their permitted input features and frozen learned-parameter budget, and produce one hypothetical gold-valid case for which their predictions differ. If strongest meaningful models make the same predictions, record **NO DISCRIMINATING WITNESS** and stop the planned LLM run.

## 4. Human gold and source eligibility: compulsory before any model proposal

A candidate DEV document `d` is eligible only if source rights and original annotation support:

1. **Two *distinct* original human-annotated event triggers** `e_t,e_d` in the *same document and available model-visible context*, same ontology event type (unless a separately justified stratum).
2. At least **two common human-labeled argument roles** `r_1,r_2`, with source-grounded spans/referents for **both roles on both events**; gold filler/coref referents are distinct enough to disambiguate provenance.
3. No unresolved aliases, pronoun coreference, multi-span representation or repeated strings that make selecting source event **unobservable**; uncertain cases retained as `AMBIGUOUS` and excluded from the primary event-provenance endpoint, not silently called errors.
4. Event identity/role decisions validated under original schema and permitted source processing; original split, revision/archive hash, document ID and original worker/annotation dependence recorded.
5. Fixed model-view separation: no source IDs, gold spans/worker IDs or privileged event indices in prompts. Target trigger highlighting or ontology role verbalization is allowed only as a **declared interface condition**, not evidence of native event detection.

**Source status 16 Oct:** [#32](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/32) is only SOURCE FEASIBILITY. Sentry's [audit](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/32#issuecomment-6212095177) finds original RAMS docs/scorer likely one supervised focus event per record, making RAMS *probably unsuitable* for the exact two-gold-event contrast without new proof. WikiEvents is a more plausible source, but **eligible pair count, archive version/hash, source text rights and original schema remain unverified**. Do not bypass the rights gate or open official TEST to compensate. QA-SRL past historical 100 inspected candidates (55 DEV/45 TEST) and 63 selected (33 DEV/30 TEST) are development-exposed [Astra #8](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/8#issuecomment-6215695895), hence not an untouched official TEST.

If source inspection yields zero valid annotated two-event/two-role cases: **STOP THIS DESIGN for that dataset.** Retain ordinary QA-SRL for its original QA objective.

## 5. Precise observable endpoints; not paper claims

For each task unit, record original document/source group, event type, target trigger, answer role, direct/joint prompt order, exact model/revision/decoding and valid format. Each role output is tagged by scorer-only human gold as:

- `T` = target-event unique gold referent/span;
- `D` = distractor-event unique gold referent/span;
- `X` = other *source-supported* referent, neither target nor specified distractor;
- `INVALID` = no extractable answer/schema violation;
- `AMBIGUOUS` = output maps to two referents, unresolved coreference, or gold itself cannot identify original event.

For two supported roles, report the full ordered error table (`TT, DD, TD, DT, TX, XT, DX, XD, XX`), with invalid/ambiguous **separately in the original denominator**, not discarded or scored as correct. Report exactly-correct **both roles**, separately valid, per-role accuracy, whole-event rate; the comparison `DD` vs `TD+DT` is *descriptive*, never direct evidence of a native shared selector.

Use source-document/event-family clustering and predeclared error uncertainty. Conditional on covariates, compare held-out predictive log loss/calibration or likelihood of **full error categories**, not only post-hoc apparent mean differences. Freeze every competitor's fitting/training allowance and hold out entire **source documents** before looking at model predictions. Declare multiple events from one doc dependent; pretraining-exposed old news examples are developmental.

Track **target acquisition** under counterfactuals separately from any output change, and per-event pair correctness separately from unlabeled two-name outputs (Astra AP→PA collapse).

## 6. Evidence to reuse; existing work must NOT be rerun

- [Astra recovered AP/PA raw audit and QA-SRL exposure](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/8#issuecomment-6215688852): field-order run ALREADY DONE, format collapse dominates; six raw files still investigator-local; no independent receiver replay. Zero additional Qwen AP/PA runs.
- [Brain-O E057–059](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/brain-o-ideation/research/brain-o/EXPERIMENTS_057_059_NATIVE_BRIDGE_2026-10-16.md): source-sensitive native answer influence without correct donor-role retrieval; synthetic only; no established semantic binding.
- [Compass I21](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/compass-reuse/research/compass/INVESTIGATION_21_2026-10-16.md): selective interaction yet partial-program and last-hop shortcuts reproduce similar signatures, so change detection is insufficient.
- [Coordinator's already executed valid-swap nonidentification control](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/coordinator-route-identifiability-2026-10-15/research/coordinator_identifiability/README.md): reconstruction and source-lookup shortcut already observationally identical under that controlled route family. **Do not recreate this script.**
- [Coordinator RELATE-001](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/coordinator-mechanism-discriminator/research/coordinator_discriminator/RELATE_001_PROTOCOL.md): already enumerated predicate swap, participant swap, event order and voice in synthetic development; lacks native/model-competent inference and is NOT a fresh W1–W5 paper result.
- [Mira MIRA-QASRL-004](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/mira-qasrl-independent-audit-2026-10-16/research/mira_qasrl/MIRA_QASRL_004_COUNTERBALANCED_INTERFERENCE.md): already proposed overlap × target order × answer position; only design/fixtures; avoid duplicate task.
- [Nova original QA-SRL runner](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/9#issuecomment-6211859276): proposed, not executed, two-family pilot; independent original-span task and source/license/exposure gate.
- [Gate #31](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31#issuecomment-6215970009) contains full analytical nonidentification warning.

## 7. Closest literature / novelty threats (screen, not full-text clearance)

**The claim ceiling became LOWER during the frontier review**: we must check not only TabEAE and the 2026 mechanistic papers, but also prior work that already measures **multi-event consistency and field-order dependencies**:

- [EA²E, Findings NAACL 2022](https://aclanthology.org/2022.findings-naacl.202/): explicitly studies consistency constraints between linked events in a document and WikiEvents.
- [TabEAE, ACL 2023](https://aclanthology.org/2023.acl-long.701/): jointly extracts co-occurring events and argues improved semantic event boundary separation.
- [DEEIA, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.564/): multi-event prompting with dependency-guided encoding and event-specific information aggregation.
- [GEMS, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.1353/): explicitly varies **event argument generation order** and aggregates predictions.
- [CsEAE/One Small and One Large, 2024](https://arxiv.org/abs/2411.05895): event co-occurrence, document-level redundant-context interference and LLM/SLM event-argument extraction.
- [Mixing Mechanisms, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2eeff35664016c7f0f8aa704f0d9a83e-Abstract-Conference.html): joint lexical/position/reflexive retrieval competitor; do not cite ~0.95 similarity as 95% task accuracy.
- [Cell-Based Relational Binding, ACL 2026](https://aclanthology.org/2026.acl-long.2194/), [Aljaafari semantic-role circuits](https://aclanthology.org/2026.findings-acl.1964/), [Lepori encoded/use dissociation](https://aclanthology.org/2026.acl-long.676/): abstract representation, role circuits and encoding-vs-use are not free novelty claims.
- [Gan et al., Sep 2026](https://arxiv.org/abs/2609.02015): train/evaluate interface effects already studied.
- [Legible Failures, Sep 2026](https://arxiv.org/abs/2609.11216), [CoBe, Oct 2026](https://arxiv.org/abs/2610.13057): probe-mediated binding repair and counterfactual consistency are not novel generic pivots.

**Priority literature-review question:** Did EA²E, TabEAE, DEEIA, GEMS or CsEAE already use **the same event-specific coherent-error and role-order interaction endpoint**? Abstracts alone cannot answer this. If yes, require a truly new causal prediction or abandon that specific publication claim.

## 8. What we need at resumption; a bounded work order, not automatic execution

**State at pause:** This frontier is *a hypothesis and a gated scientific workplan*, not an experimental result, an automatically scheduled task, or a commitment to a paper title.

### Resume sequence — follow in order
1. **Armaggedon:** reopen this document, [#31](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31), [#32](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/32), and latest team comments. Reconcile any new source/rights/model/novelty findings. Explicitly invite agent dissent and pivots.
2. **Astra + Comput (read-only, distinct roles):** Astra writes a *one-page frozen rival-prediction matrix* specifying shared event selector vs competent query-aware joint copier vs conditionally independent field retrieval vs serialization/answer constraint. Comput tries to construct the **strongest observationally equivalent rival** and either supplies a witnessed disagreement with specified restrictions or certifies none in the present observable space. No new synthetic V5/RELATE run.
3. **Nova + Mira:** verify original WikiEvents **DEV** bytes/splits/schema/rights and counts of two annotated same-type events with two shared, distinctive roles in *one document*; Sentry verifies provenance and paper collisions. RAMS requires positive proof of two supervised events, not only an extra visible trigger. Aggregate metadata only; avoid source-text redistribution.
4. **Sentry:** compare **full methods and error analyses** (not merely abstracts) of EA²E/TabEAE/DEEIA/GEMS/CsEAE against the exact proposed metric; record `SAME CLAIM ALREADY DONE` or the precise still-open difference.
5. **Armaggedon go/no-go:** If source rights, ≥2 annotated event/role pairs, strong baseline predictions and a genuine witness are missing, **STOP/REVISE**. If they all pass, propose exactly ONE bounded exploratory DEV protocol to #31 for explicit approval. **No automatic authorization is implied by this branch.**
6. **Only after approval:** Nova single runner (two distinct pretrained families); independent Competence+format smoke **first**; preregistered balanced error-category predictions, source-cluster uncertainty, raw-output retention and document-heldout assessment. If base task cannot be solved, **STOP** without claiming special binding failure.
7. **Mechanism only conditionally:** If robust behavioral differential survives the best competitor, freeze hypotheses and run carefully site-matched native causal intervention validation under separate approvals. If not, document a negative finding and consider an alternative scientifically stronger direction.

### Named stop/pivot triggers
- **No human-gold two-event eligibility** → drop this source/task form, keep QA-SRL P0.
- **No distinguishing witness against competent joint copier** → do not run; retain descriptive error taxonomy only.
- **Earlier literature already contains the intended endpoint** → do not claim novelty; require new explanatory or causal result.
- **Only joint-output-order or label-validity artifact** → prefer interface/serialization explanation; no native binding claim.
- **Weak whole-event competence or same effect from cheap source-only model** → no unique LLM mechanism; halt extension.
- **A genuinely stronger unrelated phenomenon appears** → compare and **pivot if better**; preserve all historic evidence and independent branches.

## 9. Responsibility and publication language

| Owner | Bounded assignment |
| --- | --- |
| Armaggedon | Scientific owner, experiment deduplication, go/no-go and pivot recommendation |
| Astra | Formal and empirical prediction framework; archives, interface alternatives; **no new Qwen sweep** |
| Comput | Strongest joint syntax/salience/copier baseline and observational equivalence audit |
| Nova | Event-source/QA-SRL data runner, source provenance and optional future approved two-family calls |
| Mira | Human event/role/coref validity, source rights, multi-event natural conditions |
| Sentry | Independent novelty and communication audit, explicit prior-method collisions and acknowledgement |
| Brain-O/Compass/Brain-X | Optional independent mechanism/theory falsifiers; do not duplicate data runner or existing calibration scripts |
| Human research lead | Strategic resources, priorities, releases and approval of any major pivot requiring new resources |

**Allowed current claim:** “We have identified a prospective task-specific error decomposition and shown analytically why coherent joint errors alone cannot identify a native event selector. Original human annotation feasibility and rival predictive separation are unresolved.” **Disallowed:** “Found relational-binding circuit,” “new test of event consistency,” “human generalization proven,” “new ACL contribution,” or “real pretrained effect established.”

**Handoff:** On reactivation, reply in the corresponding **FRONTIER-001 GitHub issue** with `READ / NEW EVIDENCE / WHAT CHANGED / ONE DECISION NEEDED`. An issue posting is not receipt by a separate investigator chat.
