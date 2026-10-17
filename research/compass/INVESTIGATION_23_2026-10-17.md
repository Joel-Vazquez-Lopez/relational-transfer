# Compass Investigation 23 — Primary-method novelty audit: relational subspaces, mixed retrieval, and deployment

**17 October 2026 | Independent read-only literature assessment | No model inference | No new dataset or benchmark.**
Branch: `research/compass-reuse`. Status: **prior-art claim restriction; not a paper novelty certificate**.

## Scope and primary provenance

I independently read method/experimental/control portions, not merely abstracts, of:

1. **Dai, Heinzerling & Inui, ACL 2026**, *Cell-Based Representation of Relational Binding in Language Models*. Publication metadata: https://aclanthology.org/2026.acl-long.2194/ ; inspected full-text **arXiv HTML v1**: https://arxiv.org/html/2604.19052 (especially sections 3.1–3.3, appendices A.22–A.29). **Caveat:** final 20MB ACL proceedings PDF could not be fetched through the public-reader endpoint; possible v1→camera-ready differences were **not** independently compared. Do not assert final-paper appendix parity.
2. **Gur-Arieh, Geva & Geiger, ICLR 2026**, *Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context*. Official: https://proceedings.iclr.cc/paper_files/paper/2026/hash/2eeff35664016c7f0f8aa704f0d9a83e-Abstract-Conference.html ; inspected https://arxiv.org/html/2510.06182 , sections 3–5, selected appendix/control methods. **Caveat:** no independent bitwise comparison to final ICLR proceedings; OpenReview PDF required bot verification.
3. **Lepori, Linzen, Yuan & Filippova, ACL 2026**, *Language Models Struggle to Use Representations Learned In-Context*. Direct **final ACL PDF inspected**: https://aclanthology.org/2026.acl-long.676.pdf ; sections 2–5, discussion and limitations, including page render check. No independent reproduction.

Also read team Sentry novelty register https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/30 , coordinator FRONTIER-001 https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33 , governance https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31 , and our Compass I11/I19–I22. This audit is an adversarial review **of source-level methods**, not an empirical replication.

## 1. Actual experiment contracts (important details lost in abstracts)

### Dai 2026 CBR: the causal grid and its controls already exist

- Source: *automatically generated multi-sentence discourse*, 5 relation/entity attribute domains and different templates, plus naturalistic paraphrases; ground-truth **entity-introduction index** `ei` and **relation-type-introduction index** `ri` associated with each attribute token. Authors use Llama3-8B-Instruct and Qwen3-8B. See §3.1 and appendix A.2.
- Method: fit **supervised PLS regression** from middle-layer (typically **layer 15**) *attribute-token residual activations* to `[ei,ri]`, roughly **2–5 components**. A 2D plot organizes encoded attributes into cells indexed by both entity and relation order. **This is a fitted supervised index basis, not proof that the unmodified model consciously constructs a symbolic grid.**
- Causality: steer context attribute activations toward target cell coordinates; modify specific relation-index, entity-index and final-token residual directions. Evaluate **logits for the intended original and intervention-expected attributes**, not just any answer flip. Random-direction controls impair performance much less than CBR-direction changes (paper's Figure 5); see §3.2, Figure 6 and appendices A.22–A.24.
- Generalization: domain/template variation, shuffled/ablated probes, and **Re-DocRED** real-world relation extraction *index decoding*. But the natural dataset appendix A.29 reports stronger representation decodability when development samples resemble training examples in embedding similarity; this is **not** evidence that the authors have already demonstrated our intended natural two-event role-error law.
- Claimed result: **causally relevant indexed relational representation** in these controlled tests, across two pretrained families.

**Collision:** first causal entity×relation binding representation, first grid/latent cell, first steerable role index, first cross-context translation, first random-direction-vs-binding-direction control = **NO** novelty for our project. Stronger controls needed than a random-subspace sham alone to establish unique native event/role routing.

### Gur-Arieh 2026 Mixing Mechanisms: three causal rivals and a calibrated combined account

- Source: synthetic *bound-entity group* tasks (9 models, 3 families, 10 tasks), interchange-intervention counterfactuals; section 5 interleaves **generated entity-less filler text** between groups (up to 500 filler tokens). This is not an original human-annotated same-event-type role-collision dataset.
- Distinct hypotheses: **position** `P` (which source group position), **lexical** `L` (lookup through query partner token), **reflexive** `R` (direct pointer back to answer token). Causal-mask asymmetry matters: when a queried token appears after an answer in a group, naive backwards lexical copy is impossible, motivating reflexive pointer. See §3.1.
- The counterfactual group matrix *deliberately makes the three mechanisms predict different answer tokens*: `P→jam`, `L→ale`, `R→pie`, untreated `tea` in §3.2 Figure 1. The authors explicitly identify and then repair a pointer-vs-literal-answer confound in §3.4. That is a closer methodological precedent for our insistence on an independently separating witness than an abstract-only novelty map suggests.
- Their learned causal mixture adds a **Gaussian positional term whose spread depends quadratically on group index**, plus weighted lexical/reflexive one-hot terms, tested using interchange logit distributions; **8,000 averaged index combinations** from 150 intervened examples per `P,L,R` triple. §4 reports **~0.95 Jensen–Shannon similarity (JSS = 1 − JSD base-2)** against model distributions, with substantial ablation effects. **Not** 95% exact next-token answer agreement; do not quote or repeat that imprecisely.
- Binding degradation in interior positions plus signal mixtures, longer entity lists, and added uninformative text are already directly studied.

**Collision:** first positional/lexical/reflexive source selection, first mixed entity retrieval, first entity-load-induced positional errors, first causal prediction of source retrieval distribution, first use of counterfactuals to separate such routes = **NO** novelty.
**Key scope distinction (unproven opportunity):** a predeclared *wrong event×role source provenance* model on **original independently adjudicated human text**, fitting competence-matched syntax/lexical/position/source-salience predictors and held-out source document clusters, could be empirically different. Simply adding named bystanders, swapping roles or displaying a worse score is NOT enough.

### Lepori 2026: encoding≠functional deployment is explicitly addressed

- Source: randomized arbitrary-word random walks over grid/line topologies, paired with either direct next-word prediction or few-shot **Adaptive World Modeling** rule application. In some conditions 10 examples define a 1-,2-,3-step mapping on an already induced topology (§§2–4).
- Methods: measure latent topology via distance correlation/Dirichlet energy and compare direct **prefilled** vs instruction/delayed response conditions. Latent representation quality remains similar although downstream task behavior differs; inspect Figures 3–7 and prompt-condition comparisons (§3).
- Strong baseline caution: the authors identify a **random observed-transition guessing** shortcut for one-step AWM and distinguish it from underlying rule induction (§4); they discuss model inability to use representations, task meta-learning and longer-context variants.
- They examine frontier chain-of-thought reasoning models in §5, but **do not solve deployment mechanistically**; §7 lists this as a limitation.

**Collision:** first latent-knowledge-vs-answer-use split, first short-program composition failure despite decodable structure, first claim that downstream deployment is inert = **NO** novelty. Compass I11 and I16 are calibration evidence, not an original overarching thesis.

## 2. Exact claim matrix for our project

| Candidate sentence / claim | Prior-art status | Current Compass/team evidence | Scientific decision |
| --- | --- | --- | --- |
| "We discovered entity×relation grid cells in Transformers" | Occupied by Dai 2026 (and earlier binding IDs) | No comparable natural pretrained subspace or causal unique-route proof | **STOP** |
| "Binding uses position, lexical cues and pointers" | Occupied by Gur-Arieh 2026 | Our input edits and other agents' patches are consistent with shortcuts | **STOP** |
| "Model knows a relation but fails to use it" | Lepori/Chen 2026 direct collision | Auxiliary probes/heads are not demonstrated native answer intermediates; I11/I16 fail whole-world | **STOP** |
| "Intervention flips output, so semantic binding circuit exists" | Subspace steering, causal mixtures and patch confounds already studied | I19–I21: arbitrary flip ≠ correct update; partial-program shortcut succeeds | **STOP** |
| "Extra named entities cause wrong-event role borrowing" | Partly occupied by entity-load mixing; cross-event confusion literature | Owner-reported Qwen 64/64→57/64, **seven** errors all second-target; predeclared ≥8 threshold **failed**; raw not independently replayed | **NOT an established effect/novelty** |
| "Predict exact wrong-event *source* better than published mixed retrieval under controlled human text" | Precise comparison **not demonstrated** from the three methods inspected; other event-extraction papers still possible | Two-event I22 conditional source label is degenerate, and Comput/BX have already made rival and output parsers | **POSSIBLE narrowly scoped open empirical question, not currently proved** |
| "A unique native semantic event selector mediates two-role answers" | Related CBR/ordering-ID and causal abstraction prior art | FRONTIER-002/003 pre-inference NO-GO; unrestricted competent source copier observationally equivalent | **NO-GO without a new valid path-level witness** |

## 3. New, actionable reviewer corrections

1. **Do not misquote Gur-Arieh's 0.95:** distribution-level **Jensen–Shannon similarity**, not fraction of correct token predictions. Train/test split is on aggregated `P,L,R` intervention-index combinations; it is not an original-source-document split. In any manuscript, specify that their 95% arises from a calibrated causal-model distributional match.
2. **Do not equate CBR's independent entity/relation indices with proving native three-stage event address→participant selection→answer use.** Their PLS regression is trained on index labels, with purposeful index steering, which is stronger than our raw probes on one axis but does not settle our three-stage native path.
3. **Do not claim simply that "natural language has never been examined":** CBR Appendix A.29 evaluates Re-DocRED index decoding; Gur-Arieh adds naturalistic filler around bindings. The potentially distinct property is *source-verified human event×role collisions with source-disjoint predictive provenance*, not mere natural-looking prose.
4. **Our initial seven-error headline carries almost no source-selection mechanistic leverage.** Under I22's two-event closed candidate set, `OTHER_EVENT` is entailed by a wrong candidate answer. Worse, all seven are target-second, so FIRST_EVENT fallback and OTHER_EVENT policies coincide. The load-induced accuracy drop is still descriptive but not a binding-index signature, and >=8 prerequisite failed.
5. **Even a human-gold three-event source panel with diverging FIRST and NEAREST error predictions cannot distinguish an unrestricted syntax/content copier from a semantically named event selector.** Treat strong-rival **predictive superiority** as a behavioral result, not unique algorithm proof. For a native claim, preregister causal mediator graph, known-route competent controls, correct target acquisition, role-specific shams, and route necessity.
6. **Keep role-coded original-human annotations separate from recast universal actor/patient fields.** QA-SRL human question→span output, SemEval typed nominal relation labels, and event coreference gold are not interchangeable. NOVA owns original scoring/runner, Mira rights/semantics, Comput parser rivals, Brain-X role taxonomy, Sentry novelty, Armaggedon go/no-go. Compass owns adversarial route and metric identifiability.

## 4. Single actionable next scientific decision

**My recommendation to coordinator:** Retain current single-paper north star *only as a hypothesis* until (i) owner releases approved original Qwen 64-case item-level logs and related earlier confound reports for recipient verification, (ii) an independent review produces a legal, original-human, source-document-disjoint holdout with non-degenerate event-role provenance, (iii) the strongest compound-key/copier and Gur-Arieh-style positional/lexical/reflexive *combined* rival makes a frozen, genuinely different prediction, and (iv) at least two pretrained families pass identical clean competence and answer-scoring contracts. If no rival witness or source panel, **STOP/pivot**. Model inference, sealed data, downloaded corpus and paid compute remain blocked under issue #31.

**No additional model run authorized or requested here.** This is a primary-text novelty *narrowing* and feasibility assessment, not a claim of comprehensively ruling out every 2026 paper. The two arXiv method versions have not been compared bit-for-bit against final conference PDF updates. Preserve all failed gates and the possible non-novel outcome.

**Compass scientific status:** I23 complete as literature-only audit; I22 formal wrong-source nonidentification remains intact.
