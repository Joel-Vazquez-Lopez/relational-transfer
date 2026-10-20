# Competing conference-paper narratives

Evidence categories: **demonstrated** = completed, auditable comparison within its stated setting; **suspected** = plausible mechanism or broader scope; **hypothetical** = proposed result not yet obtained. Venue judgments are fit assessments, not acceptance probabilities.

## A. Training-relation topology and role transfer in a controlled Transformer

**Question.** Can rearranging the same amount of actor–patient evidence make a fixed unseen relation fail, and can ordered-role supervision reduce that vulnerability?

**One-sentence contribution.** In a balanced 10-entity Transformer world, matched local rewiring causes a large withheld-pair role-transfer loss, and role-order auxiliary training largely removes it beyond an unordered-participant control; a role-related scene-state intervention has a selective functional effect.

**Demonstrated.** Rewire: −26.33 points high versus sham across eight graph families, exact graph-level p=.0078125; fresh repair: +22.59-point attenuation across another eight families; controlled state intervention: +1.208 margin units ordered-versus-unordered. Familiar performance remains high. Full sources in [audit](EVIDENCE_AUDIT.md). **Suspected:** shared-partner overlap is a useful predictor. **Unproven:** overlap alone causes the result; same mechanism operates in natural language or pretrained LLMs.

**Novelty boundary.** COGS, Bogin et al. local-structure difficulty, Zhou et al. data factors, Kumon & Yanaka internal analysis, and Aljaafari et al. role circuits make broad claims old. The provisionally differentiated unit is *the matched degree-preserving rewire of a fixed withheld role pair plus a matched ordered-versus-unordered supervision interaction and role-direction state intervention*. No exact duplicate was identified in the targeted review; full-paper literature review remains needed.

**Strongest alternative.** The rewire changes other paths and co-occurrences, and the auxiliary objective may regularize optimization rather than teach a portable semantic-role rule. The state patch may exploit movement/scene cues or off-manifold hybrid states.

**Minimum missing tests.** A graph intervention that changes shared-partner overlap while also matching additional local paths/neighbor histories, or a factorial sweep that attributes variance across these covariates; a second architecture or task rendering that removes the easiest movement rule; role-versus-energy-matched control on new graph families; pre-specified graph-level analysis. A small linguistic boundary test is valuable but must not be used to imply equivalence of mechanisms.

**Feasibility and likely reviewer rejection.** High feasibility because code, generated worlds and measures exist. Reviewers may reject ACL/EMNLP main for limited language relevance, 10 identities, one architecture, manually supplied role labels, and incomplete isolation of overlap. Findings or a compositional-generalization workshop is a realistic target with added controls; an ACL/EMNLP main attempt needs a convincing second task/architecture and sharper theory.

## B. Where relational QA breaks: extraction, question interpretation, or answer use?

**Question.** Why can an LLM emit useful role fields but answer a role question incorrectly, and which failure persists under natural syntax and candidate-free evaluation?

**One-sentence conditional contribution.** A factorized evaluation of role extraction, requested-role classification and final-answer routing identifies distinct failure modes, then tests a targeted repair against deterministic inference and direct QA.

**Demonstrated now.** On candidate-supported QA-SRL Bank, deterministic routing from printed fields greatly exceeds free final answers; a noisy verifier corrects printed fields under swapped drafts but does not repair its own final answers; typed requested-role conditioning beats token-matched neutral on two fresh item banks in two starts. Removing candidates collapses field extraction to 4/40 jointly correct while routing agreement stays high. **Suspected:** the breakdown generalizes across natural constructions and model families. **Unproven:** printed fields are internal semantic representations or a learned general relation algorithm.

**Novelty boundary.** QA-SRL, role-conditioned querying, constrained structured inference and faithful CoT already exist. The possible added value is a **quantified, intervention-based error decomposition and a repair that survives candidates, syntax, question wording and source shifts**. At present that last condition fails or is untested.

**Strongest alternative.** QA-SRL questions are templatic; a last-word rule identifies the requested role. Candidates leak the two spans. A typed tag simply supplies the answer category, and exact span scores collapse when candidate support is removed. Generated fields may be post hoc output, not mediating the answer.

**Minimum missing tests.** Independently reviewed natural labels; no-candidate role extraction; unseen question families, passive/relative/cleft cases and event anchoring; independent source; untouched, direct QA, role-query, structured-output, deterministic-route and token-matched controls; second model family. Controlled field corruption must score **gold answer**, not mere field agreement.

**Feasibility and likely reviewer rejection.** Medium. Training small models is feasible, but a proper annotated challenge and cross-family replication are material work. Current candidate-supported result alone is too narrow for ACL/EMNLP main; a validated error taxonomy and robust repair could fit Findings or main. A diagnostic-only paper may fit a workshop if the new benchmark itself is strong.

## C. Typed task conditioning as a learned answer-selection method

**Question.** Does supplying the requested semantic role during training and inference teach a model to select the correct answer from its own generated role fields?

**One-sentence current contribution.** In one Qwen0.5B QA-SRL candidate-supported setting, typed conditioning improves exact final answers over a two-token neutral cue without materially changing role-field accuracy.

**Demonstrated.** Fresh2 +44/+39 of 160 exact; fresh3 +30/+26 of 80 exact at exactly matched counted input+target tokens; strong increases in final-answer/field agreement. **Suspected:** an input task specification can be a general modular training primitive. **Unproven:** it improves candidate-free answer accuracy, question interpretation, or cross-domain transfer.

**Novelty boundary.** Explicit role queries and slot querying are established. The simple typed cue is not itself publishable as a novel method. A new contribution would require isolating *why* it helps and extending it beyond a supplied two-candidate interface. Compare against a standard role-query baseline and deterministic routing, which may already solve the measured final-answer problem more transparently.

**Strongest alternative.** The cue just translates a templatic question into a two-class label; candidate list then makes answer selection easy. The result may be an interface fix specific to the training prompt. The no-candidate stress confirms a severe extraction boundary.

**Minimum missing tests and venue.** Same as B, plus cue-noise and oracle/predicted-role decomposition, role-tag wording ablations, and candidate-free *training* rather than only out-of-distribution removal. Low-to-medium feasibility for a bounded follow-up; high risk of “obvious prompting/control token” rejection at main venues. It should probably be an intervention section inside B, not a standalone paper today.

## D. Learned computational requirements / Inverted CoT

**Question.** Can a model derive the minimal operations and dependencies needed for an unfamiliar goal, then use that plan to execute a correct answer on unseen operation combinations?

**One-sentence hypothetical contribution.** Training on goal-derived requirements causes compositional procedure construction and causal use of the generated procedure across task families.

**Demonstrated now.** Corrected v4 clause learner selects both operation directions on 400/400 mixed items *when oracle clause views are supplied*, while intact-question primary test fails. V5 learned selector supplies the views and yields 400/400 in five paired starts, but a hand rule does the same and most mixed questions reuse a previously evaluated question string. No final event answer is computed in v4/v5. Earlier plan-to-answer runs are pad-mask invalid. **Suspected:** a richer system can learn to build procedures. **Entirely hypothetical:** general reasoning transfer, minimality, faithful plan use, or advantage over strong planning methods.

**Novelty boundary.** LAMBADA, Reverse Chain, DaSLaM, ExeDec, PLAN-TUNING and faithful symbolic CoT cover backward reasoning, trained decomposition and execution. A new name or output order is insufficient. The only plausible strong distinction is *learned minimal requirement construction under withheld dependency graphs with verified counterfactual plan use*.

**Strongest alternative.** The selector learns a fixed grammar rule and adds a separate model with explicit keep/drop labels; it does not discover a procedure. An executor could ignore printed plans or use a shortcut.

**Minimum missing tests.** A task generator with at least four operation primitives and several withheld dependency structures; programmatically verified answers; matched answer-only/forward trace/plan/placebo supervision and compute; intact-question full answers; rule-based and oracle-plan controls; counterfactual edits with predicted effects; multiple starts; a second task family and natural-language boundary. Feasibility is **low-to-medium** for an initial controlled paper and **low** for an ICLR/ICML/NeurIPS-level general method without a much larger program. Keep as a separate research branch.

## E. Construction exposure causes a cross-form trade-off

**Question.** Does changing training syntax while holding events and budget fixed redirect actor/patient accuracy across forms?

**Demonstrated.** The aligned-versus-decoupled Qwen0.5B intervention repeats an opposite passive/active effect on two lexical banks, with all valid outputs for those forms. Mixed coverage failed to dominate. **Novelty threat:** augmentation and syntactic-transfer trade-offs are well studied. **Missing:** natural reviewed test, distinct model family, held-out question families and a successful objective that beats the data-only control. This is a strong subsidiary result or possible Findings-style analysis, not obviously the strongest standalone conference story.

## Comparative ranking

Scores are qualitative judgments about **the current record**, not acceptance estimates. `High` risk means greater chance that essential evidence fails or is unavailable.

| Rank | Candidate | Novelty position | Current evidence | Reviewer interest | Additional work | Risk | Best present role |
|---:|---|---|---|---|---|---|---|
| 1 | **A. Matched rewiring + ordered-role repair** | Medium–high if graph mechanism isolated | **High within controlled task** | Medium–high in compositional learning | Moderate | Medium | Near-term paper spine |
| 2 | **B. LLM failure-chain diagnosis** | Medium if external repair survives | Medium, **conditional on candidates** | High | Substantial | High | Second paper direction |
| 3 | E. Construction trade-off | Medium | Medium on synthetic LLM forms | Medium | Moderate–substantial | Medium–high | LLM section or focused paper |
| 4 | C. Typed role cue alone | Low | Strong but narrow | Medium | Substantial for generalization | High | Intervention inside B |
| 5 | D. Inverted CoT method | Potentially high only after new results | Low; narrow feasibility only | High | Large | Very high | Separate ambitious program |

**Recommendation:** pursue A as the strongest paper with modest additional work. Pursue B as the strongest ambitious *LLM-focused* paper, with C as one arm and E as a motivating robustness result. D should remain a separate research project until it produces intact-question, final-answer, cross-task transfer. Combining A, B and D as if they prove one mechanism would weaken the causal story. A short, explicitly bounded LLM evaluation section in A can show why the question matters, but it must report the failed natural transfers and cannot be treated as confirmation of the graph mechanism.
