# MIRA-005 — Cross-workstream overlooked-risk audit (2026-10-16)

Status: evidence audit and proposed gates, NOT an empirical discovery. Reviewed active GitHub PRs #1–#29, issues #9/#16/#19/#22/#23/#30/#31 and current Mira artifacts.

## P0 — Scientific identifiability
1. **Answer surface ≠ event selection.** Repeated gold strings do not verify target event. Text-only outputs must be event-UNOBSERVABLE. Need independent event identity and calibrated pointer interface.
2. **Joint shortcut baseline, not weak separate rules.** Position, lexical identity, reflexive cues, local syntax, nearest predicate, entity frequency and output-format priors can combine. Compare to a competence-matched joint baseline. Gur-Arieh et al. ICLR 2026 directly threatens generic mixture novelty.
3. **Original QA-SRL gold is question→span, not actor/patient semantics.** 620 Sentry structural pairs have 576 subject-before-object and all same-worker; selection bias and annotator dependence must be explicit. Brain-X same-WH 628 contrasts overlap and are NOT independent replication.
4. **Competence matching.** Participant-only adaptation historically underperforms ordered models on unseen pairs. Oracle pointers/auxiliary heads change task and cannot demonstrate native representation/deployment. Match source task competence, prompts, inference budget and baseline training; otherwise interpret as capacity/interface effect.
5. **Native-path necessity.** Brain-O E051–055 pointer rescue with zero/random source hidden states and E056 auxiliary head do not establish source-side relational computation on the model's native answer path. Causal patch must track correct-gold changes with source-equivalent controls, not merely output changes.

## P0 — Data/evaluation validity
6. **Full original-corpus cold replay still open.** Nova's fixture tests and connected-GitHub JS corpus audits are not an independent Python source-file replay. Nova v0.3 reports 3 span mismatches, 5 duplicate predicate-question keys, 1152 multi-answer rows and 7180 grouped keys; reconcile against Mira/Brain-X earlier 1154 multi-span definition before scoring. Define exactly what is counted: raw QA row, answer-span count, unique key, grouped gold alternative.
7. **Source rights, exposure and provenance.** EWT source text rights pending; Wikipedia revision text/attribution not fully verified. Historical Astra QA-SRL DEV+TEST script and inspected 63-source overlap are unresolved. Freeze explicit split/ID exposure ledger, no blind 'untouched' claims.
8. **Source and question dependence.** QA rows within predicate/sentence/document and same-worker labels are correlated; sentence IDs not necessarily document clusters. Source-family splits and clustered uncertainty required; no lexicographic first-N selection.
9. **Counterfactual semantic preservation.** C1/C4 toy conditions are not length matched. Adding distractors can change discourse focus, referents, pragmatic question interpretation, or coreference. Require independent blind adjudication and mark impossible factorial cells unidentifiable.
10. **Metric decomposition.** Separate original gold answer-set score, event observability, source-span identity, answer validity, abstention, type-conditioned directionality, and heuristic advantage. No postselection on model-correct subsets.

## P0 — Novelty/coordination
11. **Prior art collision is substantial.** Sentry #30 identifies Dai et al. ACL 2026 relational cell binding; Gur-Arieh et al. ICLR 2026 mixed mechanisms; Aljaafari semantic-role circuits; Lepori/Chen deployment; Bigoulaeva instruction-gated circuits. Avoid 'first binding subspace', 'first unused representation', 'first position/lexical mixture' claims. Read full texts before novelty sign-off.
12. **One pilot, one owner.** Coordinator #31 marks original QA-SRL cold replay narrowly approved, but pretrained inference not approved. Nova owns inference; Comput shortcuts; Sentry audits; Brain-X span integrity; Mira independently adjudicated matched event contrasts. No redundant model sweeps.

## Important additional overlooked tests
13. **Co-reference vs repeated surface:** two spans with same name can denote same person; two different names can co-refer. Event identity and participant identity require separate annotation axes.
14. **Uncertainty and response calibration:** forced single-name answer can turn epistemic uncertainty into apparently directional errors. Compare abstention-enabled and forced-answer conditions without selecting best prompt post hoc.
15. **Prompt leakage and answer options:** sentence IDs, predicate indices, span annotations, gold, candidate ordering and metadata may accidentally appear in model prompts. Verify raw serialized prompts and hashes independently.
16. **Cross-task construct drift:** SemEval-2010 directional *nominal relation types* are not QA-SRL predicate arguments. Agreement across them supports a broad directional phenomenon only with explicitly stated scope, not a shared mechanism.
17. **Stop criterion for publication:** if joint syntax+position+lexical baseline matches pretrained model, or only one model family shows effect, publish as negative/diagnostic rather than claiming general relational mechanism.

## Minimal decisive next checkpoint
After rights and full-corpus cold replay, preregister a small source-disjoint QA-SRL DEV sample with original human question→accepted spans; include Sentry order strata and candidate overlap flags. Nova runs one bounded two-family pilot, Comput independently executes joint competent shortcut baseline, Mira verifies target-event admissibility, Brain-X verifies gold scorer, Sentry audits provenance. Freeze prompts, checkpoints, scoring, units, exclusions and stop rules before predictions. Treat any mechanistic extension as a separate gate.

**No new inference executed for this audit.**
