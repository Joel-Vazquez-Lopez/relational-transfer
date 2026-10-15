# Independent scientific review: ACL handoff, 15 October 2026

**Decision: CONDITIONAL APPROVAL of scientific direction; revise the proposed confirmation protocol before execution.**

**Review scope.** I inspected the uploaded `ACL_HANDOFF_2026-10-15.zip` (262 entries, ~6.7 MB uncompressed). I read the handoff's scientific summary, evidence register, historical inventory, frozen diagnostic audit, experimental protocol, mechanistic and model/data plans, literature audit, timeline, and selected raw prediction records. I ran the included integrity, data-selection, paired-reanalysis, historical patch-summary, and power-sensitivity scripts; separately reconstructed the 1.5B joint-correctness cross-tab directly from raw predictions. No new model inference, training, use of sealed examples, or GitHub changes occurred.

## 1. Findings supported by the archive

- `tools/verify.py` passed: 215 archived source-copy hashes, 253 protected review-file hashes and 336 main/follow-up prediction records checked. Verification demonstrates file integrity of the *snapshot*, not independent reproduction of model inference or training. Checking against the original source tree cannot be done from this upload because the full original workspace is not mounted here.
- `tools/audit_selection.py` reproduced the original-annotation selection: 56 of 100 previously examined events pass the narrow question filter; 24 were selected. The original QA-SRL consensus spans reproduce, but there is no independent semantic agent/patient validation.
- `tools/recompute.py` reproduced Qwen2.5-1.5B results. My separate tally of the archived parsed fields gives **2/24** jointly correct for direct QA; **11/24** for blind extraction plus oracle-role routing; **9/24** for question-visible, field-only extraction plus routing; and **6/24** for the original visible structured fields routed independently. Direct-versus-blind paired joint transitions are 10 gains and one loss. The other 13 events are ties (one both correct, 12 both wrong).
- Observed direct-to-blind joint gain **+37.50 percentage points**, with the archived bootstrap over 19 source-document clusters reporting **[+16.67,+56.52]**. This is a descriptive interval for an adaptively selected 24-event development subset, not a prespecified or externally validated population effect.
- Blind versus format-matched, question-visible field-only routing: **+8.33 percentage points** (11/24 vs 9/24); paired document interval **[-8.00,+24.00]**. This does not demonstrate that excluding the question causes the improvement. In Qwen0.5B, this comparison points the other way (2/24 vs 3/24).
- The original 1.5B visible structured-output arm omitted its required `Answer` field in **45/48** responses. Therefore, comparisons with that *generated-final-answer* arm are heavily confounded by format compliance. More generally, role routing uses an **oracle question-role selector** in the original principal diagnostic; the post hoc fitted classifier has exact question-string overlap on 18/48 held-out questions. End-to-end unseen-question generalization is open.
- `tools/recompute_patching.py` reproduced the *archived selected-case* Qwen patch summary (41/48 correct actor under actor-question donor, 0/48 under patient-question donor). It does not rerun a Transformer; the 48 examples are all active and position/role are confounded. Do not claim a semantic-role circuit.
- Historical experimental inventory appropriately distinguishes graph interventions, ability-matching failure, pretrained-model diagnostics, adapted-model branching and failed interventions.

## 2. The research direction is promising—what is and is not novel

The best candidate question remains: **When and why do pretrained LLMs fail to coordinate natural-language event interpretation, the requested semantic role, and final answer production?** A stronger empirical claim would show that a *specific, predictable interaction* among linguistic construction and question conditions explains unseen failures beyond component difficulty and output-format effects. An even stronger claim would tie the interaction to a selective internal causal model that predicts counterfactual behavior and supports a robust intervention.

Existing work sharply constrains novelty:

- Gur-Arieh, Geva & Geiger, *Mixing Mechanisms* (ICLR 2026) already reports a mixture of positional, lexical and reflexive binding-retrieval mechanisms across nine models and ten tasks, including open-ended text. **Multiple factors/mechanisms as such are not novel.** https://proceedings.iclr.cc/paper_files/paper/2026/hash/2eeff35664016c7f0f8aa704f0d9a83e-Abstract-Conference.html
- Aljaafari, Carvalho & Freitas, *Emergence and Localisation of Semantic Role Circuits in LLMs* (Findings ACL 2026) reports causally implicated semantic-role circuits and cross-scale patterns. **Showing role-related circuits as such is not novel.** https://aclanthology.org/2026.findings-acl.1964/
- Ravulapalli, Chadha & Hari, *Legible Failures* (September 2026 preprint) reports recoverable correct binding information and probe-guided steering on entity-obligation binding. **Correct information existing despite incorrect final answers is not novel.** https://arxiv.org/abs/2609.11216

A paper that merely observes pipeline gains, prints role fields, adds multiple model families, or uses activation patching is not distinct enough. The handoff's literature audit correctly acknowledges these limits; competitor code and more forward citations still merit inspection.

## 3. Required protocol revisions before approval

### A. Separate the primary scientific effect from the pipeline benchmark

The 24-event pilot supports a pipeline discrepancy, not the claimed question-interference mechanism. Freeze two separately interpretable confirmatory outcomes rather than letting the prettier one become the headline later: (1) a **semantic-extraction context contrast** using exactly matched field formats with a relevant question versus a reviewed length-matched irrelevant question (plus no-question reference); and (2) an **end-to-end performance-versus-compute comparison** with direct QA, stronger two-pass direct QA, generated extraction plus autonomous routing, and a conventional SRL + router. The first tests context effects; the second tests whether structured processing is practically advantageous. Report both even if directions disagree.

### B. Validate semantic roles without forcing grammatical mappings

The previous pilot is restricted to active, subject-first grammatical subject/object questions. That is inadequate to support a paper about semantic agent/patient binding. The rubric must distinguish agent, patient/theme, experiencer, beneficiary, recipient, omitted arguments, non-agentive subjects, ambiguous predicate senses and coreference. Record *ineligible cases* without dropping them based on model outputs. Require two reviewers to work blind to model predictions, adjudicate disagreement, and show agreement per role/stratum. Human-authored active/passive counterparts are controlled transformations, **not independent natural sentences**; keep an additional native naturally occurring panel. Confirm access and license obligations for QA-SRL and UDS sources before redistribution.

### C. Make model scale scientifically relevant

Six models from three families (Qwen2.5-1.5B, SmolLM2-1.7B, Llama3.2-3B; base + instruct) are useful for tightly controlled small-model comparisons, but do not establish modern larger-LLM relevance. Retain the three instruct families as a core confirmatory panel if they pass interface calibration, and **pre-budget at least one 7–8B+ and ideally a second independent larger-family instruct model**, whether via institutional GPU or provider API with pinned versions. Base checkpoints can be secondary: forcing the same schema across base and instruct models risks measuring interface-following rather than semantic knowledge. Provider-only inference is for behavioral replication, not hidden-state patching. Freeze any model substitutions before test outputs are seen.

### D. Reduce the first execution scope and budget it from measurements

The draft's 600 event families, 600 active/passive counterparts and 480 additional rewrites imply ~1,680 sentence realizations; with 21 generations per sentence and six checkpoints, the plan estimates **211,680 generations**. It explicitly excludes extra context arms and ancillary work, so it is not a cost ceiling. The stated 100–180 person-hours for two reviewers, adjudication and these many realizations is a planning assumption, not demonstrated feasibility; reviewer calibration may reveal a much higher burden. Do not start this run without a *measured* per-model token/time/cost pilot and an annotation-throughput estimate. Maintain a substantial final dataset if resources permit; stage it: feasibility/calibration, prospectively locked core paired panel, then extended constructions and cross-family/size checks, with the entire confirmation scope fixed before unsealing. The proposed sensitivity script itself shows 5 pp effects may need >600 events under plausible variances; use document-clustered simulation based on development covariance rather than interpret a null at 600 as absence of an interaction.

### E. Preregister a robust multi-factor test

The proposed interaction estimand

`I = (accuracy_relevant - accuracy_irrelevant)_passive - (accuracy_relevant - accuracy_irrelevant)_active`

is well defined on the risk-difference scale and is an excellent starting point. Predefine the event-level averaging, question variants, accepted spans, source weights, cluster bootstrap and handling of missing/invalid outputs. Put **independent-effects / component models (A)** and **interaction models (B)** into actual held-out prediction competition using log loss, not only in-sample coefficients. Prespecify negative-direction harmful-query predictions versus non-directional interaction detection; avoid turning any significant interaction into a post hoc confirmation of the original hypothesis. Explicitly audit floors/ceilings and note that risk-scale interactions need not imply internal computational interactions.

### F. Keep the mechanism and intervention strictly gated

Before activation patching, require a robust semantic effect on natural and controlled constructions in two families. Activation patching must cross semantic role, position, lexical identity and query context and compare whole-state steering with selective, independently chosen internal subspaces/sites; use identity and irrelevant-donor controls. Do not use test labels or oracle donor selection in an end-to-end repair. If mechanism tests fail, write a behavioral paper; if a correction is chosen after the discovery confirmation, preserve a separate unseen intervention confirmation set.

## 4. Additional QA and reproducibility issues

- **Portability test bug:** `tools/validate_portability.py` hardcodes `TemporaryDirectory(..., dir='/private/tmp')`, which fails on this Linux runner because `/private/tmp` is absent. Replace with a standard `TemporaryDirectory(prefix=...)` and run the relocation test on more than one OS. This does not invalidate the main reanalysis, which ran successfully.
- `verify.py` checks packaged file hashes. Its optional `--check-originals` requires the original project tree and is not applicable to a standalone handoff ZIP. This limitation is correctly indicated by its default output.
- The classifier's 48/48 accuracy is not proof of transfer to unseen question forms: there was exact-string overlap on 18 of the 48 questions. Future classification should be tested with question-template-family held out.
- The frozen primary analysis-source hash was saved after the full run began because the smoke-test assertion did not stop the shell; the record discloses this. Future automated smoke gates should terminate execution rather than merely log an assertion.
- Exactly how accepted answer spans and semantic alternatives are scored should be reviewed on development data and frozen before the new final split; schema-invalid outputs remain visible in error analysis and count as failed attempts for end-to-end comparisons.
- Model outputs, cost and generation-token counts must be logged exactly, not via batch-padded output lengths.

## 5. Recommended go/no-go decisions for 23 October

**Approve now:** scientific framing; independent discovery-first approach; multi-family LLM centerpiece; archived diagnostic as *exploratory* motivation; exposure registry and provenance plan; matched-format and stronger end-to-end baselines; selective causal follow-up *after* the behavioral gate; dedicated private GitHub workflow.

**Revise before freezing:** annotation workload and source licenses; semantic ontology; 600-family power/precision target; real model and GPU/API budget; primary outcome and multiple comparisons; comparison against state-of-the-art SRL pipeline; larger-model capacity check; held-out linguistic construction strategy.

**Not supported today:** question-independent extraction being the *cause* of the observed 1.5B gain; any general pretrained-model role-interference mechanism; an unprecedented novel training method; correctness of agent/patient labels in the 24-item pilot; broad family-level generalization; a positive selective mechanistic finding.

### Proposed implementation order

1. Create a private review branch with handoff documentation, an evidence/claim registry and reproducible offline scoring tests. Do **not** import giant checkpoints or accidentally expose restricted datasets.
2. Implement a stable data schema separating grammatical function, semantic role, event identity, original versus rewritten sentence and accepted spans; implement two-reviewer/adjudication workflow and a documented data-exposure registry.
3. Run a low-cost **technical/development calibration only**, measuring interface compliance, inference costs, and native-versus-rewrite label feasibility on several checkpoints, including a larger model if credits permit. No sealed outcomes.
4. Finalize the primary effect, specific rival predictions, model register, power/precision budget and protocol hashes. Only then run independent confirmation.
5. If effects survive, proceed to controlled mechanistic interventions and, possibly, one targeted correction. Start the introduction, related work and methods before confirmation; write results only after the analyses are locked.

## Bottom line

**The handoff passes a meaningful research-process audit and gives a credible route to a strong scientific paper. It does not yet supply an ACL-level novel finding.** The unusually valuable part is a clear empirical discrepancy combined with well-preserved negative results, exact provenance, and falsifiable rival explanations. The major threats are semantic-label ambiguity, format/compute effects, over-ambitious annotation and call budgets, and neighboring prior work. Address those early; do not let a December deadline convert exploratory observations into unwarranted conclusions.