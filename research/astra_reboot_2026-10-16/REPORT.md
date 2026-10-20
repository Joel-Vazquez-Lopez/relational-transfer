# Astra scientific reboot — 16 October 2026

**Recommendation: retire the synthetic field-order expansion; replace the broad “semantic binding versus copying” thesis with a narrowly predictive test of event-coherent errors versus fieldwise retrieval. Keep a methodological pivot as a conditional alternative. No new model run is justified yet.**

The most consequential findings in this session are archival and methodological: the field-order experiment marked pending in the inspected GitHub snapshot was already completed and discussed with the user; the recovered QA-SRL exposure list contains official-test items; and additional prior art occupies the obvious interface/adaptation pivot. These reconcile the record and constrain the next decision. They are not new neural mechanisms or first discoveries of the user's earlier results.

## History reconciliation after reading the audit chat

I read recent and older scientific turns of **Audit Transformer forecasting**, the archived **Assemble research handoff archive** chat, the 57-report inventory in `ACL_HANDOFF_2026-10-15_REV2/EXPERIMENT_INVENTORY.md`, the Compass foundation experiment index, and original reports for the transport and fixed-field studies. This is a targeted recovery of the research history, not a claim to have reproduced all 57 studies. The unidentified “what is saved” chat was not located by title; the user confirmed that the experiment records should supply the context.

The programme already spans graph construction, matched rewiring, strict factorial failure, five-arm adaptation, ability matching, internal patching, auxiliary participant controls, natural QA-SRL/EWT transfer, output routing, candidate-free tests, multiple pretrained families, and the completed field-order study. A new investigator must preserve this ancestry and its negative findings.

| Prior line | Status recovered | Consequence for this reboot |
|---|---|---|
| Graph/rewiring and ordered-role auxiliaries | Executed; positive absolute effects, failed strict isolation and ability-matched extension | Neither a clean graph mechanism nor an LLM explanation has been established |
| Ordinary/ordered/participant adaptation | Executed; participant-only and untouched models outperform ordered roles on the focal cleft comparison | Role-specific supervision is not the uniquely supported protection mechanism |
| Event-role transport | Original report inspected: −3/144 strict joint answers in each of two related starts versus matched slot SFT | Do not rename the same consistency objective as a new intervention |
| Role-first, typed goals, fixed-field continuation | Executed; visible fields influence answers, candidate support and routing matter | Candidate B largely overlaps existing work; archive explanation only unless a genuinely missing contrast exists |
| Natural transfer and candidate-free tests | Executed; mixed transfer, source/label problems, failed two-start screen | A synthetic gain cannot be promoted to authentic language competence |
| Cross-family and field-order panels | Executed and already discussed with the user | Reuse them; this session's contribution is independent recounting and synchronization |
| Selective updating and adaptation of reusable computation | Previously proposed in the audit chat; no new execution verified here | Potential questions, not new inventions in this reboot |

The strongest unresolved adaptation puzzle is why supervision without ordered roles can protect performance better than explicit ordered-role supervision. Optimization, changed interfaces, preserved lexical extraction, and altered computation all remain competitors. The existing evidence does not select among them. Calling this “computational preservation” would currently rename the observation rather than explain it.

The fixed-field report is particularly constraining: ordinary continuation reproduced prior answers 160/160 per start; explicit tags helped when cached requested fields were correct and hurt when they were wrong. Swapping visible fields changed outputs. This already tests text-mediated answer selection; it does not isolate a native hidden-role pathway. These are original-report counts, not another independent recount here.

### Additional prior-art checks prompted by the history

**Legible Failures** is a direct collision for “represented but unused bindings can be repaired.” Its synthetic, single-token task decodes query-specific bindings; a validation-selected, probe-targeted residual intervention improves answers with random-direction controls. The paper explicitly leaves downstream computation unresolved, reports harm to some initially correct answers, and finds weaker results for mutable/scoped state. Thus merely applying probe repair is insufficient novelty. A possible remaining question concerns how a native computation changes after a scoped update, but that requires a specific rival and goes beyond our present evidence. [Original methods, §§3–5](https://arxiv.org/html/2609.11216v1).

**CoBe** already studies changing consequences while preserving unaffected facts in conversational counterfactual rewriting. Its examples derive from 106 core scenarios with augmentation, so variants should not be treated as independent structures. The earlier selective-update proposal therefore needs an explanatory mechanism or learning result beyond demonstrating preservation/update failures. An explicit state overwrite also differs from a causal counterfactual; neither task licenses conclusions about the other without specifying semantics. [Original task and construction, §3](https://arxiv.org/html/2610.13057v1).

**Structure Tax** also overlaps broad output-order claims; its abstract was checked, not a full methods audit. [Paper record](https://arxiv.org/abs/2610.12056). These additions further weaken B as a new research direction. They do not establish that A is novel. A is a conditional mechanism question; C is presently a design discipline, not an established standalone contribution. The user seeks an explanation of LLM computation, so benchmark cleanup alone does not satisfy the scientific objective.

## VERIFIED — what I actually checked

“Verified” below means source/code inspection or saved-output recomputation in this session. It does not mean independently regenerated model predictions, verified historical execution authorization, or confirmed publication novelty.

### 1. ASTRA-ORDER-01 already has historical outputs

GitHub Astra head inspected: `c1d03ef1b452b6da59eae962710b41876337b06d`. Its [coordination record](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/c1d03ef1b452b6da59eae962710b41876337b06d/research/astra_coordination_2026-10-16/COORDINATION.md) says pending. The local directory `research/adapter_output_order_development_2026-10-16` contains six saved prediction files, protocol, inputs, code and results. These files were not in that inspected GitHub coordination record; there is no local commit history establishing their execution chronology. Preserve them as recovered historical development evidence, not a newly approved execution.

I wrote a separate standard-library raw-string parser in `audit_saved.py`, inspected the original runner, and recomputed the following without importing model libraries:

| Same 288 sentence comparisons per arm | Untouched | Ordinary adapter |
|---|---:|---:|
| Actor→Patient both correct | 177 | 215 |
| Patient→Actor both correct | 19 | 55 |
| Patient→Actor valid labeled fields | 53 | 108 |
| Patient→Actor obeys requested field order | 51 | 57 |
| Patient→Actor unlabeled two-name outputs | 235 | 178 |
| Compliant in both orders | 51 | 57 |
| Same name sequence despite reversed labels, among compliant pairs | 40/51 | 38/57 |
| Correct across both orders and all three constructions | 0/96 events | 2/96 events |

All 15 historical manifest hashes match. All 576 original-order token arrays exactly match the previous cached extraction arrays. The new parser disagrees with none of the 1,152 saved AP/PA parses. I also reran the independently authored previous scorer: 21 units, 2,016 comparisons verified; that older scorer uses parsed fields, not fresh raw parsing. Artifacts: `RECOUNT.json`, `PREVIOUS_RESCORE.json`, `audit_saved.py`.

Protocol SHA256: `95db44902dcbe169a2b6c29846ae2aeb2d849e208f23c6a19413c9085fc99d6b`. Frozen items: `b3771133fb2be6675c1f76728c058f5ae00cc8f49191340ed09be8c05b2d6012`. Original runner: `75f64aa86200fd4be19cec6239fe50759415001c14853004bc78e1d4405585f3`. Six output hashes are in RECOUNT.

**Interpretation:** most of the strict-score decline is accompanied by loss of labels. The small jointly compliant subset often preserves the emitted name sequence while reversing role assignments. That is compatible with serialization-dependent allocation, but conditioning on compliance selects a small, nonrandom subset. It cannot identify the behavior of excluded cases or a latent role representation. One ordinary seed and three dependent synthetic worlds are not cross-family confirmation. No need to repeat these calls.

### 2. Historical QA-SRL exposure is resolved for the selected panel

I located `llm_role_confirmation_v1/QASRL_ITEMS.json`, joined its 63 IDs to the saved candidates, and verified all 504 saved result records match their selected input metadata and protocol hash. No new upstream test corpus was opened.

| Partition | Selected source IDs | Prompt rows | Previously screened candidate IDs |
|---|---:|---:|---:|
| WikiNews dev | 28 | 224 | 45 |
| WikiNews test | 19 | 152 | 29 |
| Wikipedia dev | 5 | 40 | 10 |
| Wikipedia test | 11 | 88 | 16 |
| Total | 63 | 504 | 100 |

Thus **33 dev / 30 test selected sentences**, with **55 dev / 45 test candidate records** in the earlier inspected candidate pool. Exclude or explicitly flag the entire inspected candidate pool when defining future freshness; excluding only the 63 model-evaluated items would understate investigator exposure. These 100 records are a known lower bound, not proof that every other source record remains untouched. Candidate construction scanned four upstream partitions historically.

`HISTORICAL_EXPOSURE.json` contains original IDs, partitions, predicate indices and saved source-file hashes, but no sentences, questions, or answer spans. Selected-items SHA256 `21f197f07a0aafc1db4ab3faada610a0becbcfc5d0409571147cf088bbb5d254`; candidates SHA256 `61533aba900d98a7d97ff70bc08c4a6f0fec9f7a9ad8731b751170761364f888`; saved results SHA256 `608d0851da641f2462ca381fa72f7c18105bb04d458f0d560ef5bfd56b1b81e5`.

This resolves Nova's selected dev/test-count question. It does not establish annotation validity, source rights, pretraining contamination, source-document independence, or novelty. The old actor/patient conversion remains an AI-screened interpretation of human QA spans.

### 3. A simple counterexample independently replays

I inspected and ran Compass's unchanged [standard-library certificate](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/c8944ee479776c45bc6e3990faaf6242df4da5e2/research/compass/exp21/standalone_certificate.py), blob `b95f4afd4f3c1eded4368dea2ac6529ae173c745`. On its same 188 cases / 95 worlds, suffix-two execution gives selective interaction **67/188 = 35.64%**, and last-only lookup **48/188 = 25.53%**. Neither executes the full three-operation program. The complete interpreter scores 100%.

This is an independent execution of the deterministic control, not a replay of Compass neural checkpoints. It directly refutes the sufficiency of positive selective interaction for complete composition. Saved output: `COMPASS_REPLAY.json`.

## REPORTED — recovered identity and wider evidence

The [Astra findings at the inspected commit](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/c1d03ef1b452b6da59eae962710b41876337b06d/research/astra_snapshot_2026-10-15/FINDINGS.md) already withdrew the strongest early interpretations:

- The large ordered-role attenuation became **−6.10 pp**, interval **[−14.70,+2.49]**, at the selected aggregate ability match. I inspected the original local `role_transfer_phase01/ability_match_v1/RESULTS.md`: the aggregate gate passed but family-specific ability gaps remained. This is failed support for an ability-independent benefit, not proof that all supervision is useless. No model replay here.
- Qwen cleft accuracy was reported as untouched 86.49%, ordinary 51.51%, ordered auxiliary 75.53%, participant-only 83.82%. Auxiliary protection does not isolate ordered semantic supervision and does not beat untouched on that panel.
- The completed adapter diagnostic reverses ordinary/ordered rankings across interfaces; untouched extraction matches last-mentioned-actor on 227/288 outputs. The old raw reparses by Comput/Mira are reported independent scoring, not independent inference.
- The 12-event cross-family panel has severe dependence and heterogeneous model sizes/interfaces. Gemma's 1,728 correct responses are repeated variants, not 1,728 independent events. Thus universal role failure and pure scaling claims are unsupported.

[Brain-O E057–059](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/52b5582a01e0afe0f24b874d56bad5acf1203d04/research/brain-o/EXPERIMENTS_057_059_NATIVE_BRIDGE_2026-10-16.md) reports a source-sensitive internal bridge, but donor-target correctness only 180/600 and 219/600; no whole six-question world solved. I read the original report; its binary archive was not received or executed. The event-classifier prediction is a separate head, not an input to the native answer head. Accurate auxiliary classification therefore does not itself establish a correctly bound but unused native answer.

[Compass I21](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/c8944ee479776c45bc6e3990faaf6242df4da5e2/research/compass/INVESTIGATION_21_2026-10-16.md) reports 16.15 pp conditional interaction but only 116/1,128 fully correct quadruples. Neural estimates remain owner-reported here; only the deterministic counterexample was rerun.

Current [#31 decisions](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31) retain inference blocks. QA-SRL source/scorer work is not an executed two-family authentic-language result. Sentry/BX panels overlap and share annotator/source dependencies. [#32](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/32) has no verified eligible WikiEvents pair count in the records inspected. RAMS may lack two supervised focus events in one record. These are source-feasibility leads.

The three largest scientific weaknesses are **nonidentification** (several algorithms explain the same intervention), **construct/measurement failure** (format validity and QA spans are mistaken for semantic binding), and **selection/competence dependence** (small reused worlds, unequal ability, exposure and incomplete whole-event success). More prompt cells alone fix none of them.

## VERIFIED literature inspection; REPORTED paper findings

I read relevant method/limitation sections, not just titles. This is a targeted audit, not comprehensive novelty clearance or independent paper reproduction.

| Original source | Exact overlap / distinction relevant to our decision |
|---|---|
| [Dai et al., Cell-Based Representation, §3.2 and Appendix A.29](https://arxiv.org/html/2604.19052v1) | Entity/relation indices, PLS subspaces and query/final-token steering already exist. Re-DocRED is also analyzed, so “ours uses real text” alone is not a novelty gap. That appendix evaluates index decoding by contextual-similarity strata; it does not automatically establish our proposed event-error prediction. ACL PDF exceeded fetch size; inspected author arXiv v1, not a verified byte-identical proceedings version. |
| [Gur-Arieh et al., Mixing Mechanisms, §§2–4](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf) | Positional, lexical and reflexive retrieval are fitted jointly. The reported ~0.95 is Jensen–Shannon similarity of distributions, not 95% gold-answer accuracy. Their deliberately conflicting intervention predictions are a stronger reference than testing weak heuristics separately. |
| [Aljaafari et al., §3.1, limitations and Appendix G](https://aclanthology.org/2026.findings-acl.1964.pdf) | Role-conditioned next-token continuations use prepositional scaffolds and retain items correctly answered in both variants. This is narrower than arbitrary actor/patient QA, but already includes causal role circuitry, competence selection and lexical-pattern mechanisms. A general “role circuit” claim is occupied. |
| [Lepori et al., §§2–4](https://aclanthology.org/2026.acl-long.676.pdf) | Synthetic topology learning is separated from subsequent use, including instruction/prefill contrasts. “Encoded but unused” and delay/interface dependence are not our discovery. An auxiliary event classifier is weaker evidence of deployment than an analysis of the actual relevant representation. |
| [Bigoulaeva et al., §§5–6](https://arxiv.org/html/2602.07930v1) | Early instruction representations, nonlinear interactions and later pathway selection already investigated. A serialization hypothesis must distinguish prefix effects from instruction-conditioned processing. |
| [Gan et al., September 2026 preprint, §6 and Appendices F–G](https://arxiv.org/html/2609.02015v1) | Newly located high-risk collision: a four-by-four train/evaluate interface map across tasks/families, interface-dependent adaptation gains, generation-budget reversal, and failed geometry-based rescue/prediction. Merely adding Qwen roles or calling this “interface lock-in” is not a strong contribution. Paper claims are not independently reproduced. |
| [NAACL 2025 output-format study, §2](https://aclanthology.org/2025.naacl-long.15.pdf) and [EACL 2026 causal-format study](https://aclanthology.org/2026.findings-eacl.91.pdf) | Correctness versus compliance and causal-format ambiguity already have explicit treatment. An adjusted score alone is not a new framework. |
| [Geiger et al., causal abstraction](https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf) and [Sutter et al., §§4–7](https://arxiv.org/html/2507.08802v1) | Formal abstraction and the dangers of unconstrained alignments already exist. Sutter's powerful nonlinear alignment problem is not identical to our simple partial-program counterexample, but blocks a generic “patches do not prove algorithms” novelty claim. Geiger checked at primary abstract level; Sutter method/theorem discussion inspected. Méloux's cited identifiability paper remains a follow-up: its OpenReview page hit a browser challenge. |
| [TabEAE, §5.2](https://aclanthology.org/2023.acl-long.701.pdf) | Multi-event extraction, shared arguments and event semantic boundaries are established research topics. Our possible delta must be a new predictive error law/test on untouched models, not event co-occurrence itself. |

Chen and Hanna remain on Sentry's collision list; detailed independent full-text comparison was not completed in this session. None of these unchecked items is treated as cleared.

## PROPOSED — three serious directions

### A. Event-coherent errors versus fieldwise retrieval — primary scientific candidate

**Question.** When an otherwise competent model answers about the wrong event, are its participant errors coherent with one alternative event, or are fields reconstructed independently?

**Principle.** A restricted shared-selector model draws one event E then retrieves requested fields. A competing fieldwise model selects each field from syntax/lexical/position cues. This is a predictive decomposition, not an assertion that E is a native neuron variable.

**Motivation.** Brain-O separates event classification from correct participant answers; Astra's joint outputs often contain exactly the available participants; the project lacks reliable whole-event success. These do not establish either model but make joint error structure more informative than aggregate accuracy.

**Distinct prediction.** With two events and disjoint role fillers, a perfect-field shared selector predicts coherent wrong-event tuples, whereas conditionally independent field retrieval predicts their joint frequency from marginal errors. Explicitly, if B_r denotes borrowing role r from the same distractor, conditional independence predicts P(B_1 B_2|z)=P(B_1|z)P(B_2|z). A shared selector with perfect retrieval and wrong-event probability p instead gives covariance p(1−p). This elementary derivation is not a novel theorem. Imperfect retrieval, shared salience and autoregressive copying can also induce covariance; covariance alone does not select a mechanism.

**Closest work/gap.** TabEAE, CBR and Mixing Mechanisms; potential gap is held-out prediction of *which complete wrong event* supplies answers under controlled changes, beyond a joint cue mixture. Gap unverified.

**Strongest rival.** A competent compound-key copier or shared-salience retrieval mixture can duplicate the entire signature. Include it explicitly, with equal evidence and matched fitting flexibility. If it is observationally equivalent, merge the hypotheses rather than labeling it “nonsemantic.”

**Smallest discriminator.** First reuse frozen records to see whether any contain two fully gold events with disjoint fillers and the needed query/presentation cells. Do not infer new semantic gold from QA-SRL slots. If absent, Nova/Mira's existing #32 source-feasibility work decides eligibility. After approval only: one consolidated development panel, two untouched families, identical gold-supported questions in separate calls and both joint orders; fit restricted competitors on separate development documents and predict held-out joint error categories. Counterbalance source order, answer distance, lexical identity and query order where genuine gold supports it. A combined source-cue baseline is mandatory. Report full-span original-task scoring plus event identity only where annotated.

**Abandonment.** Stop the shared-event-specific thesis if the matched cue mixture predicts held-out joint errors and counterfactual target acquisitions equally well, if coherence exists only in joint generation, or if source eligibility/competence fails. Even success is a behavioral model comparison until native-path evidence excludes specified rivals.

### B. Serial answer commitment — narrow interface mechanism candidate

**Question.** Does generating an earlier field change a later semantic answer, beyond the instruction's static format effect?

**Principle.** A generated prefix becomes evidence for subsequent decoding. A first emitted participant may constrain the remaining answer through exclusion/copying, even if the original question is unchanged.

**Motivation.** Recovered compliant AP/PA responses preserve the name sequence in 40/51 and 38/57 cases. Most responses are not compliant, so this motivates a test rather than proving the process.

**Distinct prediction.** Holding the exact final query and requested schema fixed, a matched intervention on the first answer value changes the second answer in a way predicted by participant exclusion; an instruction-only account without prefix-mediated dependence does not predict that additional effect. A sufficiently flexible autoregressive model of course can: the hypothesis must specify a restricted quantitative policy.

**Closest work/gap.** Gan's interface adaptation, Bigoulaeva's instruction pathways, output-format studies; our own fixed-fields and answer-first/role-first studies already overlap heavily. A new prefix experiment is unjustified until #31 confirms the exact missing cell.

**Strongest rival.** Ordinary continuation statistics, valid-format selection and two-name complement copying. Prefix editing also introduces semantic information; it is not a clean intervention on an internal role variable.

**Smallest discriminator.** Audit existing fixed-field raw records for matched content/order cells and test a frozen complement predictor. Only if absent and approved: a three-or-more-candidate case with a gold-independent distractor, fixed second question, syntactically matched correct/wrong/unrelated prefixes, and standalone-query reference. No generic prompt sweep. Never call a gain under injected correct content a recovered latent capability.

**Abandonment.** Stop if effects are explained by the frozen complement/continuation baseline, disappear once validity is handled, or simply repeat the archived supplied-field experiment. This is lower priority because novelty overlap is severe and existing evidence mostly shows format collapse.

### C. PIVOT CANDIDATE: identify what an intervention can distinguish

**Question.** What is the smallest set of tests that separates a specified algorithm from competent alternatives, instead of merely showing sensitivity?

**Principle.** For candidate algorithms A and B and permitted test family T, define A~B iff their predicted observables agree for every t in T. A successful experiment can identify at most an equivalence class under this relation. For finite rivals, select witnesses where predictions differ; one test can eliminate multiple rivals. This is standard identification/experimental-design reasoning, not a new invention.

**Motivation.** Independently replayed Compass shortcuts satisfy the selective-interaction signature. Brain-O reports a copier reproducing crossed output interventions. The recurring problem may be the scientific test's resolution rather than a defective semantic representation.

**Distinct prediction.** A partial-program rival and full interpreter agree on many old cells but must diverge on cases where the omitted first operation changes the answer and later operations do not erase that difference. Formally choose x where B(A(x)) differs from B(x), while preserving the other matching constraints. This gives an actual witness rather than a larger average intervention effect. No new generated sweep was executed here.

**Closest work/gap.** Causal abstraction, Sutter, Méloux, causal scrubbing and Mixing Mechanisms. A general nonidentification paper would duplicate prior work. A possible contribution is a rigorously bounded, executable calibration of specific commonly used diagnostics against low-complexity rivals, with a useful witness-selection procedure; that novelty must be established separately.

**Strongest rival.** This may be ordinary unit testing/model discrimination repackaged. Also, increasing the rival class can make unique identification impossible.

**Smallest discriminator.** Reuse Compass's exact certificate and Brain-O's existing intervention tables; tabulate each rival's predicted outputs, list indistinguishable classes, and identify one missing witness. Recompute only existing evidence before asking for any new neural cell. Internal intervention comparisons must match access/site/architecture; a symbolic copier's exposed gate is not the same intervention as a Transformer residual patch.

**Abandonment.** Drop as a standalone paper if it yields no discriminating test beyond existing methods or merely restates known nonuniqueness. Retain it as a prerequisite for A. This genuinely challenges the project's central framing, but is not yet a stronger demonstrated publication direction.

## Mathematical checks on the reasoning

1. **No statistic recovers undefined role labels from unlabeled names.** Under an explicitly hypothetical completion of every invalid response, a correctness rate lies between c/n and (c+m)/n, where c is strict correct and m invalid. Base PA gives [19/288,254/288]; ordinary PA [55/288,235/288]. Their overlap prevents a robust latent-semantic ranking from these outputs alone. These are completion sensitivity bounds, not estimated internal competence. Strict operational success remains exactly observable.
2. **Gold acquisition and source sensitivity differ.** A patch can change the answer to another wrong answer. Report change probability, new-gold probability, wrong→correct transitions, and harmful correct→wrong transitions separately. Agreeing with an erroneous donor is not semantic success.
3. **A correct generic copier can implement relational computation.** If a semantic model and copier realize the same task function and all permitted interventions, no behavioral sample size separates them. Calling one semantic and the other nonsemantic does not create a testable contrast. Conversely, surviving finitely many rivals is not proof of uniqueness over all algorithms.

## Decision and explicit next gate

**Primary recommendation: modify the programme, do not declare a successful pivot.** Make predictive event-level error structure (A) the provisional scientific target, with C as the design requirement. Stop expanding Astra's synthetic order study. B remains a low-priority archival explanation, not a new training agenda.

A has the best potential connection to real language and a coherent measurable target, but depends on #32 eligibility. C is cheapest and directly supported by reusable counterexamples, but carries substantial prior-art risk. B has the strongest local behavioral motivation and least annotation dependence, but the weakest novelty position after the new literature collision. None currently warrants an ACL novelty claim.

**One immediate move:** request `ALREADY DONE — REUSE` review for recovered ASTRA-ORDER-01, hand the exact exposure manifest to Nova, and ask #31 for a single *reuse-only discriminability audit* of A against joint cue/compound-key rivals. Do not schedule new inference. Require one explicitly different predicted outcome before any future pilot. Preserve Nova's task-faithful QA-SRL pipeline; it supports competence/evaluation infrastructure but is not automatically the event-gold test.

**Future decisive falsification, conditional on approval:** fit the restricted shared-event and joint cue models on a separate development set; predeclare held-out error-category predictions and required native competence, then assess them once on eligible independently grouped documents in two untouched model families. The shared-event interpretation fails if the rival predicts the same held-out distribution within the predeclared practical tolerance, or if its apparent advantage is restricted to one output order/joint-call condition. Sample size and tolerance must be chosen from source feasibility and expected uncertainty before inference, not invented here without eligible counts.

**Stop/pivot conditions:** no source eligibility → stop A and preserve the negative result; no distinguishing witness → do not run; observed effects explained by existing priors → stop the mechanism claim; C adds no substantive method beyond prior art → use as internal QA, not a paper. A methodological paper becomes the main direction only if the audit produces a new, externally useful result with an exact prior-art gap. December 15 is a planning horizon, not evidence.

## UNKNOWN and limits

No new model inference, fine-tuning, paid call, sealed evaluation, new corpus download or restricted-data processing was performed. The historical run's asserted pre-inference lock cannot be independently timestamp-verified from local hashes alone. Brain-O/Compass neural checkpoints were not replayed. Human-role gold, source permissions and two-family event-collision performance remain unresolved. Full literature coverage and all independent branches were not exhaustively read. GitHub comments are handoffs, not acknowledgements.

An empty GitHub file-creation request was rejected by automatic approval review because it specified no repository, path or content and lacked explicit authorization for that mutation. No remote file was created; the report and audit artifacts are saved locally, and the separately user-authorized existing-issue reporting route is used for team communication.
