# Explaining Language Through Science — independent research audit
**Date:** 14 October 2026  
**Scope:** four `project-null-evidence-*.zip` archives, `PROJECT_FILE_INDEX.csv`, archive reports, implementation and protocol files, and selected raw prediction data.  
**Status:** independent **archive-level, computational spot-check and targeted reanalysis**, *not* reproduction of all model runs or a comprehensive peer review of all 5,078 included files. No model inference, training, or sealed-test evaluation was performed.

## Executive decision

**Most promising ACL-track hypothesis:** Query-conditioned semantic role extraction and answer selection can become misaligned, and the magnitude or direction of that misalignment depends on sentence construction, question wording, candidate support, and adaptation history. Our main research object should be the **interaction** between these factors, not a presumed single failed module.

**Best near-term step:** Allow the already-planned frozen/untouched-model diagnostic to finish on development-only items without changing its preregistered comparisons. Next, independently check semantic labels and build a question-conditioning-by-construction test with balanced source/patient order. Do not begin another large training sweep first.

**Principal caution:** Almost all recent output-level evidence is from adapted Qwen2.5-0.5B with automatically mapped QA-SRL actor/patient labels, and the strongest candidate-supported results collapse when the candidate line is absent. The clean mechanistic experiment in Qwen2.5-1.5B is informative but limited to active sentences and positional role confounds. No verified ACL-caliber natural-language repair yet.

## 1. Archive completeness and provenance

- `PROJECT_FILE_INDEX.csv` lists **7,410** indexed entries, with **5,078 included** and **2,332 excluded** under the combined category “model weights, binary checkpoint or archive.” The excluded category totals approximately **1.282 GB** as indexed by file size. Thus the source project was **not included byte-for-byte**; many high-value outputs and scripts *were* included.
- The four ZIP files contain **5,080 distinct paths**, including the file index and README; extraction succeeded, and **all 5,078 indexed included files are present with matching byte sizes**.
- Key archival documents previously missing from the smaller `research.zip` are now present: `RESEARCH_CHECKPOINT_2026-10-14.md`, `RESEARCH_FRONTIER.md`, `REPRODUCTION_GUIDE.md`, `role_transfer_phase01`, `phase3`, and `llm_role_*` branches.
- The historical `RESEARCH_ARTIFACT_MANIFEST_2026-10-14.json` lists **5,283** entries from its own snapshot. Its count differs from the export index, which is a distinct enumeration; do **not** treat this as evidence of a missing export. I checked manifest hashes for the research checkpoint, the ability-match result, and the construction-coverage report: these matched. The newest candidate-free outputs were not all listed in that older manifest.
- The export does not contain actual files with which to confirm the later **clean pretrained-model diagnostic** reported as just starting. No result from that future/in-progress run is incorporated here.

## 2. Reconstructed evidence ladder

| Line | Strongest result in archive | What it establishes | What it does not establish |
|---|---|---|---|
| Controlled support relocation / D5 | Prospectively bounded, isomorphic support-relocation effects in artificial relational listener | Training support location causally changes measured learned responses *in that world* | General LLM role mechanism or reusable grammar |
| Controlled rewire RQ2 | Sham/high graph contrast: role-aware attenuation **+22.59 percentage points** across eight fresh graph families | Large objective × graph-condition effect at fixed treatment in controlled task | A unique graph property or ability-independent benefit |
| Phase 2 high-weight | Attenuation **+22.57 pp**, 160 models, previously examined families | Development replication of absolute-scale effect | Ability-matched independence (other unseen performance ~10.7 pp better) |
| Matched-ability screen | Nonfocal ability **85.69% vs 85.01%**; focal attenuation **−6.10 pp** | High-weight graph attenuation does not generalize to the chosen aggregate-matched low weight | Proof of no objective effect at every weight |
| Natural Qwen continuation | QA-SRL Bank both-role direct exact **2→41/130** and **4→49/130** | Training can improve measured role answering on that recipe | Validated semantic actor/patient repair across natural forms |
| Construction intervention | Aligned→decoupled shifts passive joint accuracy up and active down in two lexical banks × two starts | Construction exposure causally redistributes generalization | An internal abstract role-circuit account or across-family repair |
| Typed goal + supplied candidates | Exact answers **43→73/80**, **44→70/80** on fresh3, matched token-length control | Tag can help route answers using fields under candidate-supported conditions | Candidate-free extraction or general authentic QA |
| Candidate-free typed goal | Both-role answers **11→31/100**, **13→21/100**, criterion failed | Strong first-start signal, weaker second-start signal; field/answer dissociation | Confirmed repeatable repair or stable independent event extraction |
| 1.5B hidden-state patch | Same-sentence selected active cases: actor-question donor **41/48** correct vs patient-question donor **0/48** | Changing the question-derived state after block 20 can causally steer answers in these cases | Semantic-role information disentangled from first-/last-person selection |

Source paths: `RESEARCH_CHECKPOINT_2026-10-14.md`; `role_transfer_phase01/{rq2_rewire_repair_v1,phase2_development_v1,ability_match_v1}/RESULTS.md`; `llm_role_natural_repair/RESULTS.md`; `research/{role_subject_balanced_followup_2026-10-12,typed_goal_token_control_2026-10-13,candidate_free_role_binding_2026-10-14}/RESULTS.md`; `llm_role_internal_v2/RESULTS.md`.

## 3. Targeted checks performed

1. Ran `research/paper_audit_2026-10-14/audit_cached_outputs.py` and `audit_common_pairs.py` against extracted raw predictions. Both reproduced their saved counts, including common-event tuple changes (**25/88 vs 40/88**, **19/83 vs 43/83**, neutral vs typed). These scripts write derived audit JSON to the *extracted copy*, not the user's source project.
2. Recounted all four 200-row `research/candidate_free_role_binding_2026-10-14/EVAL_*.json` files by event ID and requested role. Exact both-answers-correct counts: fresh1 **11→31/100**, fresh3 **13→21/100**, neutral→typed. Parser-complete counts: fresh1 **199→196/200**, fresh3 **198→191/200**.
3. Computed **new, post hoc** paired event transitions between the two arms, for the same 100 items in each start:

| Metric (out of 100) | Fresh1 neutral → typed | Fresh3 neutral → typed | Gains / losses in fresh1 | Gains / losses in fresh3 |
|---|---|---|---|---|
| Both answers correct | **11→31** | **13→21** | 24 / 4 | 15 / 7 |
| Printed role tuple *stable* between actor and patient question | **71→56** | **76→45** | 7 / 22 | 4 / 35 |
| Both final answers follow the *requested printed field* | **21→68** | **21→61** | 51 / 4 | 47 / 7 |

   Exploratory unadjusted paired exact-binomial/McNemar-style tests (descriptive; **not** prospective evidence): both-answer change *p*≈0.00018 (fresh1), 0.134 (fresh3); tuple-stability decline *p*≈0.0081 and <0.000001; both-answer-field-agreement gain *p*<0.000001 in both. These are the same inspected development items, not independent generalization, and no multiplicity correction was performed.
4. On the *common fully parsed and role-tag-correct subset* (88 and 83 events), reproduced the report's instability counts **25/88→40/88**, **19/83→43/83**. This restriction is important: all-event stability counts use slightly different coverage and should not be confused with the common-eligible comparison.
5. Re-ran `research/role_subject_balanced_followup_2026-10-12/cross_test_analysis.py`: across first and second lexical banks, passive gains on the same both-roles-correct metric were **+15, +28, +15, +18/36**, whereas active differences were **−3, −9, −5, −5/36**. This corroborates the existing report's key result, but the two banks reuse models/training starts and thus are not four independent experiments.

### A measurement warning from the raw predictions

Some events have both final answers correct even when the two reported role tuples disagree. This is not evidence for a stable internal representation (nor for its absence). A model can answer the requested field correctly while giving an inconsistent irrelevant field, or can answer from information not faithfully reflected in its fields. Accordingly, **do not use output-field consistency as a stand-in for correctness or hidden-state identity**.

## 4. What may have been overlooked or underweighted

### Opportunity A — The trade-off is *between accuracy, routing, and reported stability* (highest priority)

The archive already documents answer-field agreement increasing while field accuracy fails to improve. The new paired reanalysis sharpens this into a **three-way trade-off on the same sentence/event panel**: both-role answer performance, two-question tuple stability, and answer-to-field agreement need not move together. This is the most specific immediately testable phenomenon. Strongest next test: hold *sentence and target predicate fixed*, vary whether the requested role is visible **during extraction** versus revealed **after extraction**, and keep the answering information and total token budget comparable. Compare direct QA, role-first, deterministic field routing, equal-compute sham, and oracle/gold fields as diagnostics only. Keep human-checked actor/patient truth labels separate from grammatical proxies.

### Opportunity B — Candidate support is an enormous causal design factor (high priority)

In the token-matched typed-goal arm's **post hoc no-candidate stress test**, trained checkpoints went from **39/40 and 38/40** correct on comparable candidate-supported prompts to **13/40 and 12/40** without candidate lists (fresh1/fresh3). Printed fields were both correct in only **4/40** for either typed start, despite answer–field agreement at **39/40 and 38/40**. `research/typed_goal_token_control_2026-10-13/RESULTS.md`. This is a sharp boundary: options are not a harmless test convenience but change the participant-identification problem. Primary ACL evaluation must be candidate-free, with candidate-supported as a diagnostic control.

### Opportunity C — A surprising but limited mechanistic foothold *already exists* (high priority)

`llm_role_internal_v2/RESULTS.md` used an **untouched Qwen2.5-1.5B-Instruct** model and patched block-20 states between different questions on the **same sentence**. On 48 postselected active cases where the generic actor question was wrong and the performed-actor question right, patching from the performed-actor prompt gave **41/48** correct actor answers; a patient-question donor gave **0/48**. It is a genuine causal *question/state steering* result. But actor was always first-mentioned in the selected cases and patient always second. **Do not call this an abstract semantic-binding circuit.** Rebuild the intervention on balanced active/passive, actor-first/patient-first, role reversals and independently reviewed original sentences, with random/same-position/identity donor controls. This could give a scientific mechanism paper without starting from an untested representation claim.

### Opportunity D — Dataset/training construction interacts with question wording (high priority)

`research/role_prompt_transfer_audit_2026-10-12/RESULTS.md` finds **all-five-question joint** answers improve relative to natural source checkpoints, while *all-three actor phrasing* accuracy declines and actor contradictions increase in both starts. In the active “Who did the action?” cell, source **36/36** in each start degrades to mixed **23/36**, while the patient question and passive actor question often improve. This is a clearer example of **nonmonotonic adaptation** than pooled accuracy. It suggests a construction × requested-role × wording × training-arm test, but current panels reuse checkpoints and synthetic language.

### Opportunity E — Natural verifier corrected fields without fixing its answers (high priority)

`research/natural_role_verifier_development_2026-10-13/RESULTS.md` documents that under swapped drafts, the noisy-draft verifier preserved routed field accuracy **118→117/160** and **114→122/160** for the two starts, but its own answers remained **75/160 and 76/160**. The nominal *repair* screen failed. This independently motivates distinguishing role-field correction from answer use; it is a **visible-text/output pipeline result**, not an internal-state intervention. Beware that clean and noisy verifiers have an extra inference pass relative to direct QA.

### Opportunity F — The construction trade-off is arguably a stronger *alternative* paper result than the current speculative mechanism (medium priority)

In `research/role_subject_balanced_followup_2026-10-12/RESULTS.md`, a matched training construction substitution produces passive gains across both starts and disjoint test lexica with no passive losses on the inspected items, while active performance loses correctness without corresponding gains. The mixed-exposure arm only gains **+2 and +3/216** broad form-events relative to the better alternative and **fails** its preregistered +6 target. Crucially, the mixed arm scores **10 vs 12/36** on *all-six-forms-correct events* in both starts. A broad average can hide lower whole-event generalization. This is a good fallback paper if an independent natural-language continuation confirms the result across another pretrained family.

### Opportunity G — Old synthetic causal results fit the *theme* but not yet the LLM claim (medium/low for ACL main paper)

The D5 and Track B experimental notebooks already test how training support, cue reliability, and **which subnetwork can adapt** affect off-support behavior. In `phase3/post_p3_2_sol_discovery/research_memo.md`, learned-start branch surgery reports relation preference approximately **0.60–0.68** when the query encoder is frozen versus approximately **0.005** when the candidate encoder is frozen in that artificial task. The report also observes similar qualitative cue effects with a DeepSets candidate encoder, so Transformer-specific self-attention is **not necessary** for that outcome. These are valuable precursors to the idea of interacting subsystems. They are *not evidence* that frozen Qwen has the same mechanism or that a new Transformer-native composition law has been found. Earlier research program audits correctly rejected or parked several supposed general operation claims (`programme_reckoning_2026-10-04.md`; `research_reset_decision_2026-10-04.md`).

## 5. Uncomfortable methodological facts (keep them visible)

- **0/32** prepared active/passive challenge pairs and **0/75** EWT mappings have independent human-verified semantic-role adjudication according to `llm_role_challenge/STATUS.md` and the October 14 checkpoint. EWT syntactic subject/object labels are not guaranteed semantic agents/patients.
- The 100 candidate-free QA-SRL dev items are **actor-first** under the automatic mapping. This does not yet establish role knowledge independent of surface order. Parsing failures in the typed arms are higher than neutral arms, though modest in raw number.
- The typed-cue experiments use a **separate question-role classifier** to supply tags. It gets 191/200 tags right on the 100-event candidate-free panel. This is an extra task-relevant input signal, not evidence that the LLM itself inferred the correct semantic role from unrestricted language.
- Strong candidate-supported results rely on **known target participants** and often a known selector. Gold field routing or gold role labels supply information; they cannot be counted as ordinary unaided model performance.
- Distinct experimental windows changed arms, questions, construction, model ancestors, and sometimes prompts. Cross-study correlations are *hypothesis-generating*, not evidence of a single shared causal mechanism.
- The aggregate ability-matching gate (0.68 pp) concealed nonfocal family-level gaps of **−4.22 to +7.57 pp**. Current graph claims should not be called ability-independent.
- Frozen/preregistered controls, reused development sets, post hoc tests, and adaptations after observing results must remain separately labeled. Multiple samples/questions from the same event are not independent observations.

## 6. Ranked publication opportunities

1. **Best high-upside mechanism paper:** Question-conditioned *event-role extraction versus answer use*, conditioned by voice/order/wording, with **preregistered natural-label validity**, frozen baseline, question-timing factorial and balanced activation-patching tests. A causal and out-of-sample explanation would be the novel contribution; a targeted intervention is an optional stronger contribution.
2. **Best shorter-term empirical paper:** Under matched training budgets, **construction exposure causes opposing generalization shifts across role questions and syntactic forms**; show on independently validated natural sentences and a second model family, with both-role whole-event metrics and unadapted baseline. Stronger present causal design, but less conceptual novelty than #1.
3. **Complementary controlled-science study:** Training-support graph effects and their reversal under approximate ability-matching. Scientifically useful as an honest evaluation/interpretability lesson, but currently hard to tie cleanly to natural pretrained LLMs and weaker as the headline ACL claim.

## 7. Practical next experimental gate — do not modify Astra's active protocol

1. **Wait for Astra's frozen clean-baseline diagnostic.** Request: a compact table of each untouched/adapted model's exact actor/patient accuracy, both-correct per event, role-tuple stability and invalid output rate, broken down by voice, question and position. Retain the predeclared stopping rules and every negative result. If the clean model has no repeatable failure on human-valid data, pivot explicitly to **adaptation-induced regression**, not a universal LLM claim.
2. **Human-valid test:** independently adjudicate actors, patients and verb/event identity; separate semantic and grammatical labels; balance actor-first and patient-first, active and passive, reversible roles, omitted-candidate conditions and question phrasing. Keep reviewed development items separate from a sealed document/event-family-disjoint test.
3. **Minimal factorial:** same sentence and predicate; extraction conditioned on actor question vs patient question vs no question; answer condition with target revealed only later. Cross with active/passive and both event directions. Compare under equal compute, same answers available at test, same formatting constraints, and repeated-model starts. Fit a cluster-aware analysis with interactions and uncertainty; avoid treating a higher-order interaction as proof of neural-circuit interactions.
4. **Mechanistic follow-up:** reuse the block-20 same-sentence patch strategy only after balanced baseline and label validity pass. Add reversed-role and position/identity controls, causally test whether steering changes role-correctness rather than merely first/last-token choice. Don't rely on decodable probes alone.
5. **Repair only after identification:** compare against direct QA, ordinary SRL+router, extraction-separated pipeline, token/compute-matched sham, simple augmentation and generic consistency baselines. Score **joint correctness** and **worst-condition regression**, not just average accuracy or self-consistency.

## 8. Literature check, targeted, not exhaustive

- Wilson, Petty & Frank (2023), *How Abstract Is Linguistic Generalization in Large Language Models?* https://aclanthology.org/2023.tacl-1.78/ — argument structure and linear-order bias. Not novel to report that LLMs generalize imperfectly from active to passive.
- Hejabi et al. (ACL 2026), *Flip-Flop Consistency* https://aclanthology.org/2026.acl-long.71/ — prompt variation robustness and consistency training. Mere “prompt invariance loss” would not be distinctive.
- Ravulapalli et al. (Sep 2026 preprint), *Legible Failures* https://arxiv.org/abs/2609.11216 — recoverable bindings from hidden states and probe-guided steering in a different task. Mere recoverability/steering would not be new.
- Pyatkin et al. (EMNLP 2021), *Asking It All* https://aclanthology.org/2021.emnlp-main.108/ — roles and question-generation decomposition are established. The novelty should be the specific causal relation between *event-role extraction, query conditions, and linguistic constructions* backed by independent testing, not the existence of a role-first pipeline.

**Novelty verdict:** An ACL-level result is plausible but **not currently achieved**. The strongest scientific surprise in these results is not one successful loss—it is that **accuracy, role-field stability, and answer routing can diverge**, and their relationship depends on the conditions of the task. That is the precise finding worth trying to reproduce and explain.

## 9. Reproducibility caveats

Computationally verified in this review: archive inventory and file sizes; saved cached-output audit results; all four candidate-free raw output recounts; paired-event reanalysis and unadjusted exact test calculations; construction cross-test analysis output; three selected historical checksum matches. Taken from existing reports but **not independently regenerated by model inference**: all graph training outcomes, natural-repair training outcomes, hidden activation-patching outputs, sealed-test status and other experimental branch summaries. The archive excludes saved weight/checkpoint binaries, so full training/inference reproduction from these uploads alone is not assured.