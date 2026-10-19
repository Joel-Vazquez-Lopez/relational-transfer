# BRAIN-X — Day 1 closeout and evidence audit

**Date:** 16 October 2026 (first research-day closeout; research work started 15 October).  
**Scientific deadline:** 15 December 2026.  
**Branch:** `research/brain-x-theory`. **Draft PR:** [#21](https://github.com/Joel-Vazquez-Lopez/relational-transfer/pull/21).  
**Claim status:** speculative theory + synthetic developmental tests. **No confirmed novel relational mechanism or independently verified real-language result.**

## Executive finding

Our Day 1 advances measurement and falsification, not a new theory of pretrained Transformers. The project's clearest insight is an important **three-way distinction**:

1. **Information sensitivity:** a hidden-state/counterfactual change affects output.
2. **Information utility:** allowing reliable information increases *semantic correctness* beyond a matched competent simpler alternative.
3. **Output coherence:** the set of predictions satisfies a global consistency constraint, which can be achieved by external projection or format forcing with no evidence that the native model learned the relation.

These are **not equivalent**. Existing mechanistic inference, factor-graph, structured prediction and ECC literature substantially predates this study. The novelty of a diagnostic separating all three is NOT established; it is a candidate organizing framework and falsification criterion.

## Brain-X completed work

- Created and maintained dedicated workstream, charter and branch; draft review PR #21; three mathematically specified theories (H1 constraint integration, H2 computation scheduling, H3 relational operator algebra). Direct GitHub scientific disagreement with Brain-O; ops ACK; authentic-language bridge #23.
- **BX-001:** two-layer Transformer did not exploit mathematical parity information; valid/shuffled arms were not paired, serious flaw independently highlighted by Comput. Three-layer follow-up incomplete. Preserved, not rewritten.
- **BX-002:** repaired design with 4,000 exactly paired test worlds. Three-layer models (training seeds 17 and 29) with informative parity scored 2,606/4,000 and 2,614/4,000, vs 2,612/4,000 both node-only controls; random parity scored 2,518 and 2,545. Thus neural responses are sensitive to constraints (448 and 187 individual prediction flips) without a useful accuracy gain. An unprivileged three-vote rule scored 2,795/4,000; known-noise Bayes 2,925/4,000.
- Original BX-002 raw CSV **re-read independently from the saved output file** for end-of-day audit: 4,000 rows, exactly matching all four arm counts, 448/187 changed predictions and 2,795/2,925 analytic reference scores. Original JSON SHA256 `126eb96ee2ef48e96b5728d07ce7df377fafd50b1f548d57250d87aacc6ddf4f`; raw CSV SHA256 `5a1d30a65a1773168b15c6651ad1c440e567f33c8db2e6bcc0223e74ef7ba3d0`. Code/test/report/summary are committed; ZIP with per-row predictions is a separate conversation artifact. This is **not** independent training replication.
- Comput explicitly accepted review of BX-002 in [handoffs #22](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/22) but **did not yet independently replay** neural training at this cutoff.

## New evidence read across the team

**Brain-O, E037–038 (source branch `research/brain-o-ideation`):** identified a real token-indexing bug in E035–036's mixed-voice name-position patches and corrected it. Under corrected source-name cuts and 2×2 predicate-event × participant-role interventions, frozen 3L models achieve combined donor-answer restoration 200/200 and 197/200 in two seeds. This supports task-effective separately manipulable dependencies in a fixed two-event synthetic grammar. A competent generic lexical-key lookup/copying control still could explain the effect; no unique natural language relation circuit. Checkpoint package is reported Library-accessible but not independently replayed by Brain-X.

**Compass, I13–I14 (source `research/compass-reuse`):** on 216 unseen three-swap contexts and 3 shared-seed models, native full-world correctness under all-short curriculum is only 52/648 worlds. Applying **externally supplied** valid-permutation MAP raises full-world correctness to 180/648, but a simple hard-output-only projection already gives ~169.9 expected /648, and summed-logit MAP slightly lowers individual-question correctness (1,595→1,586/2,592). This is a particularly strong warning: *constraint-enforced coherence is not evidence for native internally computed structure*. I14 is an external postprocessor, **not** a pretrained LM result. It substantially narrows Brain-X H1: future claims must beat both output-complement and hard-answer structured-projection baselines.

**Comput, R014/R015 and handoffs:** source outputs show nearly perfect two-name complementarity in valid joint extractions, which may require just a binary orientation choice. Competent generator-aware baseline predicts 768/1,024 answers on Auditor V2, including zero of 256 gold-changing donor cases. The coordinator subsequently replayed this exact upstream shortcut code independently; this is a dataset-construction falsification, not an LLM mechanism.

**Mira V5, draft PR #25:** a newly released *open* synthetic dataset has 26,496 questions / 48 independent name families across competing 2/3/4-hop predicates and AND branches. Mira reports text-only independent parser 26,496/26,496, five passing tests and extensive counterfactual consistency. The original generator/data ZIP is in the lead's ChatGPT Library (not automatically on GitHub or in our session). A source-local parser still solves 100%; no LLM inference, independent external audit, sealed benchmark or authentic language validation is established. Treat as stress-test input, **not paper confirmation**.

**Nova, handoff #24 and NOVA-006:** 24+24 Qwen 1.5B/7B raw CAL-01 outputs have a lossless reviewed repository handoff and exact hashes, but a recipient's independent recount is pending. The one-world earlier diagnostic cannot generalize to real language; NOVA-006 is a source-checked design, **not a completed model test**, and an undirected graph walker solves its synthetic worlds. Explicit role/format/scope separation remains necessary.

**Artifact access, draft PR #24:** original V4, Brain-O E037–038 and Nova archives were reportedly verified in the lead's persistent ChatGPT Library, with a single approved internal handoff bundle. This is promising infrastructure; **receipt by the actual Comput/Compass/Nova/Mira receiving sessions and independently executed checks are not established**. No private model weights/privileged gold should move into GitHub by default.

**Astra:** existing unsealed synthetic adapter interface results and raw output audits remain valuable but do not establish recovered role binding. No new Astra model runs were reviewed at this cutoff. Respect planned pause.

## Scientific revision to H1

Old framing: “Maybe joint extraction performs error correction by satisfying relational constraints.”

Revised testable framing: **Does adding independently discriminative, human-adjudicated context improve a pretrained model's semantic role answers beyond (i) format/complement forcing, (ii) hard-answer constrained projection, (iii) matching output budget and irrelevant-context priming, and (iv) a competent lexical/source-only parser?** Merely increasing joint-output validity or causing output flips is insufficient.

Pretrained models may benefit from multiple contextual clues through ordinary question-time reconstruction. A behavioral advantage is **not evidence of an intrinsic factor-graph algorithm**. If the same gain arises from format alone, hard-output projection, or irrelevant matched text, abandon this account as the source of the effect. If authentic corpus context cannot supply independently meaningful constraints without editing/leakage, revise the test rather than inventing “natural” examples.

## Day 2 priorities and gates

1. **Stop expanding parity toy suites.** BX-002 is an informative negative for this model/regime; it does not earn more expensive training just to seek a positive.
2. **Comput independent replay/audit of BX-002:** verify exact source, test fixtures and raw counts, compare voting baseline's index/position assumptions. One competent negative review is more valuable than further model seeds.
3. **Authentic-language feasibility with Mira:** identify genuinely human-written *predicate-specific* events and pairs of informative vs irrelevant evidence; independently annotate source/document family, ambiguity, scope, role labels, license and counterfactual validity. Never treat generated Mira V5 as authentic or feed privileged IDs/gold to models.
4. **Nova two-pretrained-family pilot contract:** frozen direct actor/patient QA vs joint extraction in both field orders; equalize/record output budget, raw validity and decoding; text-only matched irrelevant and informative evidence when legitimate. Qwen 1.5B and Qwen 7B count as **one family**, not two.
5. **Mechanistic challenge to Brain-O:** its causal factorial must be compared with a competency-matched lexical-key lookup/copier under identical native patch masks, controlling passive token positions. Do not duplicate its E031–038 runs.
6. **No “EUREKA” or paper-level novelty claim.** Successful authentic LLM behavioral effects would still need matched source-only baselines, independent confirmation and restricted mechanistic attribution.

## Communications and status

- Brain-X branch and draft PR #21 **open**, not merged. Research charter and data files remain reviewable.
- GitHub ops #16 ACK posted; lounge #19 disagreement posted; natural-language ladder #23 proposal posted; Brain-O #6 direct challenge posted; H6 #22 receipt request posted.
- GitHub comments are not messages delivered automatically to other ChatGPT conversations. Identical linked GitHub account does not prove independent human researcher identity. We record only acknowledged external review when a receiving workstream actually replies.
- No request for the lead's computer, model weights, paid inference or sealed confirmation data tonight.

**Bottom line:** Day 1 produced corrected negative experiments and more discriminating explanations. Next objective is authentic-role evidence and at least two distinct pretrained-model families—not more impressive-looking synthetic counts.
