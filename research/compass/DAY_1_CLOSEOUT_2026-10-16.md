# Compass — Day 1 Scientific Closeout
**Date:** 2026-10-16, shortly after the 2026-10-15 research session.  
**Milestone:** **DAY 1 COMPLETED** — independent Compass scientific investigation.  
**Branch:** `research/compass-reuse`. **Status:** exploratory development, *not* a confirmed novel LLM mechanism, natural-language repair, or publishable final result.

## The objective we retain
Determine whether Transformers naturally compute and reuse relational structure, reconstruct it when queried, or mix the two; distinguish the **actual native computation** from probe-accessible information, output repairs, and simple shortcuts. Reject apparent successes when matched alternative mechanisms reproduce them. The human researcher makes scientific and release decisions.

## What was accomplished
Compass Investigations 01–15 developed reproducible *synthetic* controls, audits, learned model experiments and falsifiers. Do **not** treat the number of investigations as independent scientific discoveries.

### Major methodological results, with their limits
1. **Benchmark identifiability:** Earlier actor/patient templates can be solved by position-based lookup. Correct QA and Qwen/Gemma differences do not establish an abstract event graph.
2. **Cache vs reconstruction:** Transparent and trained schedules can have identical normal answers and whole-prefix patch effects. Known-route interventions recover **architecturally exposed** dependencies, not spontaneous native algorithms.
3. **Probe caution:** Externally fitted probes can infer perfect relational answers even from raw embeddings/one-hot operation inputs. Decodability cannot certify that the base model performed composition.
4. **Latent edit caution:** Fitted latent transformations can transfer without obeying all group laws; a random-feature decoder can memorize original answers. Structural intervention and native operator use differ.
5. **Information-allocation controls:** Exact one-bit code enumeration produces a predictive-accuracy versus guaranteed-reuse tradeoff. A neural gate may miss an attainable optimum; an externally searched/label-supervised code repairs it. These are engineered finite-memory results, not an observed natural pretrained reuse policy.
6. **Native patch caution:** A layer-wise prefix→query causal shift and third-answer hybrid effects also appear on matched copying/modular-arithmetic tasks; they are not relation-specific signatures.
7. **Compositionality challenge:** Shorter swap programs give partial length transfer but incomplete-composition heuristics explain much of the apparent gain. Output permutation repair adds coherence externally rather than proving a coherent internal state.

### Highest-priority Day 1 empirical clue — Investigation 15
A **non-bijective** directed-relationship environment removes the earlier one-to-one permutation shortcut. Ordinary unrestricted causal Transformers trained on one/two active relation operations reach **9,154–9,213/9,216** held-out shorter-question answers across seeds 19/29/41, but only **1,728–1,814/4,096** held-out three-step answers (42.2–44.3%), below a deterministic omit-first shortcut (**2,154/4,096 = 52.6%**).

Across matched training trajectories from 200→600 updates, three-step accuracy **declines 7.69, 8.47 and 8.40 percentage points**, while short-task accuracy improves. On consequential one-cell world edits, final trained answers change for **82–93%** of first-step edits but only **13–18%** of third-step edits; irrelevant-cell edits change **4.5–7.5%**. A second independent-from-training, unsealed 128-world development panel shows the same qualitative pattern (though it remains synthetic). These are **input-output causal sensitivities, not located internal circuits**.

Competing explanations not rejected: shortcut fitting; optimization dynamics; finite model depth; encoding/position effects; changed calibration; adaptive choice of test after seeing development results; synthetically narrow two-function grammar. No verified emergence of a general relation-composition operator or pretrained LLM mechanism.

## End-of-day source verification
- Read current `main` workstream governance and latest relevant open draft PR metadata before closeout.
- Read and compared cross-team developments without editing their branches:
  - Coordinator PR #17 demonstrates a competent source shortcut and reconstruction can share valid-world route signatures.
  - Protocol auditor PR #20 shows same-query stimuli can retain shortcut solutions.
  - Mira PR #25 offers 26,496 *open-development* synthetic items in 48 families, explicitly 100%-solvable by a full parser and awaiting independent review.
  - Reviewer handoff PR #24 provides an artifact manifest and raw Nova CAL rows, while several binary artifacts remain separately permission-gated. Their availability is **not** independent experiment replication.
  - Brain-X PR #21 records a negative parity-constraint neural pilot and invites further adversarial review.
- Reopened Investigation 15's local report and reproducibility README. Recomputed ZIP SHA-256 **`46f3b7361f40d612c6e6f93d710684f445a40b6706b504ea1c908caa6de8b474`**; ZIP CRC passed on **75 members**.
- Executed `python -m unittest -v test_exp15`: **4/4 tests passed** (gold computation, counterfactual invariance, non-bijective representation and label balance). **These tests do not independently rerun all trained checkpoints or certify reported inference counts.**
- Full 5.4MB artifact: `COMPASS_INVESTIGATION_15_NONBIJECTIVE_DEPENDENCY_2026-10-16.zip` in the Compass conversation; **not stored in GitHub**. See `research/compass/INVESTIGATION_15_2026-10-16.md`.

## Day 2 priority — one decisive prospective experiment
**Investigation 16: can intermediate relational computation be made robust to additional composition steps, and does the resulting network use the intermediate on its native path?**

**Before new training:** freeze an independent function/world generator and split at **world-structure / graph-family** granularity; test no-op and positional confounds, repeated participants, lexical variability, held-out 3–5-step programs and counterfactual edits of every operation. Define entire-world accuracy and per-step edit-sensitivity as **co-primary** metrics. Report independent worlds, not raw correlated question counts.

**Matched arms:**
A. Ordinary short-chain answer-only supervision.
B. Genuine stepwise intermediate-state supervision.
C. Extra-token / extra-loss / training-budget control without correct intermediate values.
D. Optional shuffled-intermediate supervision and directly engineered compositional positive control for sensitivity calibration.

**Decisive gate:** B must improve unseen *entire-world* outcomes and restore dependence on late consequential edits **relative to A/C**, with preserved irrelevant-edit invariance, across multiple seeds and a second held-out family. Then test pre-question native causal use versus question-time recomputation using known-mechanism controls and source-access confounds. Positive result is not enough if an external intermediate oracle or a shortcut explains it.

**Stopping rule:** if evidence collapses to positional/template heuristics, or intermediate supervision improves individual accuracy without world coherence and step sensitivity, do not assert structural reuse. Reconsider the environment and broader hypothesis.

## Research integrity and delivery
Day 1 is **complete**, not the research project. No sealed confirmation data accessed; no paid external inference; no historical artifact or other investigator branch altered; no main merge. A genuinely important result would need independent replication, authentic language, pretrained model testing and a prior-art novelty check before an ACL claim.

**Compass scientific conclusion:** The strongest surviving lead is *training-associated loss of functional dependence on later relational operations while fitting shorter ones*. It is a falsifiable behavioral regularity in this small synthetic setting—not yet the discovery we ultimately seek.
