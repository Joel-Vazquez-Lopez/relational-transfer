# Preregisterable MSc pilot protocol

**Question:** Can a randomized post-training history change later learning on *independently generated* task families in a way not predicted by the checkpoint's starting competence and simple exploration proxies? This is narrower than claiming a universal plasticity trait. It extends the SFT→RL question already studied in [L1–L4](01_literature_map.md).

## Design and estimands

Use one exact open-weight **base** checkpoint for all branches (candidate: Qwen2.5-0.5B base; check license, tokenizer and training compatibility locally before commitment). A 1.5B model is an optional scale check, not needed for the minimum pilot. Fix initial weights, data-generator versions, prompts, answer verifier, context length and evaluation decoder. Randomize **history seed**, data order and branch assignment. Histories:

1. **Control:** base with no history; and a compute-matched continued-SFT control on neutral/mixed examples.
2. **SFT trajectory:** same target prompt family, supervised solutions; save checkpoints at 0, 25, 50, 100% of a fixed token budget, with an intentionally overtrained checkpoint only if affordable.
3. **RLVR trajectory:** same target prompt IDs and verifier, on-policy grouped rollouts (GRPO or a simpler REINFORCE-with-baseline if software is unstable); save checkpoints at matched *generated-token* and update budgets. Report both budgets because they cannot be simultaneously equal to SFT exposure. Do not call them data-identical.
4. **Optional DPO:** pairwise preferences derived from the same fixed rollout pool. This isolates an offline preference objective imperfectly; DPO adds no on-policy exploration. Add only after SFT/RL branches work.

**Primary randomized estimand:** branch difference in absolute held-out learning-curve AUC across the prespecified two target families, averaging over history seeds. **Secondary:** late-budget accuracy, held-out log loss, old-task retention, task×branch and update-rule×branch interactions. Baseline-matched comparisons are descriptive, because checkpoint performance is downstream of treatment.

## Task construction and leakage controls

- History tasks: procedurally generated small symbolic reasoning problems, e.g. arithmetic expression evaluation under a fixed grammar, with exact executable answers. Vary surface vocabulary to avoid memorizing one token template. Keep solution lengths short enough for controlled RL sampling.
- Before freezing the generator, calibrate difficulty on a separate development pool so the base solves roughly 10–60% of prompts within the fixed rollout group; otherwise RLVR groups may have no reward variance. Freeze the generator and verifier before comparing histories. If no difficulty band provides usable rewards, stop the RL branch rather than silently changing its task distribution.
- Future family A: **new compositional operator grammar** (e.g. stack-machine instruction sequences with withheld instruction compositions). Future family B: **independent finite-state/string transformation rules**. Generate both from code after finalizing history recipes; keep generators and seeds sealed until checkpoint selection is frozen. Within each family, split at the rule/template level, then generate disjoint train/validation/test instances.
- Distinguish *unseen generated instances and rule combinations* from impossible-to-certify “unseen concepts” in opaque pretraining. Run exact and normalized near-duplicate checks against all project-generated corpora. Record any overlap with public benchmarks, but public benchmarks are descriptive only.
- Use fixed data ladders (e.g. 16, 64, 256, 1,024 examples per future family), one held-out test set per family (at least 400 items where feasible), and no test feedback during tuning. Evaluate old history task and a small general language/control set before and after adaptation for forgetting.

## Adaptation and baseline controls

- At least two update configurations: **full-weight fine-tuning** and **LoRA** (fixed rank/modules); each uses a small preregistered learning-rate grid chosen on validation tasks, with the same grid for every history. Fresh optimizer state is primary; carried-state is an optional artifact check.
- For each starting checkpoint, record item-level correctness, log loss, Brier score/calibration, difficulty-bin scores, output length, entropy, pass@1/16/64 at fixed generation-token budget, and per-item solution-class coverage. Build a baseline profile; report standardized mean differences and item-level agreement (Cohen's κ or Jaccard of error sets). Do not pair two checkpoints solely on mean accuracy.
- Record history tokens, generated tokens, update steps, wall time, optimizer/scheduler, KL to base on fixed probes, layer-normalized `||θ−θ₀||/||θ₀||`, and output entropy. Parameter norm is not an invariant measure of functional distance; report it as a descriptor.
- Adaptation curves must use *identical* training item order and evaluation prompts within paired seeds. Reset adapters and optimizer between ladders. Measure at 0, 16, 64, 256 and 1,024 examples, preferably with intermediate steps to separate initial slope from attainable performance.

## Analysis, uncertainty and stopping rules

Start with **3 independent history seeds** as a feasibility screen; a confirmatory result requires **5 or more independent history seeds** if variance warrants it. Use 3 downstream-order seeds per history seed if feasible, but do not count them as independent history replicates. Randomization inference or a hierarchical model with history seed as unit is preferable to item-only bootstrapping. Show paired seed dots and 95% intervals, plus item-stratified uncertainty. Correct the two primary family tests by a prespecified Holm procedure or designate one primary and one replication family.

Set a practical threshold before training: e.g. **5 percentage points absolute test accuracy** at 256 examples or **10% relative AUC** at equal adaptation examples. These are design thresholds, not prior effect estimates. After the first 3 history seeds, estimate variance and simulate power for the intended paired design; if a 5-seed study cannot distinguish the threshold from zero, scale down the claim or stop. An equivalence conclusion requires the confidence interval to fit inside ±threshold; a nonsignificant difference alone is inconclusive.

Compare held-out prediction of adaptation AUC using (i) current aggregate score, (ii) task-wise/item profile, (iii) validation loss and high-`k` coverage (L1), (iv) profile plus entropy/drift, and (v) those features plus history label. Split by complete history seed and, for transfer claims, hold out a whole task family. Report rank stability across LoRA/full FT; strong rank reversals reject a universal plasticity index. Exploratory PCA/gradient geometry is allowed only after the primary test.

**Stop conditions:** no stable verifiable reward; repeated zero-variance RL groups; no overlapping baseline profile for the secondary matched analysis; >2× measured compute budget; inability to run ≥3 history seeds; or pilot intervals so wide that the confirmatory design exceeds resources. If a robust effect is entirely explained by known L1 proxies, report a replication and do not claim new mechanism. If there is no effect within informative equivalence bounds, reject the target claim.

## Compute and schedule

These are **planning ranges, not benchmarked measurements**. Run a 50-step SFT, 50-step RL and 100-generation throughput/memory benchmark on the actual cluster, then replace ranges using `GPU-hours = measured hours per unit × planned units` plus 30–50% margin. Fix max prompt/response lengths (e.g. 256/256), rollout group 4, batch and precision in the manifest. RL generation is often the bottleneck; the exact GPU and software stack can change cost substantially.

| Scope | Work | Planning envelope | Outcome even if stopped |
|---|---|---|---|
| **Phase 1, low compute, 3–5 weeks** | 0.5B base; 3 SFT seeds and 3 short RLVR seeds, 3 checkpoints per path; two generated task families; LoRA adaptation plus one full-FT sanity pair. | Roughly **80–250 H100-equivalent GPU-hours** across training, rollouts and adaptation, plus storage for checkpoints/logs; actual requirement must be measured. One 80GB GPU or smaller GPUs with memory checks. | A preregistered causal feasibility/replication study with learning curves, leakage controls and an honest power estimate. Useful without Phase 2. |
| **Phase 2, ambitious, 6–10 more weeks** | 5 history seeds, full FT and LoRA for every selected checkpoint; optional DPO; 1.5B scale replication if resources remain. | Roughly **400–1,200 H100-equivalent GPU-hours**; 1–2 modern 80GB GPUs for 1.5B full FT/RL may be needed, contingent on measured memory/throughput. | Cross-procedure and cross-scale inference. |

No MSc claim should depend on Phase 2. Maintain a versioned manifest, exact training/evaluation scripts, checkpoint hashes and a sealed holdout list. If AI Sweden has only a 24GB GPU, reduce to 0.5B, short sequence lengths and adapter-only history, and explicitly limit conclusions to adapter training; do not silently label those full-weight RL results.

## Mechanism ladder and intervention gate

After a repeatable primary effect: (1) cheapest and most informative—entropy/high-`k` coverage, item-level loss and gradient cosine on fixed mini-batches; (2) representation CKA/effective rank with fixed token/layer probes and negative controls; (3) Hessian trace or sharpness estimates only on a small selected checkpoint subset; (4) activation/weight patching or randomized entropy restoration for a causal mechanism. Each costs more and has more interpretation traps. A one-shot correlation is not causal evidence. A new intervention phase is justified only if it outperforms simple replay, early stopping, base anchoring, and the SFT→RL preparation methods in L2–L4 under equal compute. Otherwise stop at characterization.
