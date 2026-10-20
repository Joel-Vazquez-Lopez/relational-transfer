# D6 final report — discriminative demand under matched exposure

## Outcome

**Mechanical preregistered category: D6-INVALID.** All 12 models completed exactly 8,000 updates and all frozen evaluations. The pretraining audit, checkpoint integrity, familiar reliability, candidate permutation, object-slot permutation, and parameter-state gates passed. The neutral-message gate failed, so the confirmatory D6 causal result is invalid by the frozen decision rule.

## Frozen identity and execution state

- Preregistration SHA-256: `ed5423c7779699181bbd3fda6fe476eafc9a69bd5b52ed5f1479e32e85ec2657`
- Implementation SHA-256: `019e7de6a40bcb5ca7af8f3990171a830751ff6b3ddf61d39932cb761fce4b3c`
- Pretraining audit: **PASS** (`model_initialized=false`; 830.303 seconds; every critical check true)
- Models: **12/12 COMPLETE**, each with 8,000 updates and 1,024,000 examples
- Valid final checkpoints: **12/12**, with hashes matching both status and metrics records
- D1–D5: unchanged

## Validity gates

### Observation

- Familiar accuracy: REQUIRED 99.6528%–99.7892%; REDUNDANT 99.6776%–99.9132%. Both reliability gates passed 6/6.
- Candidate permutation: 8,064/8,064 inverse-mapped predictions agreed for every model.
- Object-slot permutation: every model exceeded 99.8% example-level agreement (threshold 99%).
- Evaluation parameter hashes: unchanged for every model.
- Neutral-message accuracies ranged from 0.271701 to 0.498512. **0/12** models passed the conjunctive frozen gate (point in [0.20,0.30] and bootstrap interval containing 0.25).

### Inference

The neutral-message failure mechanically makes the experiment D6-INVALID. The confirmatory geometry and behavior contrasts below cannot be promoted to a valid causal conclusion under the frozen protocol.

### Speculation

The structured four-candidate sets may carry exploitable candidate-only/event-composition information even when the oracle message is replaced by the all-zero message. D6 did not preregister a diagnostic that can distinguish that possibility from a poorly calibrated neutral-message control, so no stronger explanation is claimed.

## Frozen primary demand-context results (descriptive because invalid)

| Metric | Six paired effects (1801–1806) | Mean | Median | SD | dz | 95% paired-seed bootstrap CI | Raw p | Holm p | Frozen metric pass |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Geometry `I_demand` | 1.283680, 1.215030, 1.353690, 1.681911, 1.659435, 1.560817 | 1.459094 | 1.457254 | 0.200788 | 7.266841 | [1.315727, 1.603279] | 0.015625 | 0.031250 | True |
| Behavior `I_behavior` | 0.232391, 0.289931, 0.369544, 0.451141, 0.432912, 0.443204 | 0.369854 | 0.401228 | 0.090827 | 4.072050 | [0.299207, 0.431858] | 0.015625 | 0.031250 | True |

Both descriptive primary contrasts meet their frozen metric-level thresholds (.20 geometry; .10 behavior), direction-consistency rule (6/6 positive), CI rule, and Holm rule. They do not override D6-INVALID.

## Frozen held-out transfer results (descriptive because invalid)

| Metric | Six paired effects (1801–1806) | Mean | Median | SD | dz | 95% paired-seed bootstrap CI | Raw p | Holm p | Frozen metric pass |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Transfer geometry | 0.193888, 0.271330, 0.417068, 0.595100, 0.607155, 0.623189 | 0.451288 | 0.506084 | 0.186728 | 2.416819 | [0.312385, 0.578809] | 0.015625 | 0.031250 | True |
| Transfer behavior | 0.122024, 0.219990, 0.317584, 0.336558, 0.368056, 0.362103 | 0.287719 | 0.327071 | 0.097269 | 2.957976 | [0.211020, 0.351418] | 0.015625 | 0.031250 | True |

Both transfer contrasts also meet their metric-level frozen rules, but the invalidity gate prevents a D6-A assignment.

## Exposure and manipulation audit

### Observation

- 48/48 direct demand-manipulation rows passed REQUIRED targeted-endpoint necessity and REDUNDANT targeted-endpoint redundancy.
- 192/192 symbolic condition/cell/event rows passed candidate correctness, unique candidate, identical object inventory, two-same/two-opposite event, targeted-foil, context, and style checks.
- All six paired schedules had exactly 1,024,000 examples: 2,666 complete 384-item epochs plus one 256-item residual.
- Every structural residual cell appeared in exactly four of six seeds.
- For every seed, every preregistered query/candidate/position/identity/event/context exposure counter was exactly equal across conditions; intended cue assignment differed.
- All rendered-pool and evaluation-pool checks passed; native evaluation rendered-scene multisets were equal; train/evaluation renderer key spaces were disjoint.
- Maximum full-schedule object-slot relative range by seed: 0.2100%, 0.0878%, 0.1707%, 0.3434%, 0.2730%, 0.1663%, all below 1%.

### Inference

The pretraining manipulation and exposure match were implemented as preregistered. The experiment's invalidity arises after training from the neutral-message control, not from a failed exposure or demand-construction audit.

## Directional secondaries and full matrices

The predeclared source-role, target-role, pooled redirection, pairwise ordering, substituted margin, and margin-change values are in `predeclared_directional_secondaries.csv`. The complete condition × seed × stratum × directed substitution × partner × role × event table is in `partner_substitution_matrix_by_model.csv`; its seed-mean companion is `partner_substitution_matrix_seed_mean.csv`. These are descriptive and do not redefine the primary result.

## Operational event and deviations

### Observation

A chat/runtime interruption occurred during the live D6 execution. On-disk verification found REQUIRED seeds 1801 and 1802 complete and seed 1803 active with status `UPDATED_DO_NOT_RESTART` and no final checkpoint. The original process was still live and continued; it was not restarted. It subsequently completed seed 1803 and all remaining runs. The earlier chat-level “Invalid prompt” message was not a scientific deviation.

### Inference

The interruption affected conversation state only. It did not interrupt the training process, alter any model state, duplicate a completed run, change the protocol, or require checkpoint recovery.

Deviations from preregistration: **none**. Operational restarts: **none**. Exploratory rescue runs: **none**.

## Relation to D4 and D5

### Observation

D5 validly showed that relocating relational support relocated final-query geometry and behavior. D6's descriptive contrasts point in the preregistered discriminative-demand direction in all seeds, including held-out partners, but D6 failed its neutral-message validity rule.

### Inference

D6 does not supply valid confirmatory evidence that discriminative demand, rather than matched exposure, caused the observed difference. D5 remains unchanged and its conclusion is neither rescinded nor strengthened by an invalid D6.

### Speculation

The descriptive pattern is consistent with the working hypothesis, but interpreting it causally would require resolving why the neutral-message gate fails without using D6 outcomes to redesign or rescue this experiment.

## What D6 establishes and does not establish

It establishes that the frozen construction achieved its pretraining exposure/demand manipulation, all 12 models and evaluations completed, and the preregistered validity logic assigns D6-INVALID. It does **not** establish a causal discriminative-demand effect, systematic identity binding, a mechanism, symbolic variables, linguistic generality, or an LLM result.

## Reproducibility artifacts

- `configs/d6_preregistered.json`: frozen machine preregistration
- `outputs/d6_preregistration.md`: frozen human preregistration
- `src/d6_demand.py`: frozen scientific implementation
- `results/d6_demand/pretraining_audit/`: complete pretraining audit and six schedules
- `results/d6_demand/runs/`: 12 checkpoints, traces, predictions, query vectors, redirection rows, and candidate blocks
- `results/d6_demand/metrics.json`: frozen aggregate decision record
- `outputs/d6/`: report and inspection tables

## Exactly one next experiment — NOT RUN

**D7 candidate-only predictability diagnostic (NOT RUN):** Freeze a separate diagnostic that trains and evaluates a candidate-only baseline on D6's fixed candidate sets, with condition/seed pairing and held-out nuisance geometries, to determine whether candidate configuration alone explains the neutral-message above-chance performance. This directly addresses the failed gate without modifying, rescuing, or rerunning D6.
