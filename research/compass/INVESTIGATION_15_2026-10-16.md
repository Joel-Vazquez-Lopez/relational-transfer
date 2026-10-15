# Compass Investigation 15 — Non-bijective relational composition and input-dependency specialization
**Date:** 2026-10-16 · **Branch:** `research/compass-reuse` · **Status:** synthetic development finding, not a native LLM mechanism or verified ACL novelty.

## Context and protocol
Before running: inspected updated main `research/WORKSTREAMS.md` and coordinator PRs #15/#17, protocol auditor #20, and Brain-X #21. Those workstreams strengthen shortcut and causal-identifiability controls. No other investigator's branch was edited.

Investigation 14 found that output-permutation constraints repair apparent whole-world coherence. This study removes that confound: input worlds contain **two independent, generally non-bijective directed functions** A,B on 4 entities, with repeated targets allowed. The model sees the complete eight-cell source table, an initial entity, and a 3-position A/B/NOOP program. Output is the final entity after composition.

Unrestricted 3-block causal Transformer, width 48, no hardcoded composition operator, route gate or event cache. 12,000 unique training worlds, each disjoint from 128 evaluation worlds and a second unused 128-world development panel. Training samples only **one- and two-step** programs, with interleaved NOOPs to ensure every operation position can hold A or B. Evaluation: all eight length-three operation chains × all four starts × 128 worlds = **4,096 answers** per panel. 1,024 world/chain conditions, **998 non-bijective** final maps on first panel. Three seeds (19,29,41), 600 updates, AdamW LR .003, batch 128. All labels, splits, pseudorandom generators and predictions are preserved in the standalone artifact.

**Preserved design correction:** an initial training setup always placed NOOP in the third position. It scored 67.3% on the interleaved short test and 35.5% on triples (seed 19), but its failures cannot isolate composition; retained only as a positional-confound negative control.

## 600-step held-out generalization
| Seed | Held-out one/two-hop correct (out of 9216) | Three-hop correct (out of 4096) | Entire correct 4-start chains (out of 1024) | Entire correct 32-answer worlds (out of 128) |
|---|---:|---:|---:|---:|
| 19 | 9211 | 1728 (42.19%) | 198 | 2 |
| 29 | 9154 | 1756 (42.87%) | 193 | 3 |
| 41 | 9213 | 1814 (44.29%) | 197 | 0 |

**Hard shortcut:** a deterministic program that discards the *first* operation and executes only the final two answers **2154/4096 (52.59%)**. An additional pairs-only curriculum (training only 2-op queries) achieves 1729,1821,1980/4096 for seeds 19,29,41. Correctness alone does not identify their internal algorithms.

**Second disjoint development panel:** Three-hop scores **1896,1932,1979 /4096** (46.3–48.3%); the same omit-first shortcut scores **~57.7%**. One/two-step scores 9207,9169,9213/9216. Whole-world successes only 1,0,2/128. This is unsealed corroboration, not independent sealed confirmation.

## Nonmonotonic transfer during training
Compare the *same deterministic training trajectories* at 200 and 600 steps for the one/two-step curriculum:
| Seed | 200-step 3-hop /4096 | 600-step /4096 | 200-step short /9216 | 600-step short /9216 |
|---|---:|---:|---:|---:|
| 19 | 2043 | 1728 | 5617 | 9211 |
| 29 | 2103 | 1756 | 5312 | 9154 |
| 41 | 2158 | 1814 | 5940 | 9213 |

Three-hop scores fall **7.69, 8.47, 8.40 percentage points** while short accuracy rises to ~99–100%. Descriptive *world-clustered paired bootstrap* 95% intervals on late-minus-early are [-11.35,-4.05], [-11.77,-5.03], [-11.40,-5.13] percentage points. This effect was discovered on development data and needs prospective replication; intervals do not account for experiment selection or model-family variability.

## Counterfactual step-dependence — the key mechanistic clue
For each three-hop query, identify which source-cell is used at step 1/2/3. Only use a position-specific edit if that source cell is unique along the original path and the edit changes the **gold final answer**. Separately change irrelevant (never-touched) cells; gold must remain invariant. This is an **input-intervention sensitivity measure**, not a hidden-state circuit test.

At 600 steps, the answer changes in response to consequential edits with these rates:
| Seed | Step 1 | Step 2 | Step 3 | Irrelevant-cell edit |
|---|---:|---:|---:|---:|
| 19 | 93.4% | 66.8% | 13.1% | 4.5% |
| 29 | 86.4% | 72.9% | 14.6% | 5.9% |
| 41 | 82.1% | 57.5% | 18.1% | 7.5% |

Step-specific n = 2876/3100/2978; irrelevant n=4096 per checkpoint. A second disjoint world panel confirms ordering: first-step **81–93%**, third-step **15–20%**. At 200 steps, first-step sensitivity is 49–58%, third-step 28–36%, and irrelevant 17–19%. These changes track the short-task fitting process but **do not prove a causal internal operator, cached state or intrinsic reuse strategy**.

The more conservative explanation is training specialization for <=2 active operations, with less use of the final one. Note that a learned heuristic may produce similar responses, and the empirical design was developed adaptively.

## Scientific decision
No general relational composition has been demonstrated, and no pretrained LLM was examined. The interesting, replicable-on-development signal is a **training-related redistribution of behavioral causal input dependence across program steps**, together with degradation on longer programs. This remains close to known function-composition and length-generalization literature (Tang, Lake & Jazayeri 2026; Awasthi & Gupta 2023; Petty et al. NAACL 2024). Novelty is **unverified**.

Next: freeze a new independent graph family and preregister intermediate-state supervision versus a matched extra-token control, require three-to-five step *whole-world* correctness and stepwise counterfactual sensitivity, and only then investigate whether native hidden computations are precomputed or question-conditioned.

## Reproducibility and provenance
Verified chat artifact: `COMPASS_INVESTIGATION_15_NONBIJECTIVE_DEPENDENCY_2026-10-16.zip` · **SHA-256 `46f3b7361f40d612c6e6f93d710684f445a40b6706b504ea1c908caa6de8b474`** (5.4 MB). Contains **75 files**, including `exp15.py`, `test_exp15.py`, checkpoint state dictionaries, both development panels, raw answer/logit records, positional design failure, world-clustered bootstrap, counterfactual response audits, and file hashes. The ZIP is delivered in the Compass conversation, **not hosted in GitHub**. No sealed examples, paid/external model API, frozen source changes, cross-workstream edits or main merge.