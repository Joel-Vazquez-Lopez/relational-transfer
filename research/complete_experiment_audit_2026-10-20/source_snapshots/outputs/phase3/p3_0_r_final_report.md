# P3.0-R — Controlled Response Atlas Acquisition

**READY FOR P3.1.** This is an acquisition and measurement-validity result only. P3.1-F was a valid archive-insufficiency stop; the new atlas repairs the missing exact state–action–response join for controlled input/scene interventions. It does not test latent factors or any mechanistic hypothesis.

## Policy and frozen design

Only A/B/C/D × seeds 1701–1703 discovery checkpoints were loaded, with checkpoint/atlas SHA-256 verification. Protected seeds 1704–1706 were not opened or hashed. Existing P2.1 exact scene tensors and D5 oracle-channel encoders were reused; no task network was retrained or modified. The D8 instrumentation and P2.1 three-stage extraction informed the read-only trace. The pre-network inventory, preregistration, base plan, schemas, code hash, and crossing audit are under `phase3/p3_0_r/`.

The base selector chose 128 P2.1 blocks/model, 16 per evaluation-kind×family stratum, using only metadata and fixed hashes. Total 1,536 bases. For each base, all 10 branches independently restart from the same canonical input: BASE, atomic I/E/C/R/X, and crossed IE/IC/EC/IEC. Support topology was retained as an acquisition condition, not treated as an inference-time action. The query input is an exact oracle message; the P2.1 query scene is preserved as provenance but is not consumed by the frozen D5 model. The action definitions and fidelity rules are in `intervention_schema.json`.

## State, trajectory, and output capture

Every branch NPZ contains its exact float32 oracle message, all four 3×14 candidate tensors, three token-state stages for query (3×5×48) and candidates (4×3×4×48), final normalized 48-dimensional vectors, raw four-score vector, softmax probabilities, and chosen position. BASE states and outputs are stored separately and linked by offsets; index rows include both scene hashes and all provenance. A transformed query with no exact candidate has undefined correctness, not a fabricated wrong label. This is a rich observed state, not a complete resumable computational state. Shards use lossless NPZ compression and no downcasting.

## Network-integrity and archived-replay gates

Across 12 deterministic hook-free/traced pilots, maximum direct-versus-traced score difference was 1.90734863e-06 (frozen tolerance 1e-5); repeated traced state arrays were bitwise identical. For 1,536 archived BASE four-score rows, 19 matched bitwise, 1517 matched only within tolerance, and 0 mismatched; maximum score difference was 3.81469727e-06. All 15,360 static/runtime action fidelity checks passed. Loaded model state hashes were asserted unchanged after each shard.

## Final coverage and joinability

Final: 12 models, 1,536 base states, 9 nonbase actions/base, 15,360 complete state×branch records, and 6,144 crossed branch records. The 10 action families each have 1,536 records. Every index scene hash replays from its shard; every crossed action has its atomic siblings linked by the same base ID. Raw shard size totals 77,923,237 bytes. The planned acquisition used 15,360 query and 61,440 candidate encoder-item passes; the 12 eight-record integrity pilots added 288 query and 1,152 candidate encoder-item passes (total 78,240 encoder-item passes, or 15,648 paired episode-item evaluations).

| Condition | Seed | Bases | Records | Crossed | Context triples | Identities | Events | Kinds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_original_clique_hole | 1701 | 128 | 1280 | 512 | 19 | 6 | 2 | 4 |
| A_original_clique_hole | 1702 | 128 | 1280 | 512 | 19 | 6 | 2 | 4 |
| A_original_clique_hole | 1703 | 128 | 1280 | 512 | 19 | 6 | 2 | 4 |
| B_relocated_clique_hole | 1701 | 128 | 1280 | 512 | 20 | 6 | 2 | 4 |
| B_relocated_clique_hole | 1702 | 128 | 1280 | 512 | 20 | 6 | 2 | 4 |
| B_relocated_clique_hole | 1703 | 128 | 1280 | 512 | 20 | 6 | 2 | 4 |
| C_degree_balanced_matching_holes | 1701 | 128 | 1280 | 512 | 20 | 6 | 2 | 4 |
| C_degree_balanced_matching_holes | 1702 | 128 | 1280 | 512 | 20 | 6 | 2 | 4 |
| C_degree_balanced_matching_holes | 1703 | 128 | 1280 | 512 | 20 | 6 | 2 | 4 |
| D_path_connectivity_swap | 1701 | 128 | 1280 | 512 | 19 | 6 | 2 | 4 |
| D_path_connectivity_swap | 1702 | 128 | 1280 | 512 | 19 | 6 | 2 | 4 |
| D_path_connectivity_swap | 1703 | 128 | 1280 | 512 | 19 | 6 | 2 | 4 |

Task-native identity×event×configuration coverage has 48 aggregate cells, of which 36 are populated. The 12 absent focal-kind cells are structural in the full P2.1 archive: every empty selected model-level cell was also absent before selection (0 selection-induced empty cells). One-, two-, and three-factor matrices are in `final_factor_coverage.csv`; model-level counts and missing flags are in `final_crossing_coverage.csv`, with atomic/crossed totals in `final_intervention_coverage.csv`. Whole-context, configuration, identity, event, two-factor, triple-action, and cross-seed holdouts are feasible under the predeclared readiness rules. No response values were inspected for scientific patterns.

## Deviations, hashes, and claim boundary

A pre-forward implementation tightening added an exact geometry and single-distractor-identity fidelity check for the composed C+X case; the base/action definitions and thresholds did not change. No post-forward code or design repair was required. `technical_repairs.md` records both hashes. All atlas, index, QC, code, config, and report artifacts are SHA-256 listed in the reproducibility manifest. No prior D5/D8/Phase-II/P3.1 artifact was modified.

The Step 0 readiness verdict is **READY FOR P3.1** for this controlled-input action library. It says a provenance-safe response atlas now exists. It does not establish machine-native factors, conditional computation, interactions, redundancy, synergy, compositionality, causal latent organization, or any P3.1 scientific result. No P3.1 model was fitted and P3.2 was not designed.
