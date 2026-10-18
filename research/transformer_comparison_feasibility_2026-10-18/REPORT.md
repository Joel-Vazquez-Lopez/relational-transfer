# Public release feasibility and saved-failure audit

Two completed tasks:

1. Audited whether the released Congruity trained-bank experiment can be reproduced from the inspected public checkpoints.
2. Localized the preceding learned-network transfer failure using saved outputs only.

## Published trained bank: prerequisite not met

Pinned Congruity code:0bdc4f6ff45c9e8319d2450bcd63b082b4e167e9. Its modified InterpBench gitlink c1841ad7dd892e8b6b116d74d2b675d7519e1d74 is accessible, and its training code can be inspected. The bank used by `left/algo.py` is separate: weights/configurations/correspondences in four variant directories. `left/scripts/download.sh` points to an institutional SCP source. No attempt was made to connect to that private server.

Pinned original public model repository `cybershiptrooper/InterpBench` revision a1242a84f8a4d07d1be9a8f5ec710415198016b2 has436 inventory items, no pagination. Of nine required task IDs, only63 and124 have public weights/configs. Zero of the six released algorithm chains is complete. Presence alone would also not establish equivalence to the modified trained bank.

Checked rehost `mib-bench/interpbench`, pinned revision6ce0d290e72e13019f3cc88dc6b1ac3673725655, contains IOI checkpoints rather than the needed permutation components. The checked Congruity GitHub release list is empty; checked model searches by project/name and the matching author handle returned no model entries. These bounded searches do not prove global artifact nonavailability. Full pinned inventories and exact source URLs/hashes are preserved.

A rebuild from the released code is possible in principle, but not equivalent to a quick checkpoint evaluation. `train_model` requests up to120,000 examples per component; the released shell script requests30epochs, with early stopping potentially reducing work. The parsed `--num-samples` option is not passed into the visible training-data call, so it cannot be assumed to bound that call. No runtime cost estimate or training-performance claim is made. We did not start that larger reconstruction, install the full InterpBench stack, or silently use unrelated checkpoints.

**Status: published trained-bank comparison NOT RUN.** This is an artifact/resource prerequisite outcome, not negative evidence about Congruity or our scientific hypothesis.

A separate frozen compiled-reference alternative was executed and documented in `../reference_transformer_comparison_2026-10-18/REPORT.md`. It uses official Tracr and two public routine ideas, a custom restricted nuisance sampler, and the unchanged released practical comparison functions. Its limited result is explicitly not a trained-model reproduction or theoretical Algorithm1 implementation.

## Saved learned-network failure: preserved and localized

No new model inference, training, intervention, predictor fitting or threshold adjustment was performed in this audit. Source artifact: learned-intervention commit037d708a23290b4cbb7af7d5e84a441c08aab8bf, raw SHA2569c0a29f9161abea0b5b2d6f423f40bab5bacde50cd01e9e86a546d54bf709afd.

| Transfer | Original RMSE | Stage1 fraction of squared error | Stage2 | Stage3 |
|---|---:|---:|---:|---:|
| 11→37 | .05442 | 40.08% | 41.18% | 18.73% |
| 23→37 | .14698 | **95.17%** | 3.27% | 1.57% |

The second transfer still fails its frozen RMSE≤.1 gate. We saved48 source×stage×grammar-cell summaries and1,536 individual case-stage-dose records. Questions/option orders repeat neural feature inputs, so these are not independent linguistic replications. Localization does not by itself distinguish a genuine learned-computation difference from perturbation mismatch.

This audit is exploratory, because those held outcomes were already observed. Any proposed remedy needs a new prospective test. We did not rescue the original result by dropping stage1 or changing scoring.

## Resources and reproducibility

- `audit_release.py` reproduces the availability check from pinned inventories, offline.
- `diagnose_saved_transfer.py` reproduces the failed-transfer localization from the existing saved NPZ; no network or model execution.
- `SOURCE_RECEIPT.json` records exact upstream locations, hashes and modified fork commit.
- `PUBLIC_ASSET_AUDIT.json` records every component and chain; inventories retain their original HTTP status and pagination information.
- `UNBLOCKING_MANIFEST.json` specifies the minimum resource types for a later trained-bank reproduction. No external artifact-request message was sent.
- Compiler setup logs document pinned libraries in the local `runtime/` directory, not changes to the shared environment. No pretrained checkpoint was downloaded. That runtime and the unmodified upstream source cache are excluded from Git; the reference study's setup script reconstructs them by pinned URL/version/hash.

No paid compute, external annotation APIs or extra pretrained inference were used. Source setup repairs and their scientific boundaries are recorded in the reference study's `PREFLIGHT_REPAIRS.md`.
