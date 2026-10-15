# Reviewed artifact handoffs — 16 October 2026

**Research lead request:** make the existing evidence actually findable and independently checkable, not merely announced. This is a **draft private-repository review handoff**, not publication, external redistribution, or a frozen-test release.

## Discovery in the research lead's ChatGPT File Library

The exact files below were found at the **Library root** and materialized to an isolated workspace for checksum/ZIP inspection. The research lead's Library is separate from GitHub; **a Library listing does not prove another ChatGPT research session has access**. Each receiver must search their own authorized Library for the filename, retrieve bytes and report a local SHA-256 match. Request an explicit approved transfer through Ops [#22](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/22) if access is unavailable.

| Handoff | Exact Library filename | Verified archive SHA-256 | Inspected | GitHub access now |
| --- | --- | --- | --- | --- |
| **H1 Mira V4 → Comput/Nova** | `MIRA_V3_FALSIFICATION_AND_MULTISTRUCTURE_V4_2026-10-15.zip` | `50451ea957fb5e81a03ea7b637e34fd976906e9968186683f5118cd4c66b58d7` | ZIP CRC pass; internal SHA manifest 18/18 | Original remains in Library; **not yet receiver-confirmed** |
| Mira's independent audit (supplement) | `MIRA_DATASET_EVALUATION_2026-10-15.zip` | `5f687f1ba2baf1f1b087b3d23957816b9d40e6dbd692e65ad821a1b9024be2be` | ZIP CRC pass | Library only |
| **H2 Brain-O E037–038 → reviewers** | `BRAIN_O_POSITION_AND_FACTORIAL_037_038_2026-10-15.zip` | `14a6cdaa2b2c7403b00fb6fadb46eb553d1c25abb5985a6d6a59f865b8b7cf86` | ZIP CRC pass; internal manifest 13/13 | Source and checkpoints remain in Library; **not receiver-confirmed** |
| **H3 Nova CAL-01 Qwen 1.5B+7B → Comput/Mira** | `nova004-cal-qwen7b-results.zip` | `dceb5086e27cf87c855e7f738407dabcff815c60a94f63ad50f7028373e6f17c` | ZIP CRC pass; raw predictions 24 + 24 | Raw synthetic predictions copied byte-for-byte to this draft branch in `nova_cal01/` (verified source ZIP Git-blob hashes); **recipient still to replay** |
| Nova earlier 1.5B-only archive (duplicate evidence) | `nova004-cal-results.zip` | `dbc73d30e4a455df1704e18f117312cf9d181c5fb983590b103ced7094ac19f0` | ZIP CRC pass | Library only; later combined ZIP is preferred |

All five ZIPs were actually found and checked; no inferred or fabricated local path. The original source archive hashes match those previously reported in GitHub research notes.

## What reviewers must do

### H1 — Mira V4

Original ZIP includes exact development generator `build_v4.py`, `model_inputs_v4.jsonl`, `gold_v4.jsonl`, `challenge_v4.jsonl`, baseline predictions, and verification scripts. **2,592 correlated synthetic QA rows in 60 families**, not 2,592 independent tests. Its README explicitly warns that several source-predicate-blind strategies perform well.

For evaluated LLM inputs use **only `passage` and `question`** from `model_inputs_v4.jsonl`. Do not serialize `id` or `split` into the model prompt; ID encodes generator/world bits. Keep gold and `generator_edges` privileged and separate. `structure_holdout` is openly inspected development data, not sealed confirmation. Before sharing beyond authorized research agents, verify generator provenance and license/consent of every included source.

**Comput:** reply in [#22](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/22) with `H1 RECEIVED / SHA-256 VERIFIED` or `BLOCKED: cannot access Library file`. Then independently rerun the original benchmark's shortcut audit. **Nova:** audit data-interchange schema; avoid label leakage.

### H2 — Brain-O E037–038

ZIP includes two compact **trained synthetic PyTorch checkpoints** (`.pt`), frozen source and scripts, E037 correction, E038 factorial results, replay logs and SHA manifest. Original checkpoint grammar has fixed small name and verb vocabulary; don't evaluate unseen suffixed names from the protocol-auditor PR #20 on these weights as if they were in vocabulary. **Do not upload checkpoints to GitHub by default**, forward outside authorized reviewers, or use `torch.load` on an untrusted archive. Inspect code/hashes and isolate execution first.

**Brain-O / Compass / Mira:** verify archive SHA, exact tokenizer, weights and scripts through an authorized access path. Post `H2 RECEIVED + reproduced outcome / mismatch` with source revision. This is synthetic, not pretrained native relational mechanism evidence.

### H3 — Nova CAL-01

Two **lossless raw prediction** JSONL files now committed here for reviewer comparison:
- [Qwen2.5-1.5B `qwen15_predictions.jsonl`](nova_cal01/qwen15_predictions.jsonl), 24 generated strings; source member SHA-256 `20511d7cc8210c8d4eb3d31fb76636a08f0b5dd3e188960010819081b627054a`.
- [Qwen2.5-7B `qwen7b_predictions.jsonl`](nova_cal01/qwen7b_predictions.jsonl), 24 generated strings; source member SHA-256 `104a6b28655445c3bac59c3c7b22027b8a4c2e6c8870d2e0c3ed9b6d65a0df2f`.

Model checkpoint weights are **not** included. Original ZIP manifests embed an absolute personal local model directory; those manifests are deliberately **not copied verbatim**. Key provenance: dataset SHA `12bf01a370b434067781f0057e204b379c9452b050b7feafcd9f9c4edf0d303d`, MLX-LM 0.32.0, temperature 0, max output tokens 12, deterministic Qwen Instruct 4-bit checkpoints, 24 prompts/model on **one synthetic world**. Original calibration prompt generator is `NOVA-004-CAL-v0.1.zip` in Library, requiring a separate authorized transfer/hash check for exact prompt replay. Rows here preserve originals; reviewer must independently verify individual member hashes against source ZIP, input/prompt contract, and format vs semantic accuracy.

**Comput / Mira:** reply `H3 RECEIVED + raw-row recount` or identify missing data. Do not interpret these 48 dependent generations as model-family generalization or a causal mechanism.

## Reproducibility checks

On a trusted local machine with the authorized original ZIPs:

```bash
python3 evidence/reviewer_handoffs/verify_intake.py \
  --mira-v4 /path/to/MIRA_V3_FALSIFICATION_AND_MULTISTRUCTURE_V4_2026-10-15.zip \
  --brain-e038 /path/to/BRAIN_O_POSITION_AND_FACTORIAL_037_038_2026-10-15.zip \
  --nova-cal /path/to/nova004-cal-qwen7b-results.zip
```

This **does not** run models, load pickle/checkpoint files, open sealed evaluations or send text to external services; it checks archive integrity, expected record IDs, paired source membership and hashes. Output is a local audit only.

## Merge/release gate

- Do **not** claim H1/H2 complete until a receiver confirms access and repeats verification.
- Do **not** merge this draft or distribute weights/privileged gold externally without lead and independent scientific review.
- Keep synthetic developmental artifacts distinguishable from authentic human-reviewed language and untouched confirmatory data.
- No new paid inference, large GPU run or sealed-data access is authorized by this branch.
