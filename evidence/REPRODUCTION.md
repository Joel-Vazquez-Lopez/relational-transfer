# Reproduction and audit instructions

This GitHub foundation is an indexed summary. To independently test the numerical claims:

1. Obtain the authorized `COMPASS_RESEARCH_FOUNDATION_2026-10-15.zip` from the research lead. Do not assume repository access also grants access to this external archive.
2. Verify the ZIP SHA-256 against `SOURCE_PROVENANCE.md`; extract into an isolated workspace, **not** into this Git checkout.
3. Run `python3 VERIFY_PACKAGE.py` at the extracted archive root. It checks every non-manifest file's SHA-256 and verifies selected headline metrics directly from included JSON/JSONL.
4. Recompute the target claim independently from raw predictions, using the original `source/` experiment protocol and prompts. Record event-level disagreements, sample clusters, input normalization, and whether any oracle labels entered the evaluation.
5. Keep independently coded replications and falsification baselines on separate research branches. Do not modify frozen source files.
6. Record exact model ID/revision, quantization, chat template, decoding, source hashes, seeds, costs, and hardware before any new inference.
7. Publish reviewable source, output summaries, and machine-readable derivative results through a PR. Leave sealed confirmation sets and restricted datasets outside the shared repository.

The original archive's reproduction guide states that raw metrics can be reanalyzed, but re-executing model inference requires retrieving the pinned public checkpoints. Several historical adapter weights are absent, and hosted Gemma serving may not be reproducible byte-for-byte. CPU-only environments may be adequate for analysis but not major pretrained-model inference.

A passed archive-integrity check establishes file consistency, **not** scientific correctness of source analyses.
