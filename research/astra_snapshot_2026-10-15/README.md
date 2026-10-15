# Astra research snapshot — 15 October 2026

This snapshot complements the repository's curated `evidence/` foundation. It preserves historical source code, protocols, analysis summaries and reports without changing their scientific status. **Archival reports are not independently reproduced results.** See `MANIFEST.json` for byte-level SHA-256 provenance. Original directory relationships are retained under `archive/`.

## Reading order

1. [Findings and decisions](FINDINGS.md).
2. [Historical checkpoint](archive/RESEARCH_CHECKPOINT_2026-10-14.md).
3. [Research frontier](archive/RESEARCH_FRONTIER.md) and [reproduction guide](archive/REPRODUCTION_GUIDE.md).
4. [Completed cross-family diagnostic](archive/research/llama_gemma_cross_family_2026-10-15/RESULTS.md).
5. [Current technical preflight](archive/research/adapter_readout_diagnostic_2026-10-15/PREFLIGHT.md) and its proposed protocol.

## Scope and reproducibility limitations

529 files were selected from the controlled role-transfer programme, LLM experiments, and research development folders. Included: Python source, Markdown records, protocol/source-lock/validation/analysis JSON. Historical prose may describe plans or intermediate decisions superseded by later results; use FINDINGS.md and the dated reports together.

Excluded: credentials and environment files, model weights and adapters, raw predictions, natural-language corpus rows, review sheets, sealed evaluations, private attachments, environments, logs, and large binary archives. Raw data and checkpoints remain in the local research archive. Scripts can contain original absolute paths and references to intentionally omitted resources: this is a source/evidence snapshot, **not a turnkey reproducible distribution**. Do not run historical runners blindly; some launch training or paid inference. Credential references in code are variable names only, not credentials.

A SHA-256 match verifies copied file identity, not scientific validity. The original source locks refer to the original complete local studies, not a promise that all locked files are published here. New experiments must use separate output directories, preserve historical protocols, and respect independent-review and sealed-data gates.

## Current execution status

The adapter diagnostic has completed loading/format preflight only. Its proposed full protocol is not frozen and no full diagnostic result exists. A terminal punctuation issue in the extraction parser needs a prespecified semantic-scoring rule, with strict formatting scored separately. No model training or new paid inference was performed for that preflight.

## Completed diagnostic update

The initial snapshot above preserves the preflight stage. The full diagnostic has now completed: see [results](completed_adapter_diagnostic/RESULTS.md), frozen protocol, source, analysis, and all 21 synthetic raw output files in `completed_adapter_diagnostic/`. Its own artifact manifest verifies the complete diagnostic. These synthetic raw predictions are an explicit exception to the original snapshot exclusions. No private natural data or weights are included.

## Post hoc shortcut audit

[Read the shortcut audit](adapter_shortcut_audit_2026-10-15/RESULTS.md). Reanalysis of saved predictions identifies a strong last-mentioned-actor pattern and shows why cleft-only accuracy gives an incomplete account. No new inference was performed; frozen outcomes remain unchanged.
