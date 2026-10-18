# Linguistic Development Atlas pilot

**CLOSED — technical NO-GO. Two Qwen forward passes; zero held-out or Llama passes.**

The user explicitly accepted AI-only annotation. The frozen runner then failed while saving its first Qwen row: both the input row and scorer supplied the expected keyword. One main capture was saved and recovered offline. No complete linguistic comparison exists. This was a Codex implementation defect; the linguistic hypotheses remain untested.

## Results and artifacts

- [REPORT.md](REPORT.md): full outcome, missing outputs, startup deviation and limitations.
- [EXECUTION_STATUS.json](EXECUTION_STATUS.json): exact counts and closed status.
- [qwen7b/PARTIAL_OUTPUT.json](qwen7b/PARTIAL_OUTPUT.json): recoverable scored output, input, token/site ledger and array schema.
- [qwen7b/FORWARD_JOURNAL.json](qwen7b/FORWARD_JOURNAL.json): both calls recorded before execution.
- [RUN_qwen7b.log](RUN_qwen7b.log): fatal traceback.
- [ANNOTATION_DECISION.json](ANNOTATION_DECISION.json): direct user waiver and AI-only limitation.
- [ZERO_INFERENCE_STARTUP_RECOVERY.json](ZERO_INFERENCE_STARTUP_RECOVERY.json): GPU access failure and administrative recovery before actual inference.
- [LOCK.json](LOCK.json), [PROTOCOL.json](PROTOCOL.json): prospective frozen source/settings.
- [run.py](run.py), [instrument.py](instrument.py), [analyze.py](analyze.py): original frozen scripts, unchanged after outputs. Full atlas analysis was not run.
- [recover_failure.py](recover_failure.py): offline-only recovery and error reproduction, no model calls.
- [SERIALIZATION_REPAIR_PROPOSAL.diff](SERIALIZATION_REPAIR_PROPOSAL.diff), [OFFLINE_REPAIR_CHECK.json](OFFLINE_REPAIR_CHECK.json): mechanical repair proposal, not applied.
- [RESULT_MANIFEST.json](RESULT_MANIFEST.json): exact local hashes, including selected-site tensors and full-vocabulary logits. Large arrays remain local.

The original [design](../linguistic_development_atlas_design_2026-10-18/DESIGN.md) and implementation frozen at commit 7d90d6563c00a330ceb1255e8cea1436e8dccbf5 remain intact. PREFLIGHT_CHECKS.json and PREPARATION_MANIFEST.json are historical preparation records; they do not certify completed native validation.

The run is closed under its stop rule. No rerun, held-out evaluation, patching, adaptive experiment, paid call or model download followed the serialization failure.
