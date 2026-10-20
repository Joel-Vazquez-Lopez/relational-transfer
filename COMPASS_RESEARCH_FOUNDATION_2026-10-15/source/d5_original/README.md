# Original D5 source copy

This folder contains unchanged copies of the D5 source, three configuration files, and [the original report](original_report.md) from `/Users/joelvazquezlopez/Documents/Codex/2026-09-08/files-pasted-by-the-user-you/`. Existing trained checkpoints and results were not copied. The source and config layout is preserved so D5 resolves its paths locally.

Use a Python environment with NumPy and PyTorch. On this machine:

```sh
PY=/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python
$PY d5_original/src/d5_topology.py --run
```

The pretraining audit was completed in this workspace and saved `d5_original/results/d5_topology/pretraining_audit/pretraining_validation.json` with `pass: true` across all nine critical checks. D5 requires this file before `--run`. Do not rerun `--audit-only` in this same output directory; the original code refuses to overwrite it. The full command trains 24 original models and may take substantial CPU time. Its result files stay inside this folder.

The separate `../d5_indirect_support/` experiment changes the graph and training sampler to test the indirect-support hypothesis. Its results should not be described as a literal rerun of original D5.
