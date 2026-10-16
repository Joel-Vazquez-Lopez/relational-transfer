# Compass Investigation 19 — replay instructions

This is a **read-only synthetic falsification** of the interpretation of Compass 18's instruction-versus-operand prediction-change contrast. The original Transformer weights were not modified, and no new training or pretrained inference was run.

- [Scientific report](../INVESTIGATION_19_2026-10-16.md)
- [Stdlib raw-output scoring audit](audit_raw.py)
- Full 22-member ZIP supplied in the Compass conversation: `COMPASS_INVESTIGATION_19_CORRECT_UPDATES_2026-10-16.zip`, SHA-256 `89dfb6d3f21837ba7bc0ec0da2cb76212d6fe76b67c366fd352ff61bf73081c5`.

The full archive includes `reference18/` checkpoint files and Python model source, all **1,128** reconstructed response triples, the new relevant-vs-offpath B-table edits, per-model source-world cluster bootstraps, four passing tests and a file-hash verifier. The ZIP is **not hosted here**, and GitHub's summary alone does not permit rerunning the model inference.

After extracting the archive, run:

```bash
python exp19_correctness_audit.py --exp18 reference18 --out results
python -m unittest -v test_exp19.py
python cluster_audit.py
python audit_raw.py results/paired_predictions.jsonl
python verify_archive.py
```

Or run the GitHub `audit_raw.py` against the archive's `results/paired_predictions.jsonl`.

**Main falsification:** across 1,128 source-world-dependent paired comparisons, instruction edits changed predictions **690** times but produced **270** correct new answers, whereas matched operand edits changed predictions **302** times but produced **457** correct new answers. Exact both-route correct updates occurred in only **201** cases. On a 546-row same-old-to-new-value irrelevant-operand null, even a gold-irrelevant edit changed the prediction **104** times. These observations do **not** identify a native mechanism or establish a pretrained-language-model result.
