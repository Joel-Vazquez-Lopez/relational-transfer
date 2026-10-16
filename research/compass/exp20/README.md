# Compass 20 — counterfactual precision calibration

Read [the full scientific report](../INVESTIGATION_20_2026-10-16.md). This is a **read-only post-hoc synthetic** reanalysis of three frozen causal Transformer checkpoints from Investigations 18 and 19. No new weights were trained.

The full self-contained package, **not committed to GitHub**, is `COMPASS_INVESTIGATION_20_COUNTERFACTUAL_PRECISION_2026-10-16.zip` in the Compass conversation. SHA-256: `60797f315860e70b45334dbe3ded9384f3837ef9315fc8b1ae1410de556b723a`. The ZIP has **26 entries including the manifest**, with 25 validated file hashes; the package contains all frozen weights, executable CPU PyTorch probability replays, JSONL predictions, bootstrap and six passing tests.

The GitHub [stdlib scorer](audit_counterfactual_precision.py) only reads the archive's raw JSONL files; it does **not** infer new model predictions itself. After extracting, run:

```bash
python audit_counterfactual_precision.py \
  --original /path/to/reference19/results/paired_predictions.jsonl \
  --logits /path/to/frozen_logits.jsonl \
  --offpath /path/to/offpath_logits.jsonl
```

To regenerate logits from frozen checkpoints, run `python frozen_logits_replay.py` and `python offpath_logits_replay.py` **inside the extracted archive**. No source corpora, sealed examples, paid inference or newly annotated labels used.

**Critical caveat:** 95 correlated synthetic source worlds, post-hoc edit selection and models that cannot generalize to whole worlds. This is a failure-of-evidence calibration, not identification of a natural relational circuit or ACL-level novelty.
