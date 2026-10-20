# Interrupted v3 run: padding-mask implementation error

**13 October 2026.** V3 was designed after v2 to replace an irrelevant clause with one constant marker rather than a variable number of `<mask>` tokens. It generated a fresh bank and froze `PROTOCOL.json` before training. One `joint_replay` start (seed 11) completed and was evaluated; the paired `clause_decomposition` training was interrupted during epoch 2. No other v3 starts were run.

During execution, inspection found that `Planner.forward` used `ids == 0` as its attention padding mask. In the sorted `VOCAB.json`, token ID `0` is `<cls>`; actual `<pad>` has another ID. Thus the model hid the class token as an attention key and allowed real padding as keys. The same inherited defect affected the prior scratch Transformer and v2 planner pilots. V3 was stopped immediately instead of completing a known-invalid sweep. Its partial checkpoint/output are retained only for audit, **not** reported as a scientific result.

The corrected v4 branch uses `vocab["<pad>"]` explicitly, regenerates a fresh bank excluding all prior prompt signatures, and labels its outcomes exploratory because its design was informed by these attempts. No frozen script or result from v3 was overwritten.
