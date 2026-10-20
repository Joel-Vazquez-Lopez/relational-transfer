# Clause-decomposition auxiliary training: frozen negative test

**13 October 2026.** This is a separate, prospectively frozen follow-up to the failed trace-order pilot. It tests whether training a model to judge each question clause separately breaks a shortcut caused by perfectly correlated operation directions. It predicts **two role directions only**; it does not bind participants or answer the event question.

**Implementation correction discovered afterward:** the model's attention mask treated token ID `0` as padding, but `0` belonged to `<cls>` in the saved sorted vocabulary. Actual padding was not masked. All scores below remain intact for audit, but are **bug-affected and cannot support a scientific conclusion about clause training**. A corrected model must use the vocabulary's actual `<pad>` ID and a fresh held-out bank.

The fresh data exclude all 5,200 exact prompt signatures from the first pilot. Training has 6,000 two-step questions with `FF` or `RR` operation pairs. The test has 400 new same-pattern questions and 400 `FR`/`RF` questions, with no exact prompt overlap. Five paired seeds compare identical two-layer Transformer classifiers. Both arms receive the same full-question primary views, labels, updates, and an additional pair of equal-length auxiliary views. `joint_replay` repeats the full view; `clause_decomposition` masks words deemed irrelevant to the queried step. The masks use the benchmark's known clause boundaries, so this is oracle structural supervision during training. Primary inference gives **both models the full unmasked question**.

The prespecified screen required high in-domain plan accuracy and a mixed-pair gain of at least 40/400 in four of five paired seeds. It **failed**.

| Seed | Joint replay in-domain | Clause training in-domain | Joint replay mixed, full question | Clause training mixed, full question | Clause training mixed, masked diagnostic |
|---|---:|---:|---:|---:|---:|
| 11 | 400/400 | 400/400 | 0/400 | 0/400 | 200/400 |
| 23 | 400/400 | 400/400 | 0/400 | 0/400 | 400/400 |
| 37 | 400/400 | 400/400 | 0/400 | 0/400 | 0/400 |
| 41 | 400/400 | 400/400 | 0/400 | 0/400 | 0/400 |
| 53 | 400/400 | 400/400 | 0/400 | 0/400 | 0/400 |

In every mixed full-question evaluation, both arms predicted the **same direction for both steps** on all 400 items, despite the true directions being opposite. This resembles a correlation shortcut, but the incorrect padding mask prevents that interpretation from being established. The auxiliary masked-view result varied by seed and did not transfer reliably to unmasked input. That variation is not a positive method result.

One design flaw became apparent after evaluation: replacing each irrelevant word with `<mask>` preserves the *number* of removed words. Since outer question forms have different lengths and directions were perfectly correlated during training, a masked view can still expose a shortcut about the other clause. This is a plausible explanation for the seed-sensitive masked diagnostic, not a proven account. The frozen v2 script and results remain unchanged. A separately labeled v3 test will replace each irrelevant clause with a **single constant marker** and use a fresh bank.

`PROTOCOL.json` fixed the data, hashes, seeds, arms, metric, and screen before training. `ANALYSIS.json` and the per-seed raw `EVAL_*.json` files retain all predictions and paired metrics. These five starts share the same 400 mixed examples. No natural, sealed, or project-proposal data were used.
