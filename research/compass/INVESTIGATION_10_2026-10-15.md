# Compass Investigation 10 — Explicit-route causal cuts (2026-10-15)

**Status:** completed local causal-route calibration, **not** a discovery about natural pretrained Transformer computation.

## Research context
Before running, read `main:research/WORKSTREAMS.md` and recent workstream updates. Brain-O's 016–020 event/role selection experiments explicitly do not establish relation caching. Compass remains independent; no other branches or sealed data were changed.

## Design
Read-only loading of all **nine saved Investigation 08 checkpoints**: seeds 19/29/41 × reconstruction costs .25/.40/.75. Each has a shared causal self-attention prefix, an explicit one-bit prefix cache head, a question-conditioned raw-reader head, and a hard query-time route gate.

For every target world/question and every different donor world (16 × 4 × 15 = **960 paired interventions per checkpoint**), hold the target gate fixed. Independently substitute **only the cache head output** or **only the raw-reader head output** from the donor into the final selector. Compare with the natural target prediction; also retain both-head donor controls.

This is a direct hard-gate head-output interchange **at an architecturally known interface**. It does not identify an unknown internal representation or causal path.

## Observations

| Cost | Seed | Natural raw route uses /64 | Cache donor changes /960 | Raw donor changes /960 |
|---|---|---:|---:|---:|
| .25 | 19 | 44 | 220 | 352 |
| .25 | 29 | 44 | 220 | 352 |
| .25 | 41 | 60 | 15 | 480 |
| .40 | 19 | 0 | 440 | 0 |
| .40 | 29 | 0 | 480 | 0 |
| .40 | 41 | 0 | 512 | 0 |
| .75 | 19 | 0 | 440 | 0 |
| .75 | 29 | 0 | 440 | 0 |
| .75 | 41 | 0 | 440 | 0 |

All nine raw-reader branches scored 64/64 on the familiar 16-world evaluation. At costs .40/.75 the route gate never uses that perfectly accurate branch and raw-reader donor changes have **zero** effect on the answer. At cost .25 both routes can causally affect outputs. This behavior is mechanistically explained by the **explicit hard gate** and hence is a calibration of the proposed assay, **not** evidence of spontaneously selected Transformer relational reuse.

## Scientific interpretation
The ability to produce correct answers from a branch differs from that branch's **native causal contribution** to final answers. Behavioral accuracy and whole-prefix donor patches cannot establish computation reuse. Targeted interventions can identify branch utilization when the routes are architecturally exposed, but cannot be interpreted as discovery in an unpartitioned pretrained network merely by analogy.

## Reproducibility
Complete standalone package `COMPASS_INVESTIGATION_10_CAUSAL_ROUTE_AUDIT_2026-10-15.zip` is attached in the Compass conversation and includes `causal_route_audit.py`, frozen `shared_router.py`, the nine source checkpoint copies, per-checkpoint paired raw response JSONs, `SUMMARY.json`, replay stdout, `REPORT.md` and SHA256 manifest. The ZIP is **not** stored in GitHub. Verified by rerunning the audit after making package paths relative. No retraining, new external API, or changes to saved source models.

## Next decisive requirement
An otherwise shared/unpartitioned model with matched successful known caching and question-time reconstruction controls, an assay blind to their schedule labels, and held-out constructions. Patching must selectively isolate the *computed relation* from mere preserved raw evidence, without presupposing where that relation lives. Until then the native-algorithm claim remains open.
