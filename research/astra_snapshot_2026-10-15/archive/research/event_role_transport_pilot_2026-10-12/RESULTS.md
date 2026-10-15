# Event-role transport pilot: screen failed

**12 October 2026.** Exploratory matched Qwen2.5-0.5B pilot. The protocol, new 36-event lexical bank, metrics, and screen were frozen before training or inference on that bank. The human-review challenge remained sealed.

## Comparison

Both arms continued the same natural-role source adapter in each of two starts. Both saw 24 training units (each containing the two directions of an event), three sentence forms, three QA questions and an ordered actor–patient tuple per direction. Each had 48 optimizer updates and 1,152 supervised rows. The `transport` arm additionally penalized differences in candidate-role distributions across meaning-preserving forms, across swapped event directions, and between role slots and direct answers. The `slot_sft` arm had all the same QA and tuple labels without these auxiliary constraints.

On a new bank of 36 directed events using different names and verbs, each checkpoint answered five questions in each of four sentence forms: 720 prompts/checkpoint, all valid candidate-name outputs. The primary metric counts an event-form only when both *held-out* actor and patient question wordings are answered correctly.

| Start | Arm | Strict held-out joint /144 | Active /36 | Passive /36 | Subject cleft /36 | Object cleft /36 | All five /144 |
|---|---|---:|---:|---:|---:|---:|---:|
| fresh1 | Slot SFT | 31 | 4 | 12 | 13 | 2 | 21 |
| fresh1 | Transport | 28 | 4 | 11 | 12 | 1 | 18 |
| fresh3 | Slot SFT | 38 | 5 | 17 | 14 | 2 | 30 |
| fresh3 | Transport | 35 | 4 | 15 | 14 | 2 | 29 |

**Frozen positive screen failed.** Transport changes the primary score by **−3/144 in both starts** relative to its supervision-matched control. It does not improve object-cleft joint accuracy. This run provides no evidence that the auxiliary event-role transport loss repairs held-out role QA.

## Interpretation and limits

This is a narrow negative result for a specific implementation and budget. Both arms received explicit slot labels, so this experiment isolates the extra distribution constraints **conditional on slot supervision**. It does not compare the slot-SFT arm to ordinary QA-only fine-tuning under this exact schedule, nor establish whether a dedicated slot representation or a different optimization schedule would work. The pilot has two related starts, one 0.5B family, constructed sentences, 48 updates, and no human-verified natural test. The low object-cleft joint scores in both arms (1–2/36) make that failure the clearest remaining target.

The direct-answer link may be too weak when it aligns only two-candidate distributions at answer positions. It may also be constrained by training on three forms and one patient question wording. Those are hypotheses for later tests, not findings from this run. A useful next test would check whether the newly trained checkpoints can fill actor–patient slots correctly on the same held-out events; if they can while direct QA fails, then explicit answer grounding deserves a different implementation. If both slot and direct QA fail, the bottleneck is earlier in event interpretation or lexical transfer.

## Audit

`PROTOCOL.json` records the frozen design and source hashes. `run.py` trains and evaluates the four checkpoints; each checkpoint directory contains its adapter and `TRAINING.json`. `EVAL_*.json` contain all 2,880 raw answers. `analyze.py` and `ANALYSIS.json` audit the counts and frozen screen. No existing checkpoints were overwritten.
