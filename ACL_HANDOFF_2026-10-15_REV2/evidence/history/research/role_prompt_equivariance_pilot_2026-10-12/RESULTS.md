# Coupling role answers improves consistency, but does not clear the repair threshold

**12 October 2026. Status:** completed prospective exploratory Qwen2.5-0.5B training pilot. The [protocol](PROTOCOL.json), new lexical test, loss weights and positive screen were frozen before training or inference on this test. The reserved human-review challenge remained sealed. This is a research record, not student submission prose.

## Question and matched comparison

The preceding [wording audit](../role_prompt_transfer_audit_2026-10-12/RESULTS.md) found that ordinary role training could improve patient answers while harming an actor question absent from training. Here both new training arms see the **same 48 directed events, forms, three questions, 96 updates and 864 answer targets** from the same natural-role source adapter in each of two starts:

- **Balanced SFT:** ordinary answer cross-entropy.
- **Role coupled:** the identical cross-entropy plus a loss that brings candidate-name distributions for the *same role* closer across question wording and sentence form, and a contrast that separates actor from patient questions. Both event directions and the supervised gold answers prevent a same-name-everywhere solution on the training examples.

The new evaluation uses **six further disjoint name pairs and three further disjoint verbs**, yielding 36 directed events. Each appears in active, passive, subject-cleft and object-cleft form, with three actor and two patient questions. “Carried out” (actor) and “Which person was VERBed?” (patient) were absent from this training. Both arms receive the same 720 prompts per checkpoint.

## Primary held-out result

The primary outcome is the number of event–form groups for which **both held-out questions** get their respective correct answers. Each denominator is 144 (36 events × four forms).

| Start | Balanced SFT | Role coupled | Paired gain | Frozen screen |
| --- | ---: | ---: | ---: | --- |
| Fresh1 | 101/144 | **109/144** | +8 | Meets numeric gain |
| Fresh3 | 100/144 | **103/144** | +3 | Fails numeric gain |

The predeclared positive screen required at least **+8/144 in each start**, with limited regression in three previously strong cells. The regression conditions are met, but the **primary screen fails** because Fresh3 gains only three. In the paired comparison, Fresh1 has nine newly correct and one newly incorrect event–form; Fresh3 has three newly correct and none lost. These are related starts on the same test items, not independent replications of 144 observations each.

## What improves, and what remains broken

| Start | Arm | All five answers correct /144 | Actor-question contradictions /144 |
| --- | --- | ---: | ---: |
| Fresh1 | Balanced SFT | 91 | 21 |
| Fresh1 | Role coupled | **107** | **10** |
| Fresh3 | Balanced SFT | 88 | 20 |
| Fresh3 | Role coupled | **98** | **10** |

These are useful **secondary** signals: with identical training data, the auxiliary objective increases complete event–form correctness and roughly halves contradictory actor answers in both starts. It is not merely making all actor answers identical while ignoring patient questions, because the all-five *correct* count rises too. However, some of this gain comes from questions and forms included during training; the held-out primary gain is less consistent.

The difficult object-cleft condition remains unresolved. The held-out actor-and-patient joint count is only **3→5/36** in Fresh1 and **3→4/36** in Fresh3. For the verb-specific patient question on object clefts, accuracy is **3→11/36** and **1→10/36**—an improvement, but still poor. Subject-cleft performance is near ceiling in both arms, so it provides little room to distinguish methods. The main improvements in all-five correctness occur in active and passive conditions.

All **2,880 test outputs** are valid candidate names. The pattern therefore is not a first-token parsing artifact. Nevertheless, the evaluation uses constructed sentences, one small Qwen family, two related source starts and a forced two-name response format. It does not prove a stable internal role representation or transfer to authentic language.

## Scientific implication

This is the first matched pilot here suggesting that **an explicit role-and-prompt coupling loss can change behavior beyond adding the same question variants to supervised training**. The evidence is promising but too weak for a repair claim: the frozen primary threshold fails, object clefts remain poor, and we have not compared against a generic hard-example or margin regularizer. It also does not establish that the proposed objective is new in the literature. Prompt-consistency training has substantial precedent, including [Elazar et al. (2021)](https://aclanthology.org/2021.tacl-1.60/) and [Hejabi et al. (2026)](https://aclanthology.org/2026.acl-long.71/).

The next diagnostic should explain the **object-cleft patient failure** before increasing model size or stacking more objectives. A useful control would test whether the gain comes from role-specific coupling or simply from extra emphasis on low-margin examples, with the same data and weight budget. Any eventual paper-level repair would also need independently checked authentic sentences, held-out question families rather than a single new phrasing per role, and a second model family.

## Audit files

- [Training and inference code](pilot.py), [frozen protocol](PROTOCOL.json), four saved adapter checkpoints and training traces.
- Four EVAL prediction JSON files, [analysis code](analyze.py), and [machine-readable analysis](ANALYSIS.json).
- Checked: script/protocol/checkpoint hashes match; all four arms completed 96 updates and 864 answer targets; all 2,880 test records are unique and valid.
