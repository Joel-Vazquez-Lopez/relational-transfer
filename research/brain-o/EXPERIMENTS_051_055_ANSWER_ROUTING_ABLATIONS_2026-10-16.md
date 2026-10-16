# Brain-O Experiments 051–055 — Answer rescue or structural shortcut?
**16 October 2026 | EXECUTED & SAVED REPLAY VERIFIED | synthetic open-development only**

## Scientific question
Two frozen 3-layer Transformers with nearly perfect (verb, context-tag) **event-index** readouts still answered only about 33% of three-event agent/patient questions. Can a newly learned answer head exploit their accurate event selection? Does success require learned source-side relational representations or merely investigator-supplied structure?

### Source models and protocol
- Frozen trained selector models: `e050_seed1040_step750.pt` and `e048_seed1041_step750.pt` from Brain-O E045–050. Both used *generator-derived privileged event-index auxiliary supervision while training*; no backbone retraining here.
- E051–053: 1200 generated training worlds / 300 held-out generated worlds per seed; six correlated questions/world (1800 eval answers). E054–055: 1500 training worlds / distinct 300-world heldout panel per seed.
- The generator, grammatical constructions, participant name inventory and fixed clause positions are unchanged. These are not independent language/source families.

## E051–053: explicit grammar-pointer rescue **falsifies latent source-binding interpretation**

| Arm (1800 QA per seed) | Seed 1040 correct | Seed 1041 correct | Complete six-answer worlds |
|---|---:|---:|---:|
| Newly trained MLP from original final answer hidden state | 577 | 593 | 0 / 300 each |
| **Explicit candidate-name pointer + trained model event prior** | **1796** | **1800** | **298/300; 300/300** |
| Same pointer, event prior uniform | 600 | 600 | 0/300 |
| **Source hidden vectors all zero; explicit role + active/passive + candidate positions retained** | **1796** | **1800** | **298/300; 300/300** |
| Source vectors from **randomly initialized** Transformer, trained event prior retained | **1796** | **1800** | **298/300; 300/300** |

A wrong-event prior gives 0/1800 correct in both. Oracle correct-event prior yields 1800/1800. All use the same *explicitly wired* structure-to-candidate pointer with learned local scores. The pointer is allowed to know participant candidate token positions, voice marker, query role and source name IDs. **It can reach ~100% while ignoring all learned source representations. Therefore the 100% answer rescue is a transparent engineered copier calibration, NOT recovery of a novel latent relational state.**

## E054–055: less-structured frozen-hidden-state bridge (different heldout panel)
A ~9816-param MLP combines the query-role hidden state with a selector-weighted sum of **three clause-end hidden states**. It receives **no explicit active/passive flag, no name-ID candidate pointer, and no direct name copying**, but it is still given the fixed clause-end positions.

| Arm (1800 QA per seed) | Seed 1040 | Seed 1041 | Whole worlds |
|---|---:|---:|---:|
| Learned source states + native trained event prior | **1036/1800 (57.56%)** | **1053/1800 (58.50%)** | 4/300 each |
| Same architecture, independently trained with **uniform event prior** | 586/1800 (32.56%) | 596/1800 (33.11%) | 0/300 |
| Same architecture, trained with **zero source state** | 594/1800 (33.00%) | 600/1800 (33.33%) | 0/300 |
| **Random untrained source states** + trained query-role/event streams | **1026/1800 (57.00%)** | **1037/1800 (57.61%)** | 2/300; 4/300 |

Switching a routed model to an intentionally incorrect event prior changes ~60% of answer predictions. The trained event posterior therefore matters functionally. But random source hidden vectors nearly match trained source vectors (difference 10 or 16 answers/1800), and entire-world success remains very low. **Do not infer learned abstract binding from this result.**

### Main scientific decision
The positive result is a *measurable availability/usability distinction under a constructed routing interface*. The strong falsifier is that **learned source-side representations were unnecessary for the perfect scaffold rescue**, and near-unnecessary for the weaker neural readout. The general relational-computation mechanism remains unidentified. This is compatible with generic lexical access, residual copying and structural readout specialization. Two seeds and correlated same-grammar worlds preclude generalization claims. The E051 and E054 architectures differ, so the 100% vs 58% comparison is not a matched architectural ablation.

### Next gate
1. Require a **competence-matched ordinary neural copying/lookup alternative** with the same internal patch interfaces; source random-control and parser sham are mandatory.
2. If pursuing a native rescue, intervene on **the existing model's answer pathway**, not construct an explicit symbolic pointer with privileged candidate locations. Include exact heldout-world and sham/distractor causal tests, equal-param controls and representation sensitivity.
3. Run authentic human-adjudicated multiple/repeated predicate/role pilot and >=2 untouched pretrained **model families** after issue #23 licensing, label, competence and model-view gates. Astra owns output-field-order tests. Do not duplicate.

## Reproducibility and limits
Full conversation artifact: `BRAIN_O_E051_055_2026-10-16.zip` SHA-256 `a23b9e9d60fefa738aca51486c957b6ea5f04a65b760110bca65ab43e9a47890`, 50 members including complete scripts, two frozen selector checkpoints, 24 saved readout checkpoints, raw E051 per-query predictions, outputs, logs, detailed methodological report and SHA256 manifest. ZIP CRC passed and 49/49 member hashes verified. `python e051_055_verify_saved.py` independently reproduced **24/24** heldout correct-answer counts from saved weights. Raw files/weights remain attached to Brain-O conversation rather than GitHub; do not assume other agents received them.

**Status: PROGRESS and strong local FALSIFICATION of a tempting overclaim, NOT validated EUREKA.** No paid model inference, sealed data, or modifications of other branches.
