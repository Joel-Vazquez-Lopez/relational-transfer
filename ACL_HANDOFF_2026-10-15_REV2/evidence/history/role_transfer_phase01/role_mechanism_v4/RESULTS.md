# First internal-state analysis of ordered-role transfer

**Completed 7 October 2026.** This analysis reads frozen checkpoints from the [72-model behavioral confirmation](../role_aux_confirmation_v1/RESULTS.md). It is an exploratory sequence: [v1](../role_mechanism_v1/PROTOCOL.json) was fixed before its runs; [v2](../role_mechanism_v2/PROTOCOL.json), [v3](../role_mechanism_v3/PROTOCOL.json), and [v4](PROTOCOL.json) were documented adaptive follow-ups. None changed model weights or the behavioral confirmation.

## Can a simple readout recover roles from unseen scene states?

For each of the first paired starts on eight graphs, a fixed ridge linear readout was fitted to scene-tower states from **observed pairs only**. It predicted the source and target identity separately, then was tested on scenes from all 25 unseen pairs, with independent scene generation. This used 16 frozen models: ordered-role and matched unordered-identity training on each graph.

| Scene-tower state | Unordered control, both roles correct | Ordered-role model, both roles correct |
|---|---:|---:|
| After first Transformer block | 11.55% | 13.67% |
| After second block | 65.27% | 76.38% |
| Final scene representation | 64.47% | 76.27% |

At the final state, the ordered model's readout is **11.80 percentage points better** on average, with a positive difference on all eight graphs. The first-block difference is small and changes sign on two graphs. This supports the presence of more linearly accessible *ordered* role information later in the ordered model's scene tower. A probe alone does not show that this information controls the answer.

## Does changing the role-related state alter decisions?

For each unseen evaluation episode, the model scores a correct scene and a scene with actor/patient reversed. We kept the query fixed and exchanged only selected components of these two candidate scene representations. The readout learned from observed pairs defined a role-order direction (`source` minus `target`) and a separate unordered-identity direction (`source` plus `target`). Five random subspaces had the same nine-dimensional rank. All tests used 25 unseen pairs times 128 balanced episodes per model.

On the first start, the role-order component was more consequential than the unordered-identity component on every graph. In [v3](../role_mechanism_v3/PROTOCOL.json), the identical procedure was repeated on the two remaining starts per graph, which had not been examined in the internal-state analysis. The ordered model's correct-minus-reversed score margin fell **4.71 units** after the role-order swap, versus **1.86 units** for the unordered-control model. The paired difference in margin reduction was **+2.86 units**; it was positive on all 16 held-aside starts and all eight graph averages. This meets the v3 descriptive replication rule. It remains within the same eight graph layouts and is not an independent new-graph confirmation.

## Control for intervention magnitude

The role-order projection captured more of the correct-versus-reversed embedding difference than a random projection. V4 therefore scaled each random projected shift to exactly the **same L2 norm per episode** as the role-order shift, before exchanging components and renormalizing. It covered all three starts, all eight graphs, and both auxiliary arms: 48 frozen checkpoints.

| Model arm | Ordinary binary role accuracy | After role-order swap | After matched-energy random swap | Ordinary score margin | Role swap margin | Matched random margin |
|---|---:|---:|---:|---:|---:|---:|
| Unordered identities | 90.20% | 85.06% | 89.08% | 4.81 | 2.93 | 3.18 |
| Ordered roles | 98.81% | 87.50% | 97.01% | 7.41 | 2.50 | 3.80 |

In the ordered models, the role-order swap lowered binary role accuracy by **11.31 points**, while the equal-energy random swap lowered it by **1.80 points**. The matched-random minus role-swap margin difference was **+1.30 units** on average and positive on **all eight graph averages**. The same difference was only **+0.25** in the unordered-control models. Thus the effect cannot be explained only by the role subspace moving the state farther than the random controls do. The [machine-readable summary](SUMMARY.json) reports each graph and both arms.

| Graph | Final-state probe gain, ordered − unordered | Held-aside-start margin-drop contrast, ordered − unordered | Matched-energy random − role-swap margin in ordered arm |
|---|---:|---:|---:|
| fresh0 | +11.25 points | +1.98 | +1.12 |
| fresh1 | +7.13 points | +2.23 | +0.89 |
| fresh2 | +17.75 points | +3.06 | +1.36 |
| fresh3 | +6.00 points | +2.60 | +1.29 |
| fresh4 | +14.50 points | +3.23 | +1.43 |
| fresh5 | +6.63 points | +2.55 | +1.42 |
| fresh6 | +28.25 points | +3.79 | +1.39 |
| fresh7 | +2.87 points | +3.41 | +1.46 |

## What this establishes and what remains open

These observations support a **causal contribution of a linearly identifiable role-order component** of the scene representation to unseen-pair decisions in this controlled architecture. They make the behavioral training result more informative than accuracy alone: ordered supervision changes a later scene state in a way that matters to the role decision.

The analysis does not identify a unique neuron, attention head, or complete circuit. The intervention creates hybrid internal states that need not occur naturally; the readout was trained using explicit observed-pair role labels; and the generated scenes contain movement cues to the source entity. The controls reduce, but do not eliminate, alternative explanations. The adaptive v2-v4 sequence should be independently replicated on new layouts or a distinct architecture before making a strong mechanistic publication claim. It says nothing directly about a pretrained language model's internal computation or whether role training repairs prompt sensitivity there.

Raw per-model results are in the `fresh0.json`–`fresh7.json` files under `role_mechanism_v1/`, `role_mechanism_v2/`, `role_mechanism_v3/`, and this directory. Analysis code is `../role_mechanism_v1.py`, `../role_mechanism_v2.py`, `../role_mechanism_v3.py`, `../role_mechanism_v4.py`, and `../summarize_mechanism_v4.py`. Every stored checkpoint hash was checked against its frozen model file.
