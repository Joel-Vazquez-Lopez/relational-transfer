# B-D21 frozen discovery protocol: source cue × relative event

Status: **discovery only**. Selected after B-D20 and before B-D21 data generation or model training.

## Question

Does changing which feature is useful during training redirect the learned event computation when the relative trajectories, identities, candidate foils, model, initialization, and update schedule are otherwise matched?

## Worlds and matched information

Start from the B-D15 source-balanced renderer. For every generated candidate scene, add a global x translation of magnitude 0.65 to all three objects at every timepoint. This leaves every within-scene relative trajectory unchanged. In **NO_CUE**, choose translation sign independently of event with probability 0.5. In **CUE**, use positive translation for approach/event0 and negative for recede/event1. The translation marginal is balanced in both worlds. Source initial x ranges become disjoint by sign, so event is source-decodable in CUE but not in NO_CUE by design. The four-choice episode still includes role reversal, event reversal, and their combination, and candidate order is randomized. Event cue alone cannot solve role selection.

Generate 12,000 train and 2,400 independent test episodes from fixed latent scenes. Save both world arrays and their labels before training. Freeze the data hash, code, model configuration, seeds, optimization settings, and outcome definitions. Use new exploratory seeds 3801 and 3802. For each seed, initialize NO_CUE and CUE from identical parameters and use the same batch-index sequence: AdamW, 3,000 updates, batch128, learning rate 0.0008, frozen speaker/emit, fixed oracle messages. Save checkpoints at 1,000, 2,000, 3,000. No canonical model is retrained.

## Frozen tests

1. Independent four-choice accuracy at each checkpoint. A model with final accuracy below 0.95 is not used to infer a mechanism.
2. On 600 B-D15 exact-source paired base/alt scenes, evaluate the complete 2×2 grid of **relative event** (approach/recede) × **translation cue** (+/−). For NO_CUE, all cells are supported by training. For CUE, diagonal cells are supported and off-diagonal cells are explicit cue–relation conflicts outside its joint training support.
3. For each scene, compute event-query preference (s=10(q_0-q_1)\cdot c). Report supported diagonal accuracy, off-diagonal preference by relation and by cue, and relation/cue/interaction contrasts over the full 2×2 grid.
4. Report whether the two matched-seed worlds diverge despite similar task accuracy. If CUE follows cue on conflicts while NO_CUE follows relation, proceed to causal pathway analysis. If both follow relation or both use cue, inspect data/optimization before any mechanistic claim.

## Strong mundane alternatives

The global translation changes absolute coordinates as well as cue availability. CUE conflict cells are off its joint support. Behavior there indicates the learned function's extension, not confirmed in-support use. The listener may use target or global position rather than the source token for the shortcut. An apparent mechanistic difference can also be a seed/optimization effect; matched initialization and two seeds constrain but do not eliminate that possibility. The diagnostic therefore motivates pathway tests and does not itself establish a transformer-native operation.

## Boundary

P3.2-E remains the canonical endpoint. All B-D21 artifacts stay in this directory. Never read protected seeds 1704–1706, P4, or other marked confirmation material. Do not modify canonical artifacts.
