# Phase 1 validity record (2026-10-13)

This is the prospective 12-entity factorial for the controlled Transformer. It
does not reuse model outcomes to choose graphs. The older ten-entity result is
exploratory evidence because the original rewire moved both shared-partner and
three-edge-path counts.

## Graph intervention

- Eight nonisomorphic anchor graph classes are selected by a deterministic
  graph-only search. Each graph is connected, 12-node, degree four, with 24
  observed unordered pairs and the same withheld focal pair (0, 1).
- Every family contains common-neighbor counts 1 and 2 crossed with simple
  three-edge-path counts 3 and 4. Each cell is at most one degree-preserving
  two-edge switch from the anchor.
- All four cells in a family match total triangle count and the two focal
  vertices' triangle counts. Four-cycle count differs by at most one.
- The graph factors are separated; other local motif counts can still change.
  A result supports a local graph-structure effect under these controls, not a
  unique mechanism of common neighbors or path length.

## Training and evaluation

- Every observed pair is shown in both directions, both events, four selected
  scene types, and four answer positions. The pool is 1,536 configurations per
  cell. Degrees ensure each entity has the same occurrence count across cells.
- The fourth coded-message token is a dedicated padding token. Distractor
  identities are exactly balanced within each semantic candidate over
  five-epoch blocks. The actor and patient are never used as distractors.
- The focal 320-item evaluation balances actor direction, event, answer
  position, and all ten possible distractors, with two independent renderings.
  A second, prespecified scene seed checks rendering sensitivity.
- Binary role accuracy compares the correct candidate only with its
  role-reversed counterpart; four-choice accuracy and event accuracy are
  separate. Familiar performance is a gate, not the structural endpoint.
- Graph family is the independent analysis unit. Seeds and rendered scenes
  are repeated measurements; they are not treated as independent families.

## Preflight

The separate development graph (generator seed 89999) was trained for 960
updates. It reached 0.9891 familiar four-choice accuracy. Its unseen-pair
accuracy was not evaluated. The preflight showed the 12-entity task can be
learned and did not alter the frozen factorial design.

## Remaining threats

The scene geometry may make the moving actor easy to identify. This does not
remove the held-pair transfer question, but it limits claims about natural
language. A degree-preserving edge switch changes individual observed pairs,
so the two graph counts are not the only changed properties. Crossed factors,
matched motifs, family replication, and held controls narrow that ambiguity;
they do not eliminate it.
