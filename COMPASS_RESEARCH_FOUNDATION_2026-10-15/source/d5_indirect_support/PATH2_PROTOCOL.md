# Same-distance graph control — locked before training, 5 October 2026

The prior eight-entity experiment found that the withheld pair with two common
training neighbours performed much worse than one with none. Its contrast also
changed shortest path (two versus three). This experiment tests whether a
similar direction appears when **both withheld pairs have shortest path two**.
It is a prospective, conditional follow-up to a discovered pattern, not an
independent first discovery or a public preregistration.

## Fixed design

- Source: `path2_control.py` SHA-256
  `adca58662986f1ca8c3bb6fd8f9a241dc72ee4d90debb05a0d6d259dc5c0c1f0`,
  importing unchanged `run.py` SHA-256
  `b12e4bf3e65770bdd39d1fb2640627eb76f6b96e6903f7efcb72f4c44c1ee03d`.
  The first training invocation failed before its first optimizer update due to
  an 18-versus-20 candidate-width typo. The source hash here is the repaired
  version, fixed before any training outcome was produced.
- Three nonisomorphic, connected, ten-entity degree-three graph worlds are fixed
  in the source. The withheld pairs have one versus two common training
  neighbours; both have shortest path two. All endpoints have degree three and
  participate in exactly one triangle. Global graph and 960-entry canonical
  semantic training pool are identical between conditions within each world.
- Seeds: 9201, 9202, 9203 in every world and both conditions; 18 models total.
  Each uses the same D5 listener except that the entity one-hot and symbol
  vocabulary expand from eight to ten identities. Train for 6,000 updates,
  batch size 128, AdamW learning rate 0.0008, final checkpoint only.
- Within a world and seed, construct each episode once in canonical labels and
  then relabel identity channels and oracle endpoint symbols. All candidate
  trajectories, answer positions, distractor roles, episode order, and random
  seeds are paired. Held-out `(0,1)` evaluation examples are literally
  identical across conditions. Use 1,024 independently rendered familiar and
  1,024 held-out examples per model.
- The audit checks all 960 canonical pool entries in each world, exact scene
  pairing, absent direct focal edges, graph properties, and identical held-out
  examples. No condition, seed, graph, or checkpoint will be discarded based
  on held-out outcomes.

## Outcomes and decision rule

The primary descriptive contrast is shared-two minus shared-one mean
correct-minus-reverse held-out score margin, paired by world and seed. Report
all nine contrasts and their mean within each graph world. Familiar four-way
accuracy must be at least 95% in **all** 18 models to treat this as a transfer
test. The prior reverse-direction pattern passes this follow-up only if every
world's mean margin contrast is below -0.20 and at least seven of nine paired
contrasts are negative. Otherwise report a mixed or failed pattern. Held-out
four-way accuracy and query separation are secondary; do not substitute them
for a failed primary result. No cross-world inferential p-value is planned
from only three graph worlds.

Even a passed comparison does not identify common-neighbour count alone:
other pair motifs can differ. A failed comparison does not erase the previous
replication; it narrows its possible explanation. Neither outcome directly
supports a claim about authentic language or pretrained LLMs.
