# Task v2 preregistered behavioral pilot

**Outcome:** **FAIL**  
**Seed:** 1701  
**Training:** exactly 8,000 updates; no restart, extension, tuning, or checkpoint selection  
**Mechanistic analysis:** not performed  
**Checkpoint SHA-256:** `d596197b4ce8ea655effebea305a09efe3ffe363a4dd946e04341b34bf8233c6`

## 1. Environment and integrity validation

The identity split was frozen before training as `H={0,1,2}` and `G={3,4,5}`.
The ten eligible training triples were the nine containing one H and two G
identities plus the all-G triple. Training contained no active H–H relation.

Task v2 used an exact 3,840-example semantic supercycle:

`2 families × 10 triples × 6 orders × 2 events × 4 selected members × 4 answer positions`.

The pretraining audit found:

- Family S/R: 1,920/1,920;
- base event 0/1: 1,920/1,920;
- every eligible triple ordering: 64 occurrences;
- every selected member/answer position conditional cell: exactly one;
- candidate distractor count deviation conditional on active endpoints: zero;
- source/target/event selected-meaning frequencies: 64–128, explicitly not
  claimed to be globally uniform;
- ambiguous trajectories, held-out semantic leakage, geometry collisions, and
  renderer rejections: all zero;
- identical candidate rendering rules for correct and foil scenes.

Two audit-code/generator problems were found before training and preserved in the
research log: an audit metadata indexing error, and random-uniform distractor
sampling whose realized conditional counts were uneven. Distractors were changed
to a deterministic rotating uniform schedule before the scientific run. A
six-color search was also corrected to use only Task-v2-supported triples.

All final generator checks passed.

## 2. Restricted semantic-oracle baselines

Finite enumeration reproduced Astra's balanced-mixture ceilings exactly:

| Restricted information | Exact ceiling |
|---|---:|
| Event only | 50.00% |
| Canonical orientation only | 45.83% |
| Unordered pair + orientation, no event | 62.50% |
| Source only | 50.00% |
| Target only | 50.00% |
| Source + event | 87.50% |
| Target + event | 87.50% |
| Unordered pair + event | 75.00% |
| Complete source + target + event | 100.00% |

The explicit pair/event dictionary has a 75% IID ceiling. Its declared fallback
for unseen pairs is uniform over four candidates, giving 25% held-out accuracy.

A six-message graph coloring was found for all 60 meanings and verified to assign
four distinct colors to every supported Family-S and Family-R candidate set. It
achieves zero behavioral error while remaining a holistic categorical code. This
confirms that Task-v2 success would not by itself imply tokenwise factorization.

## 3. Training

The frozen 118,208-parameter speaker/listener trained for exactly 8,000 online
updates on CPU. Wall time was 467.46 seconds (17.11 updates/s). Loss fell from
1.666 at step 0 but remained noisy near 0.5–0.7 late in training; the recorded
step-7,900 loss was 0.660. No result was viewed early and no rescue was attempted.

## 4. Primary behavioral results

Each candidate block was evaluated with all four possible speaker meanings against
the same candidate rendering. IID used 2,040 blocks per family (8,160 examples);
each held-out endpoint used 2,016 blocks per family (8,064 examples).

| Split/family | Source | Target | Event | Ordered pair | Joint |
|---|---:|---:|---:|---:|---:|
| IID — S | 6,478 (79.39%) | 6,743 (82.63%) | 6,307 (77.29%) | 5,570 (68.26%) | 4,629 (56.73%) |
| IID — R | 4,916 (60.25%) | 4,916 (60.25%) | 5,191 (63.62%) | 4,916 (60.25%) | 3,421 (41.92%) |
| Common held-out — S | 5,055 (62.69%) | 4,998 (61.98%) | 5,020 (62.25%) | 3,003 (37.24%) | 1,982 (24.58%) |
| Common held-out — R | 3,996 (49.55%) | 3,996 (49.55%) | 4,017 (49.81%) | 3,996 (49.55%) | 2,015 (24.99%) |
| New-inventory — S | 5,203 (64.52%) | 4,929 (61.12%) | 4,910 (60.89%) | 3,050 (37.82%) | 1,998 (24.78%) |
| New-inventory — R | 3,989 (49.47%) | 3,989 (49.47%) | 4,116 (51.04%) | 3,989 (49.47%) | 2,016 (25.00%) |
| Full held-out — S | 5,058 (62.72%) | 5,036 (62.45%) | 5,015 (62.19%) | 3,039 (37.69%) | 2,022 (25.07%) |
| Full held-out — R | 3,995 (49.54%) | 3,995 (49.54%) | 4,019 (49.84%) | 3,995 (49.54%) | 1,978 (24.53%) |

### Equally weighted family aggregates

| Split | Source | Target | Event | Ordered pair | Joint |
|---|---:|---:|---:|---:|---:|
| IID | 69.82% | 71.44% | 70.45% | 64.25% | 49.33% |
| Common held-out | 56.12% | 55.77% | 56.03% | 43.40% | 24.78% |
| New-inventory held-out | 56.99% | 55.30% | 55.96% | 43.64% | 24.89% |
| Full held-out | 56.13% | 56.00% | 56.01% | 43.61% | 24.80% |

The seed failed to acquire reliable IID behavior. Composition is at four-way
chance regardless of whether the distractor inventory was familiar or novel.

## 5. Family-S ambiguity tests

| Split | Source contrast | Target contrast | Event contrast |
|---|---:|---:|---:|
| IID | 2,197/4,080 (53.85%) | 2,316/4,080 (56.76%) | 2,088/4,080 (51.18%) |
| Common held-out | 1,009/4,032 (25.02%) | 1,000/4,032 (24.80%) | 935/4,032 (23.19%) |

All six ambiguity endpoints failed the 95% point threshold. These are joint
accuracies on the preregistered two-member ambiguity subsets; a missing required
factor cannot reliably resolve the paired alternatives.

## 6. Block-bootstrap confidence results

The preregistration used 2,000 block-bootstrap resamples and a one-sided
Bonferroni-adjusted alpha of 0.005 across ten endpoints. Simultaneous lower bounds:

| Endpoint | Point | Lower bound |
|---|---:|---:|
| IID S joint | 56.73% | 55.40% |
| IID R joint | 41.92% | 40.50% |
| Common S joint | 24.58% | 23.44% |
| Common R joint | 24.99% | 23.85% |
| IID S source contrast | 53.85% | 51.89% |
| IID S target contrast | 56.76% | 54.95% |
| IID S event contrast | 51.18% | 49.17% |
| Common S source contrast | 25.02% | 23.51% |
| Common S target contrast | 24.80% | 23.02% |
| Common S event contrast | 23.19% | 21.53% |

Every primary and ambiguity confidence criterion failed.

## 7. Held-out cell robustness

Every held-out directed pair, event, and family is present in
`heldout_cells.csv`, with 672 examples per cell. The worst was Family S,
source 1, target 0, event 1: 130/672 = 19.35%. The required minimum was 90%.
No weak cell is hidden by aggregation.

## 8. Communication controls

The equally weighted IID evaluation contained 16,320 examples.

| Condition | Joint accuracy |
|---|---:|
| Normal | 8,050/16,320 = 49.33% |
| No message | 4,080/16,320 = 25.00% |
| Exact independent replacement | 66,585,600/266,342,400 = 25.00% |

All-four-target evaluation makes a deterministic message-free choice correct once
per block, so block-bootstrap intervals for no-message and candidate-only controls
are exactly `[25%,25%]`.

Independent replacement used the exact empirical product of every observed
message and every IID target. Donors were exactly balanced between same/opposite
event, orientation, and candidate family: 133,171,200 each. Source matched in
51,492,864/266,342,400 pairings, target in the same number, and exact meaning in
6,214,656. Joint prediction positions were approximately uniform and joint
accuracy was exactly chance.

## 9. Candidate-only baseline

The separate 58,081-parameter permutation-equivariant model trained on the same
online Task-v2 stream for 8,000 updates. Its loss remained at `log(4)` and every
family/split result was exactly 25% because each candidate was the correct target
once within each evaluated block. All control intervals were `[22%,28%]` as
required. There is no evidence of candidate-only leakage.

## 10. Permutation invariance

- Candidate permutation: 16,320/16,320 predictions agreed after inverse mapping;
  accuracy stayed exactly 49.33%.
- Object slots: 15,697/16,320 predictions agreed = 96.18%; accuracy changed from
  49.33% to 49.20%, a 0.13-point loss.
- Speaker messages agreed for 15,853/16,320 examples = 97.14%.

Candidate invariance passed. Object prediction agreement failed the >99% gate,
despite the small aggregate accuracy change.

## 11. Protocol diagnostics

The IID protocol used seven symbols and 38 whole messages. Whole-message entropy
was 3.6044 bits. Symbol frequencies were: 0 = 46.23%, 6 = 30.66%, 7 = 10.05%,
4 = 7.64%, 5 = 3.57%, 1 = 1.58%, and 3 = 0.27%.

Across family/meaning conditions, mean dominant-message frequency was 56.92% and
mean pairwise same-message probability was 42.82%. The messages were substantially
sensitive to nuisance realization.

The complete `P(message | source,target,event)` and
`P(message | family,source,target,event)` tables and every raw message are supplied.

## 12. Descriptive decoding

Decoders were fitted on even candidate blocks and evaluated on odd blocks, so no
candidate rendering crossed the split.

| Target | Whole-message held-out accuracy |
|---|---:|
| Source | 4,057/8,160 = 49.72% |
| Target | 4,680/8,160 = 57.35% |
| Event | 5,357/8,160 = 65.65% |
| Ordered pair | 2,630/8,160 = 32.23% |
| Complete meaning | 1,944/8,160 = 23.82% |

Per-token-position decoding is preserved in `metrics.json`. No position is called
a source, target, event, or other semantic register. These are associations only.

## 13. Hybrid-counterfactual readiness

Semantic support is present:

- Family S contains base `(s,t,e)`, source hybrid `(u,t,e)`, target hybrid
  `(s,u,e)`, and event reversal `(s,t,¬e)`;
- source donor `(u,s,e)` and target donor `(t,u,e)` occur as ordinary meanings in
  other blocks;
- Family R supplies a joint endpoint-reversal control, which cannot distinguish
  selective replacement from donor copying.

Natural IID performance on these families was inadequate: source replacement
53.85%, target replacement 56.76%, event replacement 51.18%, and Family-R joint
reversal targets 40.49%. The support exists structurally, but the trained model is
not behaviorally ready for causal intervention.

## 14. PASS-gate table

| Preregistered criterion | Result |
|---|---|
| Generator/evaluation integrity | PASS |
| S and R IID joint ≥95% | FAIL — 56.73%, 41.92% |
| S and R common-composition joint ≥95% | FAIL — 24.58%, 24.99% |
| Four primary simultaneous lower bounds >90% | FAIL |
| Six ambiguity points ≥95% | FAIL |
| Six ambiguity lower bounds >90% | FAIL |
| Every held-out pair/event cell ≥90% | FAIL — minimum 19.35% |
| No-message interval within 22–28% | PASS — exactly 25% |
| Independent-message interval within 22–28% | PASS — exactly 25% |
| Candidate-only intervals within 22–28% | PASS — exactly 25% |
| Candidate permutation agreement 100% | PASS |
| Object prediction agreement ≥99% | FAIL — 96.18% |
| Object permutation accuracy loss ≤1 point | PASS — 0.13 point |

## 15. Scientific interpretation

Task v2 is valid and removes the documented candidate shortcuts, but this seed did
not solve it. Family S is easier than Family R, yet neither is reliable. The
channel has ample nominal capacity and an explicit six-message zero-error oracle,
so raw channel capacity is not the limiting explanation.

The evidence is consistent with a joint-optimization bottleneck: under the fixed
architecture, hard channel, estimator, mixed candidate families, and 8,000-update
budget, speaker messages remained nuisance-sensitive and did not reliably transmit
the complete meaning. It is also consistent with an architecture/task mismatch or
brittle listener conditioning. The single run cannot distinguish these causes.

The result does not show that source, target, or event are absent internally; nor
does it show holistic, compositional, or variable-like encoding. Behavioral
acquisition failed before those questions became meaningful.

## 16. Failure analysis

The most immediate failure is optimization/learnability, not compositionality
alone: IID joint performance is only 49.33% overall. Factor accuracies of roughly
70% and Family-S performance above Family R show partial learning, but ambiguity
tests remain near 50% and messages are inconsistent across geometry.

The common/new/full held-out results are indistinguishable at approximately 25%,
so inventory novelty is not the limiting held-out factor. Instead, the learned
partial protocol fails on all unseen H–H active relations.

No hyperparameter, architecture, channel, split, or training extension was tried.
Any attempt to separate insufficient optimization budget, hard-channel credit
assignment, and model mismatch must be preregistered as a new experiment.

## 17. Recommendation

**STOP Task-v2 mechanistic progression.** Do not train additional exploratory
seeds and do not perform activation patching on this checkpoint.

If the project continues, the next scientifically clean experiment should be a
new behavioral learnability diagnostic rather than a rescue sweep—for example,
holding Task v2 fixed while supplying an oracle/supervised semantic message to
isolate listener/environment learnability from emergent-channel optimization.
That change was not run here and would require its own preregistration.

All code, pretraining audits, oracle mappings, checkpoints, metrics, candidate
blocks, bootstrap indices, messages, predictions, cells, and permutation mappings
are preserved in the accompanying bundle.

