# Task v1 online-geometry diagnostic

**Predeclared outcome:** **PARTIAL**  
**Run:** one seed (1701), exactly 8,000 steps  
**Mechanistic interpretability:** not performed  
**Communicating checkpoint SHA-256:** `97886054e86fc937eb07181009074873a8ab29e2ded16725845ea26a03541479`

## Executive conclusion

Changing repeated finite training geometries to deterministic online geometry
caused a large, factor-specific improvement on fresh examples of training meanings.
IID orientation rose from 53.09% to 96.55%, while IID event remained strong at
96.03%. Joint IID accuracy rose from 48.67% to 92.94%.

The improvement did not transfer to held-out endpoint pairs. Compositional
orientation was 49.64% and compositional joint accuracy was 44.21%, essentially
unchanged from the previous 51.50% and 44.79%. The predeclared PASS conditions
therefore fail. Online nuisance variation removed the main IID failure but did not
produce compositional orientation across unseen identity pairs.

## 1. Frozen design and changed factor

Seed, model initialization, 118,208-parameter architecture, vocabulary eight,
message length four, hard straight-through communication, optimizer, learning
rate, batch size 128, loss, temperature schedule, 8,000-step budget, Task v1
semantics, candidate construction, held-out pairs, and all fixed evaluation sets
were preserved. The only experimental change was generating new nuisance geometry
at every training presentation.

Training examples were keyed by `(1701, phase, step, example index)`. Each
three-batch block contained eight occurrences of every one of the 48 training
meanings in deterministic shuffled order. Candidate position and distractor were
balanced within meaning.

The online generator's 12,288-example pretraining audit found:

- every meaning exactly 256 times;
- event and orientation exactly 6,144/6,144;
- every identity exactly 2,048 times as source and as target;
- candidate position and distractor imbalance within meaning: zero;
- ambiguous trajectories: zero;
- held-out semantic leakage: zero;
- duplicate/colliding complete geometries: zero;
- randomized source/target object slots (largest count spread: 60/12,288).

## 2. Primary outcomes

Counts and percentages use the same hard-message inference and listener scoring as
Pilot 0.

| Evaluation set | N | Joint correct | Event correct | Orientation correct |
|---|---:|---:|---:|---:|
| Online fresh training evaluation | 3,072 | 2,791 (90.85%) | 2,929 (95.35%) | 2,912 (94.79%) |
| Fresh geometries, training meanings (seed 1751) | 3,072 | 2,851 (92.81%) | 2,958 (96.29%) | 2,955 (96.19%) |
| Frozen validation (seed 1721) | 3,072 | 2,802 (91.21%) | 2,927 (95.28%) | 2,922 (95.12%) |
| Frozen IID test (seed 1731) | 3,072 | 2,855 (92.94%) | 2,950 (96.03%) | 2,966 (96.55%) |
| Frozen compositional test (seed 1741) | 3,072 | 1,358 (44.21%) | 2,752 (89.58%) | 1,525 (49.64%) |

### Direct predeclared comparison

| Metric | Fixed geometry | Online geometry | Change |
|---|---:|---:|---:|
| IID joint | 48.67% | 92.94% | +44.27 points |
| IID event | 91.47% | 96.03% | +4.56 points |
| IID orientation | 53.09% | 96.55% | +43.46 points |
| Compositional joint | 44.79% | 44.21% | −0.59 points |
| Compositional event | 87.66% | 89.58% | +1.92 points |
| Compositional orientation | 51.50% | 49.64% | −1.86 points |

The causal diagnostic supports finite exposure as the major cause of poor IID
orientation. It also falsifies the stronger expectation that online variation
alone would yield orientation that composes across held-out endpoint pairs.

## 3. Prediction classes

| Set | Correct | Orientation reversal | Event reversal | Both reversals |
|---|---:|---:|---:|---:|
| Online fresh | 2,791 | 138 | 121 | 22 |
| Fresh training meanings | 2,851 | 107 | 104 | 10 |
| Validation | 2,802 | 125 | 120 | 25 |
| IID | 2,855 | 95 | 111 | 11 |
| Compositional | 1,358 | 1,394 | 167 | 153 |

The compositional error distribution is diagnostic at the behavioral level:
orientation reversals alone account for 45.38% of all examples.

## 4. Communication controls

On the frozen IID set:

| Condition | Joint | Event | Orientation |
|---|---:|---:|---:|
| Normal | 2,855/3,072 (92.94%) | 2,950 (96.03%) | 2,966 (96.55%) |
| No message | 775/3,072 (25.23%) | 1,544 (50.26%) | 1,536 (50.00%) |
| Exact independent replacement | 2,363,713/9,437,184 (25.0468%) | — | — |

Independent replacement evaluated every observed message occurrence against every
IID candidate set. The empirical donor product was exactly balanced: 4,718,592
same-event and 4,718,592 opposite-event pairs; the same exact counts held for
orientation. It included 196,608 same-meaning and 786,432 same-unordered-pair
pairs. It was therefore not anti-correlated like the old `roll(1)` control.

Independent prediction classes were nearly uniform: correct 2,363,713;
orientation reversal 2,356,306; event reversal 2,355,808; both reversals 2,361,357.

## 5. Protocol diagnostics

The IID messages used three symbols:

| Symbol | Count / 12,288 | Frequency |
|---:|---:|---:|
| 0 | 1,556 | 12.66% |
| 1 | 6,703 | 54.55% |
| 6 | 4,029 | 32.79% |

Eight whole messages occurred, with entropy 2.4914 bits:

| Message | Count |
|---|---:|
| 1111 | 749 |
| 1011 | 745 |
| 6066 | 703 |
| 6116 | 454 |
| 6166 | 261 |
| 6016 | 69 |
| 6111 | 52 |
| 6011 | 39 |

The complete `P(message | event, orientation, unordered endpoint pair)` table is
provided in the raw CSV. Across online-fresh examples of a meaning, mean dominant-
message frequency was 83.12% and mean pairwise message agreement was 72.46%; the
dominant fraction ranged from 45.16% to 98.41%. The protocol is more nuisance-
consistent than the fixed-geometry model but not deterministic.

### Held-out behavioral decoding

A categorical majority decoder was fitted on 1,536 IID examples and evaluated on
1,536 examples alternated within meaning.

| Input | Event correct | Orientation correct |
|---|---:|---:|
| Whole message | 1,479/1,536 (96.29%) | 992/1,536 (64.58%) |
| Token position 0 | 1,479 (96.29%) | 761 (49.54%) |
| Token position 1 | 768 (50.00%) | 992 (64.58%) |
| Token position 2 | 1,238 (80.60%) | 804 (52.34%) |
| Token position 3 | 1,479 (96.29%) | 759 (49.41%) |

These are descriptive associations. They do not establish position-specific
registers, semantic variables, or mechanisms. The gap between 64.58% global
orientation decoding and 96.55% task orientation suggests that message use is
conditioned on the candidate pair or otherwise context-dependent; the conditional
table must be consulted before proposing a coding interpretation.

## 6. Candidate-only online control

The separately trained 58,081-parameter permutation-equivariant candidate-only
model received the same online candidate stream for 8,000 steps but no source or
message. Its loss remained at four-way chance (`log(4) ≈ 1.38629`).

| Set | Correct | Accuracy |
|---|---:|---:|
| Online fresh training evaluation | 800/3,072 | 26.04% |
| Fresh training meanings | 764/3,072 | 24.87% |
| Validation | 744/3,072 | 24.22% |
| IID | 759/3,072 | 24.71% |

The finite-set candidate-only memorization route has been removed. There is no
evidence of a persistent fresh-set candidate shortcut.

## 7. Example-level permutation invariance

After inverse-mapping candidate permutations, predictions agreed for exactly
3,072/3,072 examples and accuracy remained 2,855/3,072.

Under independent object-slot permutations, predictions agreed for 3,026/3,072
(98.50%). Accuracy changed from 2,855/3,072 (92.94%) to 2,853/3,072 (92.87%).
Source messages agreed for 2,974/3,072 examples (96.81%). This materially improves
on the fixed model's 71.06% prediction agreement, though it is not exact invariance.

## 8. Predeclared classification

### Conditions passed

- IID joint ≥75%: **92.94%**
- IID orientation ≥75%: **96.55%**
- IID event ≥85%: **96.03%**
- no-message and independent-message controls: non-informative
- candidate-only fresh evaluation: approximately chance
- no new leakage/evaluation artifact detected
- object-slot invariance: materially improved to 98.50%

### Condition failed

- compositional joint ≥60%: **44.21%**

Therefore the result is **PARTIAL**, not PASS. Pilot 1 is not yet justified for
claims about selective endpoint computation. The evidence supports robust event
communication and robust orientation for trained endpoint pairs, but not an
endpoint operation that transfers to unseen pairs.

## 9. Scientific interpretation and next decision

Online geometry was a successful causal diagnostic. It strongly supports this
narrow conclusion: repeated finite exposure prevented orientation from
generalizing to fresh nuisance geometries of familiar meanings. The candidate-
only contrast strengthens that inference.

The remaining limitation is semantic rather than geometric. Task v1 permits a
pair-conditioned or identity-specific orientation solution that succeeds on every
fresh geometry for trained pairs but reverses orientation at chance on held-out
pairs. Behavioral results do not determine the exact representation or mechanism.

No further experiment was run. Before any new training or mechanistic work, the
research team must decide whether the scientific target requires compositional
endpoint replacement. If it does, Task v1's current holdout and candidate design
need a separately predeclared revision; that would be a new experiment, not a
continuation of this one.

## 10. Reproducibility

Communicator training took 333.79 seconds (23.97 steps/s) on CPU. The candidate-
only control took 283.21 seconds. Source, predeclaration, generator validation,
checkpoints, complete metrics, loss traces, all predictions, conditional message
tables, consistency tables, decoder split indices, and inverse-mapped permutation
predictions are preserved in the accompanying bundle.

