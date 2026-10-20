# Discovery memo: quotient-breaking choice constraints and soft trajectory bias

**Date:** 2026-09-30. **Status:** discovery only. This memo adds D52–D59 to the existing post-P3.2 discovery record. It does not change the canonical P3.2-E endpoint. No protected seeds 1704–1706, P4 material, or other protected confirmation artifacts were accessed. Canonical source, schedule, and A-run results were read only; all new scripts, checkpoints, audits, and outputs are under this discovery directory. Each diagnostic has a plan and hash manifest.

## Why this run was chosen

The prior F result showed that changing training choice sets could rescue all three withheld H–H pairs, but generic difficulty remained a serious alternative. The first audit found another confound: F replaces a choice member before the correct member is selected. Thus some changes also alter the positive oracle meaning. D52 quantified this before any new training. Subsequent arms separated positive changes, foil changes, exact same-context comparisons, and training rows that make identity merger collide a correct candidate with a foil. These arms were selected adaptively in sequence; none is confirmatory.

## New results

### D52: F was not a pure negative-candidate intervention

The exact seed-1701 symbolic schedule has 1,024,000 rows. F changes 204,818 choice sets. Of these, **51,200 rows (5% of all training; 10% of S-family training) change the correct oracle meaning**; 153,618 change foils only. The positive-meaning distribution has total-variation distance 0.05 from A overall. The original memo's “queried relation *set* unchanged” remains true, but positive-meaning *frequencies* were not unchanged. See `d52_positive_shift.json`.

### D53: foil changes suffice; positive shifts alone do not

`F_foil_only` retains the 153,618 F rows on which the positive meaning was unchanged and reverts the 51,200 positive-changing rows to A. Its positive-meaning distribution equals A exactly. `P_positive_only` exactly matches F's positive-meaning distribution while omitting direct same-role H-versus-H candidate comparisons. Both started from the same seed-1701 state and indexed schedule, with 8,000 updates.

| Seed 1701 arm | Familiar S joint | Withheld S joint | Withheld R joint | H source/target query contrast |
|---|---:|---:|---:|---:|
| A archived | 0.995 | 0.826 | 0.487 | 0.037 / 0.056 |
| F archived | 0.997 | 0.962 | 0.989 | 1.019 / 0.930 |
| F foil only | 0.996 | 0.951 | 0.963 | 0.956 / 0.866 |
| P positive only | 0.996 | 0.775 | 0.453 | 0.052 / 0.057 |

The positive shift is therefore **not necessary** for a strong F-like outcome on this seed. It is also **not sufficient** under the P construction. P changes a full choice set to retain valid distinct candidates, so its failure is not a pure manipulation of positive frequency alone. F foil-only required 25,600 distractor repairs when reverted meanings made an archived distractor invalid; those repairs are logged.

### D54: same H exposure and similar initial scalar difficulty without direct comparison gives partial rescue

`M_context_mismatch` uses the same 153,618 modified rows and exact H source/target candidate marginals as `F_foil_only`. It keeps the swapped H identity in the negative candidate but changes its G endpoint, so there is **no same-target/event H-source or same-source/event H-target comparison**. Correct oracle meanings remain A's. On the same 4,096 sampled modified rows at the shared initialization, cross entropy is 1.6336 for F foil-only and 1.6278 for M; mean correct-versus-hardest-foil margins are −0.7088 and −0.7044. These are close scalar starting points, **not** matched gradient dynamics.

M reaches withheld S/R joint accuracies **0.893/0.811**, with source/target query contrast **0.652/0.507**. Direct aligned comparison is therefore **not necessary for any improvement** over A. It appears to produce a stronger and more reliable effect than this mismatched H foil in the tested seed. M's changed G endpoint and distractor repairs remain potential differences beyond the intended comparison.

### D55: the alternatives select different developmental trajectories

All D55 arms share an initial seed-1701 source query contrast of 0.487. By update 8,000, F foil-only has 0.956, P positive-only 0.052, and M 0.652. P compresses the initially present distinction. M initially compresses it, then partially recovers late. F foil-only amplifies it early. Withheld R accuracy follows these broad trajectories, but the query readout is downstream of learning and cannot by itself explain acquisition. See `d55_developmental.csv`.

### D56: quotient-breaking rows are a sharper task-level distinction

Let \(q_{12}\) identify identities 1 and 2 in a candidate meaning. A **positive–foil quotient collision** is a training row for which a foil has the same relation as the correct candidate after applying \(q_{12}\). This is a computable property of the labeled choice set. F foil-only has **51,214** such rows (5.001% of the schedule), P and M have zero. On the same 8,192 sampled training positions, averaging the trained model's identity-1/2 message-symbol and candidate-input columns raises F foil-only cross entropy from **0.00682 to 0.05250**, mainly on the collision rows (tied collision-row CE 0.855). The tie penalty is only 0.00005 for P and 0.00121 for M. This supports a hard *task-level* need for identity separation in F foil-only. It is not a mathematical lower bound on the finite rendered scenes, because visual nuisance can differ even when symbolic meanings collapse.

### D57–D59: hard collision rows suffice across two discovery seeds; the rest are variable

K retains only positive–foil quotient-collision rows; L retains all other F foil-only modifications. Both keep A's exact positive meanings. They partition F foil-only modifications and use the same initialization and schedule within seed.

| Seed | Arm | Modified rows | Collision rows | Withheld S joint | Withheld R joint | H source/target query contrast |
|---|---|---:|---:|---:|---:|---:|
| 1701 | K collision only | 51,214 | 51,214 | 0.948 | 0.975 | 0.930 / 0.822 |
| 1701 | L noncollision only | 102,404 | 0 | 0.906 | 0.795 | 0.608 / 0.427 |
| 1702 | A archived | 0 | 0 | 0.818 | 0.489 | — |
| 1702 | K collision only | 51,192 | 51,192 | 0.947 | 0.925 | 0.906 / 0.816 |
| 1702 | L noncollision only | 102,419 | 0 | 0.856 | 0.558 | 0.152 / 0.288 |

K alone nearly recovers the full F foil-only outcome in seed 1701 and strongly rescues a second permitted discovery seed. It improves all three withheld H–H pairs in seed 1701, even though the collision concerns identities 1/2. L has a much less reliable effect: substantial in seed 1701, small in seed 1702. D59 shows K source H contrast around 0.66 at update 500 and 0.91–0.93 at update 8,000 on both seeds. L contrast is around 0.54–0.55 at update 500, declines by update 2,000 on both, then rebounds to 0.61 in seed 1701 but continues to 0.15 in seed 1702. The late rebound is not a necessary consequence of noncollision H foils.

![K and L discovery trajectories across two seeds](d60_collision_partition_trajectories.png)

## What survived, what failed

**Survived as discovery-level evidence:** a small subset of labeled choice comparisons makes merging two identities costly and reliably selects a computation that preserves their distinction, accompanied by broad H–H generalization in the two tested initializations. The relevant experience property is more precise than raw exposure, graph topology, or F's positive-meaning shift.

**Failed or narrowed:** F was not a foil-only manipulation; positive shifts were unnecessary for the rescue in seed 1701. Direct same-context comparison is not necessary for every improvement, because M and L can improve without quotient collisions. A stable “soft H co-occurrence route” did not replicate cleanly: L was much weaker in seed 1702. Initial scalar loss/margin similarity does not imply matched training dynamics.

**Still unknown:** whether quotient-collision pressure predicts outcomes on untouched topologies, architectures, or next-token language training; whether an optimizer-level dynamical quantity predicts the variable M/L path; whether the learned query/candidate organization causally mediates the broad H–H transfer; whether generic hard-negative pressure could reproduce K's effect under a valid matched design. All D52–D59 training is adaptive discovery, not independent confirmation.

## A sharper candidate principle and its limit

For an equivalence relation \(q\) over input identities, define a task-level collision rate

\[
K_q(D)=\Pr_{(m,C,y)\sim D}\left[\exists j\ne y:\;q(C_j)=q(C_y)\right].
\]

This says how often the labeled choice task itself rejects a merged description of two candidates. In this listener, setting \(K_{12}\) to about 0.05 is enough to preserve/amplify the 1/2 distinction and broadly change unsupported behavior in two discovery seeds. But \(K_q=0\) does **not** predict collapse: E and the M/L arms show that optimization can favor separation even when a merged task solution remains possible. Thus \(K_q\) is a promising **sufficient-route candidate in the tested system**, not the general acquisition law or a transformer-native grammar. It is a property of supervised choice sets; a language-model analogue must be defined in terms of how a relational distinction changes the experienced next-token distribution, not by importing candidate foils literally.

A deeper candidate is **stability of the merged computational solution under the actual learning dynamics**. Hard quotient collisions make that solution costly. Without them, a nearly fitting merged solution may be stable under one data/optimizer path and unstable under another. The E route and the seed-dependent L route fit this possibility qualitatively. The previous D47 result warns against reducing it to one-step antisymmetric gradient magnitude. This stability account is a hypothesis, not an identified mechanism.

## Next discriminating experiment

Before any new prospective confirmation, audit a three-arm task with identical positive oracle meanings and sampled schedule: K-style quotient-breaking foils; H-bearing but no-collision foils matched for identity marginals and *trajectory-level* difficulty as far as feasible; and an exposure-only control. Freeze all symbolic support and distractor audits before training. On fresh, genuinely untouched permitted models, predict both broad H–H behavior and early distinction growth. If the no-collision arm matches K under credible difficulty and exposure matching, the collision-specific account fails. If K wins but the broader setting changes, define the boundary rather than retuning.

In parallel, the more important mechanism experiment should start from matched tied or nearly tied checkpoints and release tiny controlled H-antisymmetric perturbations under A, E, K, M, and L schedules. Compare **finite-time growth/decay**, optimizer-state dependence, and later off-support behavior, not only initial gradient norm. This can test whether a data-dependent stability law predicts both the hard and soft routes. Only after a frozen rule survives new synthetic worlds should it be carried to controlled language.

Relevant nearby primary work includes [feature suppression and class collapse in contrastive learning](https://proceedings.mlr.press/v202/xue23d.html), [the role of negatives in contrastive representation learning](https://proceedings.mlr.press/v151/ash22a.html), and [late feature learning as one account of grokking](https://arxiv.org/abs/2310.06110). These are conceptual comparisons, not evidence for this system's mechanism.
