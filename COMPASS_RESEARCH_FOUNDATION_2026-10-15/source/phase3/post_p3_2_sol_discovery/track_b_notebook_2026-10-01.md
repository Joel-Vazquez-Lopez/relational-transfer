# Track B discovery notebook — 2026-10-01

**Boundary.** P3.2-E is the canonical experimental endpoint. All B-D1 onward results here are adaptive discovery evidence. Only permitted A/B/C/D and exploratory K/L checkpoints were read; the new B-D15 model uses seed 3701. Protected seeds 1704–1706 and other confirmation material were not accessed. Canonical artifacts were not modified. Each diagnostic has a prior plan, script, outputs, and a SHA-256 manifest under its own `track_b_bd*` directory. Preflight hashes precede model forwards or training where stated. Results chosen after previous diagnostic outcomes are explicitly adaptive.

## Current question

Can a compact, predictive, causally meaningful description explain how transformer training forms, transforms, and reuses relational distinctions? A candidate operation must predict downstream function under repetition, composition, and transport. A patch that reproduces a score or an algebraic description of cosine readout is insufficient.

## B-D1: frozen operation candidate failed

The predeclared low-rank post-block-1 candidate-state V/Y maps failed the prospective composition gates on eight A/B/C/D × seeds 1701/1703 models. V² patched-score RMSE was 1.031× identity; V²Y was 0.979× identity and 1.007× additive. Predicted V⁴ drifted despite actual input/state closure. Approximate commutation did not confer correct outcomes. See `track_b_bd1/bd1_report.md`. This candidate was stopped, with no retrospective capacity or stage rescue.

## B-D2–D3: readout geometry is a strong null

B-D2 found that the 60×48 oracle-query bank is nearly full numerical rank but has mean effective rank 8.895. Top-16 query SVD reconstructs V²Y scores at RMSE 0.009 and captures 94.4% of candidate-vector delta energy. Held meanings are almost completely in the 48 trained-meaning span (relative residual 7.25×10⁻⁵). At random initialization (B-D3), effective rank is 1.5–1.8 and top-16 captures only 33–41% of candidate-delta energy. Training expands query geometry, yet low-rank score predictability can arise from the scorer and finite query bank without a reusable state operation.

## B-D4–D6: identity distinction uses a learned value relay

On 192 matched 1→2 source/target identity substitutions, K models have final candidate contrast 1.06–1.21 and score effect magnitude about 5.76–7.23; A models have contrast 0.04–0.07 and near-zero score effect. Patching the changed token after candidate block 0 into K restores roughly 0.96–1.00 of final contrast. Removing block-1 attention suppresses K contrast; removing its MLP largely preserves it (B-D4).

The global mean changed-token delta transports to new scenes remarkably well: B-D5 score RMSE is only 0.045–0.062× identity and vector RMSE 0.12–0.13× identity. It also transports to a held third identity. **The frozen operation gate nevertheless failed** because a 192-donor nearest-neighbor predictor was 15–38% better than the global map. Strong partial transport is not a passed compact-operation test.

B-D6 decomposed the exact block-1 attention-output difference. Around 99–101% of its directed change came from changed values under nearly fixed attention weights; the weight-change term was around −1 to +1%. A value-only hybrid recovered roughly 0.97–1.00 of the final effect. The B-D4 phrase “dynamic routing” was therefore misleading: attention is needed to carry token content, but its weights do not materially switch on this identity edit.

## B-D7–D10: composition and development bound the account

B-D7 fixed a 2×2 identity×event factorial before inspecting the joint outcome. Changed-token patching predicted the joint score strongly, but full-tensor additive state prediction was as good or better. The frozen token-composition gate failed. Generic local additivity remains a live null.

B-D8 showed random initialization already has identity information and some candidate contrast, yet the changed-token directed fraction is only about 0.17–0.27 and score consequences are tiny. Thus the localized K relay is acquired, even if the architecture supplies an initial identity distinction.

B-D9 tracked 36 existing K/L developmental checkpoints. In K, changed-token directed fraction rises from roughly 0.23–0.35 at step 500 to around 0.93–0.97 by step 2000 and near 1 by step 8000. L seed 1701 eventually develops a strong relay; L seeds 1702/1703 remain weak. An exploratory step-2000 rank check identified the eventual L1701 rebound, but gaps are modest and the seed trajectory was already part of discovery ecology. It is not independent prediction.

B-D10 used fresh identity pairs. A already has the same high-fraction token/value relay on observed pairs (changed-token fraction ~0.96–0.98; candidate contrast ~1.16–1.23) but compresses withheld H-H pairs (~0.06 contrast). K extends the pathway to targeted 1–2 (~1.13 contrast) and partly to untargeted 0–1/0–2 (~0.62–0.66). This favors recruitment or amplification of an existing pathway over invention of a new K-specific operator. Large directed patch fractions could still follow from large contrast; a full mechanistic mediation claim remains unwarranted.

## B-D11–D14: event confound, counterexamples, and cross-object transfer

B-D11 exposed a major world-level shortcut: an analytic rule using only source initial position and velocity predicts approach versus recede with 100% accuracy on 9,000 independently rendered canonical scenes. A source-only logistic model also reaches 100%. The renderer keeps the target near the central square while the source travels radially toward or away from it. Ordinary task success cannot establish learned comparison of two object trajectories.

B-D12 held source fixed and swapped same-identity target rows from independent canonical scenes. Only 4.7% of 192 swaps reversed the geometric event; 8.3% had weak geometric margin. Mirroring targets through the source reversed the event in all scenes but moved target starts from a max coordinate ~0.45 to ~2.05, far outside the renderer's target marginal. A and K models usually retained the original event preference under this OOD mirror. This is a robustness boundary, not proof of a strict source-only algorithm.

B-D13 selected 96 **clear**, geometry-only counterexamples before model forwards: source trajectory exactly fixed; target row independently sampled from the canonical target marginal; base and recombined distance-change margins at least 0.30 and 0.20, respectively; target starts remain within ±0.45. The recombinations violate the renderer's *joint* conditional support even though each object is individually in range. A models flipped event preference in 43–50% of these scenes; K in 18–49%. This falsifies a strict source-only description of the trained function. B-D13 post-block-0 target-token patches almost never transferred the flip (0–1%), while source-token patches transferred many (11–50%).

B-D14 replayed candidate block 0 exactly. The target input difference reaches the **source token** mostly through the target value in source-token attention. Replacing only that value under fixed baseline attention yields the same intervention as replacing all values, because no other raw token changed. It reproduces roughly 15–52% event-preference flips across A/K seeds; attention-weight-only hybrids produce at most ~1%. Exact projected attention-output decomposition assigns about 101–104% of the directed difference to values and −1 to −4% to weight changes. This is a concrete target→source→CLS information path, with value content transferred at blocks 0 and 1. It remains a path description, not an algebraic operation or a demonstration of systematic relational generalization.

## Live alternatives and current decision

1. **Two-object relation computation:** first-block target values enter a contextual source token; second-block values convey it to CLS. B-D13/14 give causal evidence for this path on target recombinations.
2. **Shortcut with incidental target sensitivity:** canonical training can succeed source-only; target sensitivity on recombinations may be off-support interpolation. This is fully compatible with B-D11 and low B-D13 accuracy.
3. **Generic distributed local learning:** static-ish attention mixing plus nonlinear MLP/readout might explain all patches without a compact relational operation. B-D5/B-D7 failures keep this live.

The next highest-information test was B-D15: train a *new, explicitly exploratory* listener in a source-balanced world where paired opposite-event scenes have exactly the same source trajectory. The new renderer and data were frozen and audited before training. This is a world change, not a new condition in the canonical experimental lineage.

## B-D15–D16: relational necessity changes the implementation

B-D15 kept the original listener architecture and fixed oracle-message format but changed the scene generator so source initial position and velocity are independent of event; the target lies ahead of or behind the direction of source motion. There are 600 exact-source paired scenes. Training and test event labels are balanced; the source trajectory is identical in each opposite-event pair. New seed3701 reached independent four-choice accuracy 0.9867 at step1000, 0.9983 at step2000 and 0.9992 at step3000, using 12,000 training and 2,400 test episodes. The world is thus learnable without the old source-only shortcut.

B-D16's evaluation plan was frozen after B-D15 task accuracy and before paired-scene model forwards. On the 600 exact-source pairs, the step3000 model chooses the right event-query side for 99.5% of base scenes and 99.3% of opposite-event scenes; preference flips in 99.17%. The old pathway does **not** dominate here. At post-block0, source-token patching flips 16.8%, target-token patching 41.5%, and all-nonCLS patching 81.3%. A first-block target-value-only hybrid into the source token flips 17.2%; block1 source-value-only flips 9.3%. These contrasts are already distributed by step1000 and become sharper by step2000/3000. Random initialization has only ~0.13 direct flip and near-zero event margin.

This is a direct mechanistic counterexample to treating the canonical-world target→source→CLS path as a universal event-computation grammar. It is compatible with a weaker, conditional principle: training organizes information through value-mediated attention, but where comparison occurs and how many token states jointly carry it depend on task support. That statement is still broad and may simply describe generic transformer learning; it is not yet a compact predictive law. B-D17 prospectively tests whether the distributed pattern repeats under a fresh exploratory initialization. Its thresholds were frozen before training.

## B-D17–D18: prospective boundary passes; seeds implement different paths

B-D17 trained a fresh exploratory seed3702 on the same frozen balanced-world data, with the mechanism-boundary thresholds fixed before training. It reached 0.9996 independent task accuracy and 0.9883 opposite-event accuracy on the 600 paired scenes. All five predeclared boundary criteria passed. Source-token and block0 target-value-into-source patches flipped only 0.015 and 0.038 of preferences; block1 source-value-only flipped 0.003. All-nonCLS patching flipped 0.942. Yet **target-token patching flipped 0.913**, compared with 0.415 for seed3701. Thus the weak old source path repeats, while token-level implementation varies substantially across initializations despite the same world and data.

B-D18 decomposed both final models. In seed3701, post-block0 event contrast norms were 1.31 CLS, 1.53 source, 3.12 target and 1.12 distractor; in seed3702 they were 0.88, 0.94, 2.92 and 0.79. Exact block1 CLS pre-MLP directed fractions were residual/value/attention-weight 0.247/0.257/0.497 for seed3701 and 0.114/0.293/0.593 for seed3702. These fractions are projections onto the actual change, not additive causal shares, though their exact underlying vector decomposition sums to the actual pre-MLP difference.

Value-only hybrids rarely flipped the event preference (~0.15 for all values in both seeds). Attention-weight-only flipped 0.097 in seed3701 and 0.508 in seed3702. Combining target values with changed weights flipped 0.690 and 0.963; combining all values with changed weights flipped 0.918 and 0.970. The functional effect is a nonlinear combination of keys/queries, values, and residual state. In seed3702, the target token itself approximates that combination unusually well. These two seeds support *implementation multiplicity*, not a unique localized event operator.

## B-D19–D20: hostile nulls break semantic specificity

B-D19 applied the same exact block1 decomposition to the original A/K models on B-D13 event-reversing target recombinations. Attention-weight changes accounted for directed pre-MLP fractions about 0.40–0.59 across all six checkpoints. Thus the contrast “original world uses values; balanced world uses routing” is **false**. B-D6's near-pure value result was for identity substitution, whereas B-D19 concerns target geometry. The edit type matters at least as much as the world.

B-D20 selected geometry-only, event-preserving target replacements for those same 96 bases. It aimed to match the raw 8D target-trajectory displacement size of the event-reversing edit. Matching was imperfect: mean L2 displacement 1.44 for nuisance versus 1.74 for event reversal, mean absolute mismatch 0.296. Across A/K seeds, nuisance edits still produced block1 CLS attention TV about 0.126–0.182, versus 0.199–0.350 for event reversals, and attention-weight directed fractions about 0.39–0.52, versus 0.40–0.59. Score preference also sometimes flipped under event-preserving edits (0.063–0.167). Geometry perturbation by itself therefore engages attention routing. The larger event-reversal effect cannot be assigned uniquely to semantic event change because edit size and movement direction remain confounded. The B-D20 control weakens semantic specificity but does not eliminate a possible event-sensitive component.

## Decision after B-D20

**Survives:** Training can amplify and localize an identity distinction into a value-mediated token relay on the original world; target geometry can causally affect event preference through cross-object attention; a source-balanced world forces successful models to use information beyond the source trajectory; two high-performing models can realize that computation through different token-level paths.

**Died:** The frozen B-D1 global V/Y state operator; a special changed-token identity×event composition law; universal dynamic attention routing for identity edits; a universal static-value event path; a strict source-only description of the original models; a world-level “shortcut→value, relational necessity→attention weights” dichotomy; and a unique localized event circuit across balanced-world seeds.

**Still unresolved:** Whether a constrained *functional* abstraction above token position predicts intervention outcomes across worlds and seeds; whether attention-weight changes encode the event decision specifically or merely track geometric displacement; whether different high-performing circuits can be grouped by an information-limited equivalence relation; and how training selects among them. The current evidence is all discovery, and the original renderer's shortcut limits generalization claims.

The next discriminating experiment needs a new world family that varies the availability of a source-only cue while keeping the joint relative-geometry distribution and permissible counterfactual edits matched. This is a programme-design decision, not another patch on the current scenes. A prospective model-level prediction should be frozen before training, including direct task behavior and a small number of functional intervention metrics. Without such a design, more original-world geometry edits are likely to accumulate correlated observations while leaving the same confound unresolved. No Track B principle here is ready for independent confirmation or language transport.
