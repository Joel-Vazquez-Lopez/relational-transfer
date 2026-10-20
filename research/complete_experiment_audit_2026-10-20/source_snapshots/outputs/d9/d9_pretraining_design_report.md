# D9 — prospective comparison-structure intervention: frozen pretraining report

**Status:** pretraining symbolic audit PASS; preregistration and implementation frozen; **no D9 model initialized or trained**. This report is a prospective design and feasibility audit, not a scientific outcome. D8A.3 is closed, D8B/D8C were not run, and protected D5 seeds 1704–1706 were not inspected. D1–D8 artifacts were not changed.

## 1. Exact design and estimand

The positive support contains 32 ordered source–target–event meanings. In the *target* arm, source/partner identities 0–3 pair with target endpoint 4 or 5. In the *source* arm, source endpoint 4 or 5 pairs with target/partner identities 0–3. Both events, 0 and 1, occur for each ordered pair. The endpoint substitution is 4↔5, in either source or target role.

For each arm, represent the eight ordered pairs as vertices: 0–3 for endpoint 4 in partner contexts 0–3, and 4–7 for endpoint 5. A weighted comparison edge `(u,v)` denotes a four-candidate hyperedge containing `(u,event 0)`, `(u,event 1)`, `(v,event 0)`, `(v,event 1)`. The selected query is each member exactly once for every candidate order and nuisance repetition. Thus every edge simultaneously specifies the semantic query, candidate set, correct answer, endpoint/partner relation, role, event, and weight. The exact 16-edge multisets for each condition are frozen in the preregistration and implementation. Every vertex has degree four in both conditions.

The COHERENT condition places its three focal 4↔5 comparisons once in each of three training partner contexts; REWIRED places all three in one training partner context. The fourth context has **zero focal comparison edges in both conditions**. Context labels rotate by paired seed. Each arm otherwise has eight opposite-endpoint-role edges and five both-role edges. This is a prospective intervention on *which meanings are compared*, not which positive meanings exist or how often they are correct.

The target causal estimand is the paired COHERENT−REWIRED effect of these **two fixed weighted comparison distributions**. It does not isolate a scalar coherence coefficient from every other changed pairwise co-occurrence: changing the comparison graph necessarily changes part of the foil tensor. Both conditions use the same oracle semantic message, unchanged D5 listener architecture, renderer, optimizer, update budget, and common per-seed schedule index order. The speaker is unused. There is no D6 repeated-PAD neutrality control.

## 2. Matching feasibility and exact inventory

Each condition/seed has 32 edge occurrences (16 per arm), four cyclic candidate orders, and four possible selected queries per order: **512 symbolic examples**. Repeating these with 2,000 fresh nuisance geometries gives **1,024,000 training examples/model**, exactly 8,000 updates × batch size 128. The full seed-keyed schedule was audited as a bijection for all eight seeds.

| Quantity, per model unless noted | COHERENT | REWIRED | Status |
|---|---:|---:|---|
| Positive meanings | 32 | 32 | Same set |
| Correct-query occurrences per positive meaning | 32,000 | 32,000 | Exact |
| Candidate occurrences per meaning | 128,000 | 128,000 | Exact |
| Negative/foil occurrences per meaning | 96,000 | 96,000 | Exact |
| Positive occurrences per ordered source–target pair | 64,000 | 64,000 | Exact |
| Event occurrences, each event | 512,000 | 512,000 | Exact |
| Source-role/target-role arm occurrences, each arm | 512,000 | 512,000 | Exact |
| Correct answers at each of four positions | 256,000 | 256,000 | Exact |
| Each candidate meaning at each position | 32,000 | 32,000 | Exact |
| Event foil for each query meaning | 32,000 | 32,000 | Exact |
| Candidate-scene object-inventory types | 16 | 16 | Each type occurs 256,000 times; exact |
| Distractor choices per meaning | 4 | 4 | Each choice occurs 500/2,000 geometry repetitions; exact |
| Directed query→foil exposures | 3,072,000 | 3,072,000 | Total exact, cells not exact |
| Focal comparison edges per arm | 3 | 3 | Total exact, context allocation differs |
| Coherence statistic per arm | 1 | 0 | Deliberate difference |

The 96 exact-matching checks across eight paired seeds passed. Source identity, target identity, endpoint identity within candidates, event-cue, ordered-pair, candidate-negative, answer-position, and role-frequency comparisons are included in those audits. The per-identity counts need not be uniform, but each is equal across conditions. Renderer smoke tests covered 512 rendered cases, including four-label reuse. The corresponding raw tables are in the audit directory.

**What cannot match:** the complete candidate-hyperedge multiset and the complete directed query→foil co-occurrence tensor. A partner-specific focal comparison count is itself a cell of that tensor. Exact equality of the full tensor would therefore force equal focal occupancy and equal coherence, contradicting a nonzero intervention. Across each paired seed, **176 directed query→foil cells differ**, their L1 difference is **1,536,000** exposures, and 56 same-event endpoint-foil cells differ; event-foil cells differ in **zero** positions. Of 16 edge occurrences per arm, ten are common and six differ; COHERENT has 16 unique edge types and REWIRED 14 because one focal edge repeats three times. These are intended and unavoidable comparison-structure changes, but changes to nonfocal pairwise exposures are a residual attribution risk. A positive D9 result would identify an effect of the frozen *graph rewiring*, not prove that focal coherence alone is the unique sufficient statistic.

## 3. Frozen comparison-coherence statistic

For an arm and seed, let `n_c` be the number of direct 4↔5 focal comparison edges in partner context `c`, and `n=Σ_c n_c=3`. Before training, define

`C = (n² − Σ_c n_c²) / [n(n−1)]`,

the proportion of ordered pairs of focal comparison exposures occurring in distinct contexts. COHERENT has occupancy `(1,1,1,0)` and `C=1`; REWIRED has `(3,0,0,0)` after context rotation and `C=0`. The required difference is exactly one in both arms for every seed. This statistic uses only the typed comparison distribution, never learned states or observed performance.

A five-level quarter-step graded manipulation is not feasible with exactly three focal edges and the common held-out context while preserving focal exposure: the available nontrivial intermediate occupancy `(2,1,0,0)` has `C=2/3`. Binary is the cleaner matched design. A support-absent × coherence factorial would remove focal positive meanings/comparisons and change D9's estimand; D5 already addresses support, so no factorial was added.

## 4. Structural shortcut audit

For any fixed candidate hyperedge, cyclic order, nuisance repetition, and **fully rendered four-candidate tensor**, the four possible queries each occur exactly once with a distinct correct position. Hence `P(Y=j | tensor, condition)=1/4` for all four positions. This constructive equality is stronger than position-only balance; no candidate-only function can exceed 25% Bayes accuracy on the frozen distribution. The exact symbolic enumeration checked **7,216 candidate-only feature classes** covering candidate meanings, ordered/repeated endpoint pairs, event pattern, object inventory, position, endpoint identity, and pair multiplicity. The maximum exact Bayes accuracy was **0.25**. The proof also covers any feature computable from the full candidate tensor, including deterministic combinations not separately listed. No learned listener or shortcut classifier was fitted.

## 5. Evaluation families and exact primary metrics

The familiar family evaluates all 32 condition-specific training edge occurrences using fresh evaluation geometry: 16 nuisance repetitions × four orders × four queries = **8,192 examples/model**. It is a learning gate and local-comparison diagnostic, not the primary generalization claim. The held-out family uses the direct 4↔5 focal pair in the fourth partner context absent from **both** training graphs: two arms × 128 fresh repetitions × four orders × four queries = **4,096 examples/model**. All component identities, events, positive meanings, and the 4↔5 distinction occur during training; only that *direct comparison in that context* is withheld. The evaluation RNG namespace differs from training and is shared between paired conditions. This separates trained-comparison improvement from cross-context recombination, though the held-out query-to-foil configuration is intentionally unfamiliar.

The exact scorer is `s(q,v)=10 qᵀv` on individually L2-normalized natural final query and candidate states. For endpoint A=4 and B=5, define `Δq=q_B−q_A`, `Δv=v_B−v_A`. The signed interaction

`D=s(q_B,v_B)−s(q_B,v_A)−s(q_A,v_B)+s(q_A,v_A)=10 ΔqᵀΔv`.

Positive `D` means matched endpoint identities are jointly favored over crossed identities. Pure score-algebra checks over 1,000 random normalized four-state cases had maximum discrepancy **2.66×10⁻¹⁵**. A future trained-model gate will recheck the literal four scores against the dot-product expression; the pretraining check is not a substitute for that gate. Metrics use natural states only, no patching or donor transfer.

The three frozen primary endpoints, each averaged equally across arm and event and then paired by seed, are:

| Primary | Model-level metric | COHERENT−REWIRED minimum | Equivalence half-width |
|---|---|---:|---:|
| P1 candidate stability | Mean of six pairwise cosines among the four context-specific, 96-nuisance-average `Δv` vectors | +0.10 | ±0.05 |
| P2 exact interface coupling | Mean signed `D` on 96 matched natural nuisance pairs in the held-out context | +0.75 score units | ±0.50 |
| P3 held-out endpoint behavior | Correct versus same-event endpoint foil on 4,096 held-out examples, ties worth 0.5; chance 0.5 | +0.10 accuracy | ±0.05 |

Query-side `Δq` stability uses the same context-pair cosine but is secondary to keep the confirmatory family to three. Additional frozen secondaries include query/candidate contrast norms, arm/event/context breakdowns, familiar and held-out four-way accuracy, trained focal-comparison behavior, event-pair accuracy, event-foil margin, and query/candidate alignment. The event selectivity diagnostic requires its paired 90% interval for event-pair accuracy to be within ±0.05 to *claim endpoint selectivity*; failure does not invalidate D9 but weakens that interpretation. No layer/head/neuron search is included.

## 6. Statistical and validity plan

There are **eight fresh paired seeds, 1901–1908**, two conditions each, for 16 models if separately authorized. For each of the three primary paired differences, the directional test is an exact one-sided paired sign flip over all `2⁸` sign assignments, with Holm familywise correction at 0.05. The frozen 10,000-resample paired-seed bootstrap (RNG seed 9901) provides a percentile 95% lower confidence bound, which must exceed zero for directional success. Each primary also needs at least **7/8** strictly positive seed-pair differences and its minimum mean effect. D9-A needs all three; no example-level pseudoreplication. D9-D equivalence requires all three 90% paired-bootstrap intervals to lie wholly within their metric-specific ±bounds, never merely nonsignificance. Per-seed rows, mean/median, Cohen `d_z`, two-sided 95% intervals, unadjusted/adjusted exact p-values, and all failures will be reported.

Critical gates, frozen before model initialization:

1. Distribution/intervention/exposure: all exact audits remain true; vertex degree four; 16 edges/arm; focal counts three; coherence 1 versus 0; fourth-context focal pair absent in both.
2. Candidate-only/position: exact Bayes accuracy 0.25 for every symbolic class and full-tensor construction; every answer position 256,000 times/model.
3. Familiar learning: **every one** of 16 final checkpoints must score ≥0.90 four-way accuracy on the 8,192-example fresh familiar family.
4. Permutations: on 1,024 frozen held-out control examples/model, inverse-mapped candidate-permutation prediction agreement must equal 1.0 and object-slot-permutation example-level agreement must be ≥0.99.
5. Score/integrity: trained-model literal score-interaction algebra discrepancy ≤1e−5; exactly 16 final-update-8,000 checkpoints; frozen hashes recorded and unchanged through evaluation; no restarts or outcome-based checkpoint choice; disjoint train/eval RNG namespaces.

Any critical failure is **D9-INVALID**. Event selectivity is reported separately, not an invalidity gate. The repeated-PAD neutral-message control is omitted because D6 showed it was not a valid neutrality intervention, whereas D9 has an exact candidate-only balance proof.

Interpretation priority is `D9-INVALID`, `D9-A` (all three primaries pass), `D9-B` (stability and/or coupling passes, behavior does not), `D9-C` (behavior passes without both internal primaries), `D9-D` (all three are within equivalence bounds), then `D9-E` (valid mixed/ambiguous outcome). D9-B/C/E preserve negative and discordant results; none triggers protocol repair.

## 7. Frozen artifacts, integrity, and recommendation

| Artifact | SHA-256 |
|---|---|
| `configs/d9_comparison_structure_prereg.json` | `ded7687fcefa112c6310ddf22573b734224a562fdd9ad9f0fe95ae8098b52b73` |
| `src/d9_comparison_structure.py` | `4b4c046d514e057746d6f23ae31936b47c4a471f557d08762505e4d9245c0d2d` |
| `src/d9_foil_audit.py` | `31a8371fbd007411f3d1062d168f320ea34c89e60c4c94771d5e1f66b6a2a9c5` |
| `results/d9_comparison_structure/audit/audit_summary.json` | `1f34dc1852f9f18e2bd673cba58dee245e76b91336891dbc347aa8a6e482cf87` |

The implementation's frozen-verification command passed without constructing a model. The audit directory contains raw graph, pool, matching, candidate-only, object-inventory, renderer-smoke, schedule, and foil-co-occurrence tables. A training entry point is prepared but requires a separate explicit authorization marker containing the preregistration hash. No D9 run/checkpoint directory exists. The first symbolic audit snapshot was retained separately as `audit_initial`; the final frozen audit is `audit`.

**Unresolved risks.** The graph intervention necessarily changes nonfocal query–foil co-occurrences and the exact candidate-set multiset, so even a positive result supports the causal effect of this *particular comparison reorganization*, not a unique mechanism or universal coherence law. The held-out direct comparison can favor local interpolation in either direction; paired geometries and a common absent focal edge control exposure but cannot make the models' learned similarity functions identical. Query/candidate natural-state metrics may not align with behavior; that is an allowed D9-B/C/E outcome. Familiar-learning or permutation gates may fail; that is invalidity, not a license to tune. The prepared training/evaluation path has not been end-to-end exercised because that would initialize a model; any future implementation fault must be handled transparently under the frozen validity rules.

**Recommendation: YES, authorize the frozen 16-model training only if the intended claim is the narrower causal effect of the two fixed, exposure-matched comparison hypergraphs.** The symbolic feasibility and anti-shortcut gates pass, the held-out comparison is absent in both arms, and the principal residual confound is mathematically disclosed rather than hidden. Do **not** authorize if the desired claim requires isolating the scalar `C` while holding every query–foil exposure cell and candidate set fixed; that stronger contrast is impossible under this intervention. Await explicit user authorization; **stop now before model initialization or training**.
