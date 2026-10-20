# D9 — Prospective Comparison-Structure Intervention: final report

**Final validity: INVALID. Frozen decision category: D9-INVALID.** All 16 preregistered models completed their 8,000 updates and frozen evaluations, but **all 16 failed the critical object-slot prediction-invariance gate** (required agreement ≥0.99; observed 0.959961–0.986328). No model was retrained, repaired, selected, or extended. Under the frozen rules, no directional graph-effect conclusion is confirmatory, irrespective of the descriptive primary values below.

The preregistration, implementation, distributions, metrics, thresholds, seeds, tests, gates, and category logic were unchanged after model initialization. The only code repair and provenance rebinding occurred *before* model initialization and were separately authorized and recorded. The protected D5 seeds 1704–1706 were not inspected; D8B/D8C and D8 mechanism discovery were not reopened.

## Frozen design and interpretation boundary

The confirmatory intervention compares the two preregistered, exposure-matched weighted candidate-comparison graphs: COHERENT distributes three focal endpoint comparisons over three training partner contexts; REWIRED concentrates those three in one. The fourth focal comparison context is absent from both and supplies the held-out recombination test. Both graphs have the same 32 positive meanings, each meaning's frequency, marginal candidate/foil frequency, source/target/event/role/ordered-pair frequencies, answer-position and candidate-position frequency, object-inventory frequency, total focal comparison count, renderer, oracle channel, listener, optimizer, schedule size, and training budget. They necessarily differ in the complete query–foil co-occurrence tensor and candidate-set multiset. Thus even a valid positive result could identify only the effect of these **two fixed comparison distributions**, not prove that an abstract scalar “coherence” is the unique cause. The pretraining [design report](d9_pretraining_design_report.md) documents the combinatorial proof and exact differences.

## Model completion and validity gates

Every condition–seed has `status=COMPLETE`, a final-update-8000 checkpoint, a matching checkpoint SHA-256 and model-state hash before/after evaluation, 80 trace rows, 8,192 familiar evaluation rows, 4,096 held-out evaluation rows, 16 geometry rows, and 1,024 permutation-control examples. All 16 checkpoint metadata records bind the final preregistration and implementation hashes. The [per-run validity table](../../results/d9_comparison_structure/analysis/run_validity.csv) gives the exact checkpoint SHA-256 and every gate value.

| Critical gate | Result |
|---|---|
| Frozen preregistration/implementation hashes and pretraining audit artifact hashes | PASS |
| Distribution construction, 96 exact matching checks, degree/edge/coherence structure, shared absent held-out focal comparison | PASS |
| 32-meaning support and exact exposure; 512 symbolic examples × 2,000 geometries = 1,024,000 examples/model | PASS |
| Candidate-only full-tensor balance; 7,216 symbolic feature classes; exact Bayes accuracy 0.25 | PASS |
| Answer position: 128/512 symbolic examples per position, hence 256,000/1,024,000 training examples/model | PASS |
| Candidate object inventory, distractor cycle, renderer smoke, common full schedule bijections | PASS |
| All 16 final checkpoints, metadata, hashes, untouched model state through evaluation, exact evaluation sizes | PASS |
| Familiar learning: each model ≥0.90 four-way accuracy | PASS, 16/16; range 0.997681–1.000000 |
| Candidate permutation: inverse-mapped example-level agreement exactly 1.0 | PASS, 16/16 |
| Object-slot permutation: example-level agreement ≥0.99 (at least 1,014/1,024) | **FAIL, 0/16; range 983–1,010/1,024** |
| Literal scorer interaction identity: maximum error ≤1e−5 | PASS, 16/16; maximum 5.72205e−6 |

| Seed | COHERENT familiar | COHERENT object-slot count /1024 | REWIRED familiar | REWIRED object-slot count /1024 |
|---:|---:|---:|---:|---:|
| 1901 | 0.998779 | 1,010 | 0.999512 | 1,005 |
| 1902 | 1.000000 | 1,007 | 0.997681 | 1,007 |
| 1903 | 0.999512 | 989 | 0.999634 | 983 |
| 1904 | 0.999634 | 1,002 | 0.999634 | 1,002 |
| 1905 | 0.999512 | 1,006 | 0.999756 | 1,010 |
| 1906 | 0.999634 | 998 | 0.998901 | 998 |
| 1907 | 1.000000 | 1,003 | 0.998779 | 998 |
| 1908 | 1.000000 | 1,001 | 0.999512 | 993 |

This gate fails for every model, not merely for a borderline aggregate. Under the frozen `failure_rule`, D9 is INVALID. The following calculations are required transparent descriptions of completed runs, **not valid confirmatory tests of the hypothesis**.

## Frozen primary analysis, computed after all validity gates

Primary estimands are paired COHERENT−REWIRED differences over eight seeds. P1 is the average of six cross-context cosines of natural mean candidate endpoint effects, equal-weighted over source/target arms and events. P2 is the signed exact final-scorer interaction `D=10(q_B−q_A)ᵀ(v_B−v_A)` in the held-out partner context. P3 is held-out correct-versus-same-event endpoint-foil accuracy (chance 0.5; ties 0.5). Full condition values and exact per-arm/event counts are in [primary_paired_seeds.csv](../../results/d9_comparison_structure/analysis/primary_paired_seeds.csv) and [primary_cell_details.json](../../results/d9_comparison_structure/analysis/primary_cell_details.json).

| Seed | ΔP1 candidate stability | ΔP2 exact coupling D | ΔP3 held-out endpoint accuracy |
|---:|---:|---:|---:|
| 1901 | −0.118776 | +0.085894 | −0.001221 |
| 1902 | −0.181409 | +0.141785 | +0.081299 |
| 1903 | −0.268030 | −0.869141 | +0.041504 |
| 1904 | −0.351854 | +1.819385 | +0.044678 |
| 1905 | −0.062611 | +0.981310 | +0.001953 |
| 1906 | −0.096163 | −3.642241 | −0.166260 |
| 1907 | −0.300274 | −0.316820 | +0.026855 |
| 1908 | −0.246938 | −1.312228 | +0.025146 |

Frozen inference uses 256 one-sided exact paired sign assignments, Holm adjustment for three primaries, 10,000 paired-seed bootstrap resamples with RNG seed 9901, a positive 95% percentile lower bound, at least 7/8 positive seed pairs, and the metric-specific minimum mean difference. No example-level pseudoreplication was used.

| Primary | C mean | R mean | Paired mean Δ | Cohen dz | 95% paired CI | Positive seeds | Exact p / Holm p | Min meaningful Δ | Directional rule |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| P1 candidate stability | 0.678226 | 0.881483 | −0.203257 | −1.944792 | [−0.271183, −0.137222] | 0/8 | 1.000000 / 1.000000 | +0.10 | FAIL; opposite direction |
| P2 exact coupling D | 2.793934 | 3.182942 | −0.389007 | −0.236532 | [−1.495417, +0.616108] | 4/8 | 0.718750 / 1.000000 | +0.75 | FAIL |
| P3 held-out endpoint behavior | 0.871033 | 0.864288 | +0.006744 | +0.090396 | [−0.047336, +0.045472] | 6/8 | 0.417969 / 1.000000 | +0.10 | FAIL |

The frozen equivalence rule (90% paired CI wholly inside ±0.05 for P1/P3 and ±0.50 for P2) is met for P3 only: P1 [−0.260356, −0.147446], P2 [−1.308078, +0.470410], P3 [−0.037822, +0.041534]. Therefore the descriptive pattern would not be D9-D even if the object-slot gate had passed. The actual category remains D9-INVALID by priority. The [machine-readable primary analysis](../../results/d9_comparison_structure/analysis/primary_analysis.json) contains unrounded numbers and frozen decision-rule results.

## Frozen secondary and event-control summaries

The familiar four-way task was nearly solved in both conditions (COHERENT mean 0.999634, REWIRED 0.999176). Trained focal-comparison four-way accuracy was 0.999674 versus 0.998535. Held-out four-way accuracy was 0.870972 versus 0.863312; held-out endpoint-pair accuracy was 0.871033 versus 0.864288. These high within-condition values show learning and some held-out task performance, but **do not rescue the invalid graph contrast** or meet the P3 meaningful paired-effect criterion.

Natural query-side cross-context stability was 0.531905 versus 0.799949 (paired Δ=−0.268043), also opposite the prospective direction. Mean query endpoint-effect norm was 1.237037 versus 0.883051; candidate endpoint-effect norm was 1.146172 versus 0.844345. Larger effect norms in COHERENT did not imply greater cross-context directional stability. These are frozen secondary descriptions only, not a new mechanism claim.

Held-out event-pair accuracy was 0.999847 versus 0.999939, paired Δ=−0.00009155. The paired 90% bootstrap CI [−0.00024414, +0.00006104] lies within the preregistered ±0.05 event-control equivalence bound. This meets the *event-control bound* but cannot license an endpoint-selective formation claim because the critical object-slot gate failed. [Secondary per-model](../../results/d9_comparison_structure/analysis/secondary_per_model.csv), [paired](../../results/d9_comparison_structure/analysis/secondary_paired_seeds.csv), and [aggregate](../../results/d9_comparison_structure/analysis/secondary_analysis.json) tables preserve all frozen secondary values.

## Operational and artifact integrity

Training used exactly seeds 1901–1908 in both conditions: 16/16 completed, each with the 8,000-update final checkpoint. Two condition-specific launchers ran seeds in ascending order and stopped on error; both exited successfully. There were **zero observed training/evaluation process failures, restarts, checkpoint selections, or scientific deviations**. No run was retrained. All completed model artifacts were preserved as written. No additional seeds, tuning, activation transfers, layer/head/neuron searches, Jacobian analyses, or other exploratory rescue runs were performed.

The approved, pretraining-only API repair changed `extract_queries` unpacking from two to three returns and did not change its normalized-query semantics. The separately approved provenance-only update rebound the unchanged scientific preregistration to that repaired implementation before the first D9 model. See the [repair record](d9_pretraining_api_repair.md) and [provenance record](d9_provenance_binding_update.md). Frozen preregistration SHA-256: `837b2bd2e9b9339143572f0d8176d28a28ca34c1307d80dd5b1ffc10d84bf976`. Frozen implementation SHA-256: `f59130607aef97ec2550beaaa2ad0dd88d969036303f2a73b2078fd45800a6c9`. Per-run checkpoint hashes and state hashes are in [run_validity.csv](../../results/d9_comparison_structure/analysis/run_validity.csv); all recomputed checks passed. The raw audit, complete training traces, per-example scores/predictions, geometry rows, controls, analysis scripts, and checksum manifest are included in the reproducibility bundle.

## Bounded conclusion

**No confirmatory causal conclusion about comparison-structure effects is licensed by D9**, because every model violated the frozen object-slot invariance gate. The prospective directional prediction did **not** succeed under the preregistered rules. The transparent but nonconfirmatory outputs show that P1 went opposite the prediction in all eight paired seeds, P2 was mixed, and the P3 paired mean was far below its +0.10 meaningful-effect threshold. The event-control bound passed, and both conditions learned familiar comparisons well; neither changes the validity verdict.

D9 does **not** establish that coherence uniquely causes endpoint formation, a shared latent relational operation, a specific attention/head mechanism, a universal transformer principle, language or LLM generality, or absence of a graph effect under a different valid assay. It does not justify relabeling the failed gate, changing thresholds, selecting seeds, or treating descriptive effects as confirmatory. The fixed comparison graphs remain the only intervention tested; full query–foil exposure was not matchable and is an explicit attribution limit.
