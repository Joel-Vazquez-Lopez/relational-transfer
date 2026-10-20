# Stage-gated experimental recommendation — 12 October 2026

Status: **plan only**. Stage A is complete when the audit and literature review are accepted; Stage B implements a new source-only benchmark; no new training is authorized by this document. The independent human review of the reserved active/passive challenge remains a separate track.

## Core decision

The cheapest experiment that actually tests “artificial sleep” is **S2 offline versus S2 interleaved on the same L0–L2 examples**, with matched ordinary augmentation and extra-update controls. S1 narrow replay was already tried in Qwen with an unhelpful result. S3 depends on a proof-safe generator and adds obvious data-coverage confounds. Begin with the existing lightweight controlled Transformer, then only move to Qwen2.5-0.5B if the small-model benchmark distinguishes methods. Do not begin with Qwen1.5B.

The first question is **whether the new benchmark itself is valid**. If shortcuts solve it, a sleep comparison cannot be interpreted. The independent natural-language challenge can later test external validity after blind human review, never as a development set.

## Stage B: benchmark implementation gate

Implement `BENCHMARK_SPEC.md` L0–L2 only, in a new directory, without opening any reserved natural challenge item. Freeze the grammar, PRNG seeds, graph manifest, split hashes, S2 pair maps, and output parser. Use existing controlled-task code where possible but retain new versioned files. Unit tests should show: no unordered-pair leakage; every meaning-preserving transformation keeps role IDs; every true reversal swaps them; no generated training item shares held-out `(unordered_pair, predicate)` key; graph matching is within frozen tolerances. Human-check a blinded source-only sample of generated L2 text. Benchmark card lists exact counts and all heuristic baselines. If a first/last mention, `by`-phrase, grammatical-subject or lexical memorization rule exceeds the predeclared validity threshold on the intended hard stratum, repair the benchmark **before** model inference.

## Stage C: minimal pilot, then a decision

Use one development graph family and **two paired initialization seeds** for engineering only. All conditions start from the identical wake checkpoint per seed. Training data and wake history are identical. Define a continuation budget of `N` ordinary batches and `M` transformed batches. Every arm except A uses `N+M` continuation updates; D and E use identical batch IDs and losses, differing only in whether the `M` transformed batches form a final block or are interleaved among the `N` ordinary batches. The diagnostic arms are:

| Arm | Post-wake information / objective | What it isolates |
| --- | --- | --- |
| A stop | No extra updates | Size of any continuation effect |
| B extra ordinary | `N+M` batches from original training examples | Extra optimization alone |
| C augmentation | `N` ordinary plus `M` transformed batches in the same block order as D; ordinary answer loss on transformed batches | Extra examples/targets without the consistency loss |
| D S2 offline | `N` ordinary batches followed by `M` transformed batches with output-consistency loss | Paired consistency plus a final offline block |
| E S2 interleaved | **Exactly the same** `N` and `M` batches and losses as D, interleaved during continuation | Ordering of identical information |

Use a constant learning rate for this diagnostic so batch type is not confounded with a changing schedule; batch order remains the intended manipulation. A reverse-block order control can later test recency if D beats E. If other strict matching is impossible, report the imbalance and limit the claim. Log per-arm tokens, batches, active parameters, optimizer steps, loss scale, and wall time. A teacher/oracle is absent except the declared symbolic role map shared by C/D/E. Select `λ_inv` and `λ_eq` on development only from a **small frozen grid**; the same grid applies to D/E. Do not choose a winner from the confirmation graphs.

Primary development endpoint: **event-level joint actor-and-patient correctness across canonical, passive, patient-fronted cleft, and genuine role reversal** on withheld pairs. Secondary: per-form accuracy; meaning-preserving invariance and meaning-changing equivariance separately; familiar-event retention; new pair versus new verb; candidate likelihood margin; generated exact answer; all shortcut baselines. An arm may increase consistency by repeating the same wrong answer, so consistency without correctness never passes.

**Pilot go/no-go:** proceed only if the benchmark has nontrivial headroom, D and E are technically matched, and D shows a repeatable event-level improvement over **both C and E** in the same direction on both development seeds without a material familiar drop. This is a screening criterion, **not** a publishable significance test. If D only beats A/B, it has shown data or objective value, not an offline schedule advantage. If D and E tie, drop the sleep-specific claim and study S2 as ordinary consistency training. If C matches D/E, drop the special-objective claim too. Negative outcomes go in the report.

## Stage D: controlled confirmation, only if Stage C passes

Freeze architecture, weights, all schedules, λ values, parser, exclusions and main comparison before running **at least 6–8 new graph families × 3 paired starts**. Families, not individual prompts, are the independent topology units. Each family gets all arms A–E, plus S1 replay and S3 only if their distinct hypotheses survive earlier gates. Prespecify a minimum meaningful effect, for example **+10 percentage points in joint-event accuracy** over both C and E with a family-level paired interval excluding zero, while familiar-event accuracy loses at most **2 points**. These are provisional design thresholds to be fixed after source-only benchmark feasibility, never from confirmation results. Report raw family effects, means, uncertainty, and all failed guards. Do not call 24 starts 24 independent graphs.

For Qwen2.5-0.5B, reuse the same causal contrasts after the controlled test, with explicit before/after forgetting curves. Compare untouched base, wake checkpoint, and each continuation arm. Evaluate direct answers and candidate scores on the **same** generated events; hold out question wordings. If human-reviewed natural material becomes available, keep it untouched until model/selection rules freeze. The 1.5B model is a later scaling check, not initial method development.

## Stage E: mechanism follow-up, only for a robust behavioral gap

Train frozen linear probes on disjoint training graphs; test on unseen graph, lexical and construction strata. Regress out participant position/identity; compare candidate likelihood to generated answers. Intervene on candidate role states using matched examples, plus random-subspace and position-matched controls. Require selective behavioral change beyond probe decodability. A null intervention limits the proposed mechanism, though it does not show no role information exists elsewhere.

## Budget and reproducibility

Use a tiny engineering calibration (one arm, one seed, e.g. 100 updates) to measure **actual local seconds/update and peak memory** before committing to Stage D. Planned small-model confirmation at 8 families × 3 starts × 5 arms is **120 training runs** plus shared wake runs; this is comparable in count to the saved 96-model RQ2 study but cost per run must be measured. Qwen0.5B is an additional gate with fewer arms after screening; do not spend the available external-model budget on this pilot. No reliable wall-time or euro estimate is claimed without calibration.

For each run save immutable config, code revision, base/checkpoint hash, split/grammar hashes, family ID, PRNG seed, examples/tokens/updates, train trace, evaluation JSON per item, and failure log. Save a frozen analysis file before confirmation. Maintain a separate `RESULTS.md` **only after** real experiments; do not fill it with forecast numbers.

## What would change direction

- **Positive:** offline S2 exceeds identical-data interleaved S2 and supervised augmentation across fresh graph families on joint correct role interpretation, with retention intact. This supports an offline scheduling effect in a controlled model; it still needs natural-language replication and prior-art review for a strong publication claim.
- **Negative 1:** S2 helps, but interleaved and offline match. Keep the useful transformation-consistency method; drop the artificial-sleep claim.
- **Negative 2:** augmentation matches S2. Attribute benefit to transformed supervision/coverage and focus on data design.
- **Negative 3:** heuristics or label ambiguity solve/confound the benchmark. Stop and repair measurement before any mechanistic story.
- **Negative 4:** controlled effect exists but Qwen or verified natural transfer does not. Report a scope boundary and investigate forgetting/format/shortcut explanations, not a general LLM repair.
