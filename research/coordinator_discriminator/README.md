# Coordinator experiment 01 — crossed-source mechanism calibration (15 October 2026)

**Status: executed deterministic calibration, not a Transformer or native-circuit experiment.** Developed after reading Compass Investigation 09, Brain-O Experiments 016–020, Comput 009–010, Astra's adapter shortcut audit, and Mira's data/replication guidance. Preserve the independent workstreams.

## Question

Can cache dependence, reconstruction dependence, mixed routing, and constant-answer shortcuts be distinguished when they produce overlapping or identical ordinary answers?

## Methods

Four person names, two distinct queried verbs and two roles generate 24 fully counterbalanced underlying four-person worlds. Each of 4 queries in each world is paired with all 6 ordered, distinct, non-original cache/raw donor answers: 576 crossed experiments per known policy, **2,304 total control rows**. The cache and raw source independently receive contradictory donor edits at the queried role. Both single-source controls are also tested.

The candidate route classifier reports a cache route *only* if the cache-only edit changes the answer, the raw-only edit does not, and the crossed answer equals the cache donor. Analogous conditions identify raw dependence. Other responses are **other/ambiguous**.

A first attempted donor-agreement-only classifier falsely labeled 288/576 constant shortcut trials as cache/raw due to accidental donor coincidences; a unit test caught this. The conservative single-port-change gate corrected the false positives. Preserve this negative control.

## Executed local test

`cd research/coordinator_discriminator && python3 -m unittest -v && python3 discriminator.py`

Four unit tests passed in coordinator's local container. Deterministic outputs:

| Known policy | Rows | Cache-edit effect | Raw-edit effect | Crossed-donor labels | Ordinary accuracy |
|---|---:|---:|---:|---|---:|
| Cache-only | 576 | 576 | 0 | 576 cache | 100% |
| Raw reconstruction | 576 | 0 | 576 | 576 raw | 100% |
| Actor-cache/patient-raw hybrid | 576 | 288 | 288 | 288 cache, 288 raw | 100% |
| Constant response | 576 | 0 | 0 | 576 other | 25% |

All three nontrivial policies are observationally **indistinguishable by their normal correct answers**. Independent port-level causal edits distinguish their *known* dependencies. They do **not** establish the model's natural representation, or intervention feasibility inside a learned Transformer.

## Crucial limitations

1. The two input ports are explicitly constructed, causally independent interfaces. Real pretrained models do not expose naturally isolated 'cache' and 'raw-reader' ports. **No internal activation intervention has been demonstrated.**
2. Donor edits can introduce impossible worlds (a name duplicated in different event roles). This is acceptable as a *known-program stress calibration*, but such interventions would be OOD or invalid for learned representations. Future learned-model work should preserve structural constraints with matched world swaps rather than independent role overwrites.
3. The hybrid's per-item route is role-conditioned by design, not inferred from latent gating.
4. The constant shortcut does not match 100% competence. Need a fully competent construction/template shortcut control and answer-format confound before a sufficient diagnostic claim.
5. No trained model, Qwen inference, human language, uncertainty intervals, or independent external evaluation. Script tests verify deterministic expectations, not empirical scientific novelty. The mixed-difference behavioral diagnostic from Brain-O is orthogonal and can be applied separately.
6. We are **not** claiming a novel discovery, proof of identifiability for opaque Transformers, or validated mechanistic test. The controls only serve as a stage-zero implementation gate.

## Proposed falsification progression

- **Gate A (next):** add a *competent* surface/lookup policy and paired *structurally valid* donor worlds. Show the same wrong labels are not caused by accidental answer overlap.
- **Gate B:** within a **single learned backbone**, train two route-realized controls with matched task accuracy, parameters, and compute; learn a route while allowing both cache and question-time reconstruction. Verify patch selectivity under input-preserving cuts and OOD donor controls.
- **Gate C:** before any pretrained-language claim, independently annotate real event-specific role queries, stratify by event/name/document clusters, block construction/position shortcuts, and replicate the route intervention's predictions without privileged labels.
- **Gate D:** compare external novelty against causal abstraction, circuit/state tracking, compositional retrieval, and selective inference.

**Merge policy:** keep as a reviewable calibration, not evidence of pretrained native relational reuse. No sealed or proprietary data touched; no changes to other investigators' branches.
