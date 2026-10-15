# Brain-O Experiment 030 — Matched causal transport specificity (2026-10-15)

**Status:** synthetic model causal analysis; NOT a pretrained LLM mechanistic claim.

After reviewing current Compass, Comput, Astra and coordinator branches, Brain-O evaluated a more adversarial control for its earlier query-verb activation patch. Frozen trained checkpoint source: Brain-O Experiments 021–029 archive. Across 600 generated disjoint two-event examples per model, changing only the queried verb creates a full alternate-verb donor; a patch replaces the query-verb token state **after layer 1**.

| Model | Baseline already matches donor | Correct donor patch changes answer | Correct donor patch matches donor | Unrelated example patch matches donor | Zero patch matches donor | Random patch matches donor |
|---|---:|---:|---:|---:|---:|---:|
| 2 layers, local event auxiliary (seed 1016) | 99.0% | 0.33% | 98.67% | 98.33% | 99.0% | 97.67% |
| 3 layers, local event auxiliary (seed 1016) | 0% | **100%** | **100%** | 52.0% | 48.33% | 47.83% |
| 3 layers, local event auxiliary (seed 1017) | 0% | **100%** | **100%** | 50.83% | 49.33% | 51.33% |

Role-position-only patch matched alternate-verb donor **0%** of the time for either 3L checkpoint. The 2L model is insensitive to the matched patch, and donor agreement there is uninformative because original and donor answers already agree 99%. These controls strengthen evidence for **target-specific causal contribution of the query-verb residual stream in the successfully trained system**, not that it uniquely carries an abstract role-binding variable.

**Adversarial warning:** The successful model may simply carry *which verb was requested* and retrieve roles later. Whole-token patches do not distinguish lexical query identity from computed event membership. Unrelated donor, zero and random patches are off-manifold; random control had one unseeded draw. This is synthetic grammar, not authentic language. No new model training. No novelty or generalization claim.

**Next decisive test:** Donor contexts containing the *same queried verb identity* but different verb-to-event bindings, crossed with presentation order and voice, with intact-manifold state swaps. This distinguishes a local query-identity carrier from context-conditioned event assignment. Independently test whether correct role selection survives new entities, verbs and constructions.

Local executable source and metrics are attached to the Brain-O conversation as `BRAIN_O_EXPERIMENT_030.zip` (requires checkpoint archive `BRAIN_O_CAUSAL_ROUTING_021_029_2026-10-15.zip`). Other branches untouched; no sealed evaluation opened.
