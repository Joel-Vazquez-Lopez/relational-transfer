# Brain-O Experiments 031–034 — A causal source→query→answer relay
**Date:** 2026-10-15 · **Status:** exploratory synthetic trained-Transformer causal evidence; **not** a pretrained LLM mechanism, natural-language generalization, or confirmed scientific novelty.

## What changed after Experiment 030
The previous successful model answered correctly when a queried-verb hidden state was transplanted from a **different queried verb**, but this could be mere query-word transport. E031 holds the **queried verb and requested role IDENTICAL** while swapping which source clause contains that verb. The correct answer changes for disjoint cases. In both successful 3-layer models, replacing only the **query-verb state after layer 1** then **failed** to redirect the answer (0/600 and 14/600), despite the full source edit switching the answer 600/600 in each seed. This invalidates a simplistic *first-layer query token already holds a transferable context-dependent event address* explanation.

## Experiment 032 — Where the causal effect travels
Frozen 3-layer local-aux checkpoints, seeds 1016/1017; 300 fresh disjoint examples each. Source-verb swap preserves both query tokens, but changes the correct answer.

| Matched intervention | 3L seed 1016 | 3L seed 1017 |
|---|---:|---:|
| Swap **source verb states after layer 1** | **300/300** donor-answer match | **289/300** |
| Swap **query verb state after layer 1** | **0/300** | **6/300** |
| Swap **source verb states after layer 2** | **0/300** | **0/300** |
| Swap **query verb state after layer 2** | **300/300** | **299/300** |

Readout-role-only patches do not produce the alternative-verb answer. A 2-layer shortcut model lacks this effect and barely responds to the source reassignment. The effect is a **layerwise causal handoff** compatible with matching lexical query to source verb positions, not proof of semantic role abstraction.

## Experiment 033 — Overlap-stratified replication
Repeat the fixed-query source-verb swap on 200 examples in each of five overlap conditions, per successful seed. Restrict the main denominator to cases where the full donor changes the correct answer: 795 (seed 1016) and 794 (seed 1017).

| Conditional alternate-answer restoration | 1016 | 1017 |
|---|---:|---:|
| Source verb states after layer 1 | 795/795 | 776/794 |
| Query verb state after layer 1 | 0/795 | 20/794 |
| Source verb states after layer 2 | 0/795 | 0/794 |
| Query verb state after layer 2 | 795/795 | 794/794 |

Full donor model accuracy: 100% in this generated panel. No answer changes were observed on cases where full donor answers stayed the same. Samples are independent *generator draws* but not independent linguistic families.

## Experiment 034 — Native attention-edge cuts
Without replacing activations, selectively block edges in frozen trained networks. Disjoint answer accuracy on 300 new cases per seed:

| Attention intervention | 1016 | 1017 |
|---|---:|---:|
| Unmodified | 100% | 100% |
| **Layer 2 query verb cannot attend to source verb positions** | **50.7%** | **48.7%** |
| Layer 2 query verb cannot attend to source name positions | 100% | 100% |
| **Layer 3 answer token cannot attend to query verb** | **48.0%** | **49.0%** |
| Layer 3 answer token cannot attend to source verb positions | 100% | 100% |

The same cuts yield ~75% on shared-agent/patient conditions, matching an event-selection shortcut signature. These cuts change attention normalization and may have indirect effects. Convergent results of state swap and edge masking support **source verbs → query verb at layer 2 → role-answer token at layer 3** as a task-effective route under these conditions.

## Mathematical diagnostic
For target world x and donor world T(x) with **fixed lexical query**, define a conditional donor-answer restoration probability C(i,ℓ) after transplanting donor state at position i after layer ℓ, restricted to pairs where the native target/donor answers differ.

The *relay contrast* R(ℓ)=C(query,ℓ)−C(source-verbs,ℓ) changes from ~−1 after layer 1 to ~+1 after layer 2. This localizes the effect under matched interventions, but is a diagnostic constructed from known factorial-style causal patching, **not a theorem or established novel mathematical structure**.

## Strongest claim and caveats
- Supported: in two independently trained successful **3-layer synthetic** models, source-verb identity and queried-verb matching use a selective cross-token causal route in successive layers; the unsuccessful 2-layer trained control does not.
- NOT supported: general lower bound of 3 layers, unique binding variable, naturally cached pre-question role relation, unseen-vocabulary or constructions, real language, Qwen/Phi/Gemma mechanics, or ACL-level novelty.
- Explicit local event-index auxiliary supervision, fixed positions, six verbs and ten names are substantial limitations.
- Prior art on compositional retrieval, state tracking, and attention routing must be compared before any novelty claim.

## Next decisive gate
New 3+ event and variable-layout training/evaluation with withheld verb families, stronger positional baselines and constrained compute matching. Jointly calibrate with Compass's identifiability controls. Test whether the same native attention route predicts OOD generalization or is a template-specific mechanism.

**Reproducibility:** Chat artifact `BRAIN_O_CAUSAL_RELAY_031_034_2026-10-15.zip`, SHA256 `fe5fd6d7db6e971111b2425219e9b21774913327ea7e4c754f77a20a5ef904d0`. 17 files: generator+model class, 3 frozen checkpoints, 4 executable experiment scripts, unit checks, raw metrics, full scientific note and per-file hash manifest; zip CRC verified. Source and weights are NOT in the repository, but bundled in this conversation artifact.

**Research integrity:** No other branch touched; no main merge; no sealed data; no paid model inference.
