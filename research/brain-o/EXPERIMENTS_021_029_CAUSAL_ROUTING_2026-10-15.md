# Brain-O Experiments 021–029 — Learned address routing across Transformer layers
**Date:** 2026-10-15
**Status:** exploratory but replicated **synthetic trained-model causal evidence**, NOT pretrained-LLM mechanistic evidence or established ACL-level novelty.

## Central result
Under the original two-event structured-token grammar (10 names, 6 verbs; active/passive; disjoint/shared/cross-role entities), we trained standard causal TransformerEncoder stacks with a query-verb, query-role, and final answer. A binary auxiliary head attached to the **query-verb position** learns whether the queried verb belongs to the first or second surface clause. No dedicated hard-coded event pointer, role gate or retrieval step is inserted into these models.

**Event identification can be learned perfectly at the queried-verb token while the answer head still ignores that information.** A three-layer network with the same local auxiliary objective, unlike two-layer systems, learned to combine the queried verb and role and solved all 2,000 generated evaluation questions in multiple training runs.

### Held-out generated results (400 per condition, same grammar)
| System | Params | Updates | Seed | Event-index head | Disjoint answer | Full panel |
|---|---:|---:|---:|---:|---:|---:|
| 2-layer, local event auxiliary | 40,572 | 900 | 1016 | 100% | 53.25% | shortcut-like |
| 2-layer, **wider**, local auxiliary | **62,232** | 700 | 1016 | 100% | **49.50%** | shortcut-like |
| 3-layer, **no auxiliary loss** | 59,532 | 700 | 1016 | untrained head, do not interpret | **51.75%** | shortcut-like |
| 3-layer, local event auxiliary | 59,532 | **700** | **1016** | **100%** | **100%** | **2,000/2,000** |
| 3-layer, local event auxiliary | 59,532 | 900 | 1016 | 100% | 100% | 2,000/2,000 |
| 3-layer, local event auxiliary | 59,532 | 700 | 1017 | 100% | 100% | 2,000/2,000 |

The seed-1016 three-layer **700-step auxiliary/no-aux comparison is matched for architecture, initialization, update count, batch generator and parameters**, and differs in supervised objective. The wider two-layer auxiliary control has **more parameters than the successful three-layer model**, yet does not learn joint answering at 700 steps. This argues against a parameter-count-only explanation under these conditions, but does not identify depth as uniquely causal across seeds and distributions.

### Causal activation transplant
For a fixed pair of source events, change only the **queried verb** to construct a donor forward pass. After layer 1, transplant only the donor's contextualized query-verb token state into the original pass; hold other token states and source events fixed, then continue the model normally.

On 500 independently generated disjoint cases:
- 2-layer local-aux model: event-index classifier flips **500/500**, answer flips **1/500**.
- 3-layer local-aux model seed 1016: event-index classifier **500/500**, answer **500/500**.
- 3-layer local-aux model seed 1017: event-index classifier **500/500**, answer **500/500**.
- Replacing the *role-position state* at the same cut does not produce this verb-switch answer effect.
- Independent packaged replay on 400 different disjoint cases: answer flips **0.25%** (2-layer) versus **100%** (both 3-layer models), with event head flipping 100% in all.

After **either layer 1 or layer 2** of the trained three-layer networks, query-verb-state transplantation reproduces essentially the full donor answer distribution. Intervening only after the final layer does not change the answer (read from a separate final-role position). This is a **causal route** for this trained system, but swapping a whole hidden state is not isolation of a unique semantic variable.

### Crucial negative and input-gradient controls
- A frozen 2-layer model's first-layer query-verb representation already supports **100% held-out event-index prediction** by a linear probe; the uncontextualized embeddings yield 48.3%. Therefore the strict story that event information is only first computed in layer 2 and therefore physically cannot reach the answer is **falsified**.
- In scratch trained role-last models, earlier query-token input gradients are only ~3.7–4.5% of final-token gradients; in verb-last models ~5.1%. At random initialization, this earlier/final ratio is already ~4.8–6.5% across six random seeds × two orders. Thus final-token sensitivity has a major architectural component, not a newly discovered semantic preference.
- A neutral final readout after both factors (2 layers, 900 steps) scores **25.3%** on disjoint cases and barely reacts to either query factor; a final-state event auxiliary also fails to learn event identification in that setup. Simply moving the readout or adding final auxiliary supervision does not solve the task.

## Interpretation and novelty limits
**Functional distinction established on this synthetic task:** the local event-index computation exists and is causally manipulable, yet can remain disconnected from downstream joint answering. In successfully trained three-layer controls, the same kind of intervention causally changes the final answer in the predicted direction. This is **not** a uniquely identified natural pre-question reuse mechanism.

**Not established:** semantic-role generalization on authentic language, a pretrained-LLM binding circuit, that exactly three layers are required, a unique mathematical object, or a new architecture principle. The auxiliary event label is supplied by the generator. The task uses a finite, repeatedly sampled structured grammar with positional shortcuts; natural-language label review and OOD confirmation are essential.

Nearby prior research on induction heads, causal tracking, entity binding, query/key similarity, and compositional retrieval prevents claiming the general idea of retrieval routing as new. Scientific value here is the explicit learned **information-present / causally-unutilized** contrast with strong controls, which merits harder replication.

## Required next checks
1. 3–5 seed depth × width × local-supervision factorial with exact update and budget matching, confidence intervals by independent event family.
2. Variable-length 3+ event contexts and **unseen** mappings, entities, verbs, passive/cleft paraphrases, randomized positions that defeat positional rules.
3. A matched successful natural-language pretrained system and prompt-format controls, human-verified semantic labels, prior-art comparison, plus intervention specificity checks.
4. Test whether the causal routing path changes under task adaptations without importing the generator-derived event-index label.

## Reproducibility
**Complete Brain-O conversation artifact:** `BRAIN_O_CAUSAL_ROUTING_021_029_2026-10-15.zip` (SHA-256 `1d41033c0c8d98dba851b41a36a68761cf22b97b06ed9f8986abba628c909105`, 50 files, ZIP CRC verified). Includes `portable_replay.py`, scripts, trained checkpoints, numeric JSON, logs, analysis and per-file hash manifest. **Archive is attached in the Brain-O conversation, not stored on GitHub**. Portable replay confirmed the published accuracy and activation-patch results without new training.

**No sealed confirmation data opened. No other investigator branch changed. Do not merge to main without coordinator review.**
