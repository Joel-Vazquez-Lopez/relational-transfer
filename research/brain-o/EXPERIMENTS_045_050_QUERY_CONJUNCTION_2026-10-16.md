# Brain-O Day 2, Experiments 045–050 — Learned compound event selection without answer use
**16 October 2026 | Status: executed, frozen-checkpoint replay verified; synthetic development, not real-language or pretrained-model novelty.**

## Main result
Three-layer width-64 ordinary causal Transformers with 106,907 trainable parameters, generator-derived event-index auxiliary supervision, and a three-event verb×context collision task can learn an accurate conjunction-sensitive event selector **without learning to use it in the final actor/patient answer**. All evaluated models had **0/200 complete six-question worlds**.

### Fixed 200-world × 6-query evaluations (1,200 dependent QAs per seed)

| Architecture / objective | Steps | Seed | Correct event index /1,200 | Correct answer /1,200 |
|---|---:|---:|---:|---:|
| E043 original, tag-position event head | 500 | 1040 | 802 | 376 |
| E043 original | 500 | 1041 | 804 | 389 |
| E045 event head moved to final role token | 500 | 1040 / 1041 | 400 / 400 | 390 / 385 |
| E045 parameter-free pooled-query readouts | 500 | 1040 / 1041 | 394 / 406 | 380 / 366 |
| E046 paired-world standard CE | 500 | 1040 / 1041 | 796 / 804 | 388 / 385 |
| E046 paired-world + margin | 500 | 1040 / 1041 | **1190 / 822** | 393 / 386 |
| E049 paired-world standard CE | 750 | 1040 / 1041 | 810 / **1200** | 393 / 397 |
| E048 + E050 paired-world + margin | 750 | 1040 / 1041 | **1196 / 1200** | 387 / 395 |
| E048 margin sensitivity | 1000 | 1041 | **1200** | 395 |

**Interpretive correction:** The impressive seed-1040 margin result at 500 steps did **not** replicate at the same step count in seed 1041. At 750 steps seed 1041 learns the conjunction under both margin AND ordinary paired CE. Therefore **the margin is neither proven necessary nor generally reliable**; longer ordinary training can also succeed. Seed 1040 standard CE remained at 67.5% event accuracy at 750, so trajectory/optimization variation remains. E048/E049/E050 reproduced the original 500-step state dictionaries exactly (max tensor difference 0) before continuing the same optimizer trajectory.

At 750 steps both margin models have ≥99.67% compound event-index accuracy, but final answers remain 32–33%; no complete worlds. E046 training minibatches are 64 **paired** worlds × 2 query variants versus E043's independent examples, so do not treat E043 vs E046 as an exposure-matched one-factor comparison. Paired-CE vs paired-margin uses the same questions, batches, seeds, steps and architecture.

## E047 — Native causal query-token state transplant
For fixed source worlds and requested role, change only the queried verb or context tag to select a different event and different gold participant; transplant one whole intermediate query-token state from counterfactual donor into target after a native Transformer layer. 200 generated worlds × two roles × two changed-key conditions = 800 correlated intervention pairs/model. Reassembled layer forward was checked numerically against the native unpatched computation.

**For verb-query changes at a matched 750 steps:**

| Margin model | Event-index prediction changes after L2 *verb token* donor patch | After L2 *tag token* donor patch | Answer changes under effective patch |
|---|---:|---:|---:|
| Seed 1040 | **99.5%** | 0% | ≤0.25% |
| Seed 1041 | 0% | **100%** | 0% |

An independently trained **standard paired-CE seed 1041** model with 100% event accuracy likewise routes the verb-key effect through the L2 *tag* state (100% event switches, only 1.5% answer switches).

This documents **task-effective intermediate causal dependence**, and a difference in effective token-state routes at matched architecture/objective/step count. It does **not** isolate an abstract binding variable: the transported whole state may encode query-lexical identity; patches can be off-manifold; multiple circuits can implement the same function. A learned answer head can ignore an accurate nearby event state. Differences between two seeds do not establish a universal computation schedule.

## Main falsifiers and alternatives
- E039 explicit learned lexical lookup/copying copier already reproduces simpler two-factor causal signatures (not architecture-matched).
- Simple readout relocation/pooling failed in E045, but more expressive learned readouts are not excluded.
- Original query-key layouts/finite grammar could induce residual and position shortcuts.
- Event labels are generator privileged *during training*, not naturally discovered semantic roles.
- Held-out worlds are generated from the same restricted grammar; not authentic language or new structural families; 1,200 questions per seed are not independent worlds.
- No claim about reuse vs question-time reconstruction, semantic binding, ACL novelty, Qwen/Phi/Gemma, or universal minimum depth.

## Next decisive intervention
Build an **ability- and compute-matched answer-routing rescue**: a trainable, inference-time label-free route allowing the final role answer position to use a *verified* conjunctive event-selection representation, compared with an equal-parameter sham/copying route; evaluate whole-world accuracy and targeted necessity cuts on 3–5-event / repeated-predicate / heldout structural families. In parallel, follow issue #23 authentic human-adjudicated roles + two untouched pretrained *model families* (field-order study belongs to Astra; avoid duplication).

## Artifact and provenance
Full self-contained conversation archive: `BRAIN_O_DAY2_E045_E050_QUERY_CONJUNCTION_2026-10-16.zip`  
**SHA-256:** `d4e7c50db3cf9a9fb666f1a64d6b4ca9d4e75b2e24a0dfe8cd151e7e8cf00cb8`  
**78 archive members, CRC pass, 77/77 internal member SHA matches**, complete code, logs, per-question counterfactual CSVs, 15 frozen checkpoint state dictionaries, scientific report, and `verify_saved.py`. **All 15 checkpoint evaluations replayed exactly**, stdout bundled in `REPLAY_OUTPUT.txt`. These binaries are conversation-attached and NOT uploaded to GitHub or automatically shared across researcher sessions. Use `torch.load(..., weights_only=True)` and review trusted source.

**Signal: PROGRESS, including negative controls and a causal dissociation in synthetic trained models; no EUREKA validated. No paid inference, sealed data, main merge or modification of another research branch.**
