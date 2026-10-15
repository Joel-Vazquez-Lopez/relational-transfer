# Scientific positioning: proposed thesis

## Strongest question

**After controlled SFT or RLVR from one base model, does training history contain reproducible information about later learning on independently generated task families, beyond starting item-level competence and known exploration/validation-loss proxies—and does that information survive a change in adaptation rule?**

This question permits a negative answer and does not presume a universal plasticity scalar.

## Closest prior work and precise gap

Kang et al. show that pre-RL SFT scores can mislead and that validation loss and high-`k` pass rate are better predictors ([L1](01_literature_map.md)). Zhang et al. show that SFT can be optimized for later RL ([L2](01_literature_map.md)). Liu et al. explicitly study and attempt to restore SFT-to-RL plasticity ([L4](01_literature_map.md)). Immediate cross-domain and compositional transfer is also studied ([L33–L36](01_literature_map.md)). Sequential representation collapse and LLM plasticity are under active preprint study ([L5–L6](01_literature_map.md)). MAML establishes optimization for rapid future adaptation ([L9](01_literature_map.md)).

The **remaining gap**, if confirmed after full-text comparison, is a modest one: randomized *post-training histories* evaluated on fresh, independent task generators under more than one subsequent weight-update procedure, with item-level starting competence and simple L1 proxies accounted for. Existing evidence mostly asks whether SFT prepares for a following RL stage or whether prior competence is retained. We cannot claim no paper has ever examined this exact configuration; this audit only establishes that it is not resolved by the sources verified here.

## Contribution and importance

The contribution would be a **controlled causal benchmark and an out-of-sample prediction test**, not the training-response notation. It would quantify how much later adaptation depends on history, task family and update procedure, and whether baseline competence, high-`k` coverage and validation loss suffice to predict it. This matters because selecting a checkpoint by current benchmark score may misallocate expensive later training, a failure already demonstrated for SFT→RL (L1–L2). The thesis asks whether that failure generalizes or is specific to the handoff and task distribution.

## What would falsify it

- Across preregistered future families, the history effect is inside practical equivalence bounds after adequate replication; or
- effects exist only for one chosen optimizer/family and show unstable rank reversals, defeating a general history-level conclusion; or
- starting item profile, validation loss and high-`k` coverage predict adaptation as well as history labels on held-out seeds/families.

These outcomes are informative if the design has enough precision. They would favor distribution-specific transfer or established exploration explanations over a new general principle.

## MSc scope versus longer program

An MSc can run a 0.5B model, short SFT/RLVR trajectories, two procedural future families, two adaptation configurations, multiple seeds and paired learning curves. It can establish or bound *one* conditional effect. It cannot prove acquisition of a novel internal algorithm, measure universal model plasticity, or establish a causal neural mechanism from correlations.

A longer program could replicate across architectures and scales, add DPO/RLHF and real-language domains, derive held-out-task predictors, then intervene on mechanisms (exploration support, representation reuse, gradient interference). Any new objective to optimize future learnability must be compared against MAML, replay, early stopping and the SFT→RL methods L2–L4 at equal lifetime compute.
