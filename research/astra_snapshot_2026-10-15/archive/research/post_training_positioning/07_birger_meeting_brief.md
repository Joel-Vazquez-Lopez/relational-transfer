# AI Sweden meeting brief — Birger Moëll

*Discussion draft, 8 October 2026. Designed to fit within two printed pages.*

## The decision

The broad idea—post-training changes future learnability even at similar current scores—is **important but already substantially studied**. The closest peer-reviewed work shows exactly this issue for SFT→RL ([Kang et al., ICLR 2026](https://arxiv.org/abs/2510.01624); [Zhang et al., ICML 2026](https://proceedings.mlr.press/v306/zhang26hd.html)). A 2026 [preprint](https://arxiv.org/abs/2606.09932) explicitly calls this SFT-to-RL plasticity. We should offer a sharper test with a credible negative result.

## Three candidate thesis directions

| Direction | Scientific question | Payoff / risk |
|---|---|---|
| **A — Conditional future learning (recommended)** | Do SFT/RLVR histories from one base change later adaptation to independently generated families after controlling starting competence, and does the ranking survive LoRA versus full FT? | Clear, causal and manageable; novelty is a rigorous cross-family/cross-update test, not the concept of plasticity. Risk: effects vanish after known controls. |
| **B — Learning a new composition rule** | After RLVR or SFT, can a model learn an unfamiliar rule faster from the same few examples, beyond its immediate compositional score? | More foundational; requires an executable synthetic environment and careful baseline controls. Immediate composition transfer already has close prior work, so later adaptation is essential. |
| **C — Optimizer-state handoff** | Are some “plasticity” effects caused by optimizer moments/scheduler carried across stages? | Fast, useful control or fallback; narrower scientific significance. |

## Recommended pilot

One 0.5B open-weight base; 3 independent SFT and RLVR history seeds; 3–4 checkpoints per path; two sealed procedural future-task families; matched downstream example ladders (16/64/256/1,024); LoRA on all and full FT on a selected pair. Evaluate absolute held-out learning curves, old-task retention, item-level starting agreement, calibration, high-`k` coverage and training budgets. Compare predictive value of history against [Kang et al.'s](https://arxiv.org/abs/2510.01624) validation-loss/high-`k` proxies. Pre-register one primary effect and stopping criteria. Full protocol: [04_experimental_pilot.md](04_experimental_pilot.md).

**Resources:** planning estimate 80–250 H100-equivalent GPU-hours for Phase 1, subject to a short throughput/memory benchmark on the actual hardware; one 80GB GPU or a smaller-device adapter-only variant; 3–5 weeks for Phase 1. A 5-seed, two-update-rule, optional 1.5B expansion may need 400–1,200 GPU-hours and another 6–10 weeks. Check cluster queue, checkpoint storage and RL rollout software before committing.

**Scientific risks:** prior art already captures much of the question; equal mean scores can mask different errors; RL and SFT cannot share identical on-policy data exposure; opaque base pretraining limits “unseen concept” claims; 3 seeds may only establish feasibility. A strong negative result is useful only if confidence intervals exclude a meaningful effect.

## Questions for Birger

1. What GPU type, memory, queue access and realistic GPU-hour allocation are available to a thesis student? Is multi-seed RLVR feasible?
2. Is the scientific priority **characterization** or a deployable post-training recipe? Would a rigorous negative answer be valued?
3. Can AI Sweden supply a reproducible base/checkpoint family, or should we use a public 0.5B base throughout?
4. Is there an internal task family with exact verification and clear provenance, while allowing publication of generators/results?
5. Are there compute/accounting constraints on full fine-tuning, checkpoint storage or rollout serving?
6. Which baseline would make the result convincing to the team: validation loss/high-`k`, data diversity, entropy, or an existing internal selection rule?
7. Would AI Sweden support a second model/seed expansion only if the Phase 1 effect and power analysis justify it?

## 60-second pitch

“Recent papers show that the best supervised checkpoint can be a poor starting point for later reinforcement learning. So the broad idea that current benchmark score misses future learning is no longer new. We propose a sharper test: train one small base model along controlled SFT and RL trajectories, then expose their checkpoints to two genuinely new, generated task families under LoRA and full fine-tuning. We will measure complete learning curves and compare them against item-level starting skill, validation loss and solution diversity. If training history predicts later learning beyond those simpler explanations, we have evidence for a distinct post-training effect worth scaling. If it does not, we can reject the broader plasticity story and identify which ordinary proxy explains it. The first phase fits a modest compute budget and is useful on its own.”
