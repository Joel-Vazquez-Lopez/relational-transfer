# Vision: how linguistic relationships become computation

## The intuition we are protecting

Look at a controlled sentence transformation across ALL layers and meaningful token positions. Qwen and Llama do not need to follow identical lines: one may respond earlier, another later; one may be stronger, another weaker; a high-dimensional response may be rotated or inverted. Yet their *pattern of rising, falling, stabilizing and interacting* may be related.

The specific motivating observation was a role swap. Subject and object separations showed related decreasing trends with different late effects; an unchanged verb responded as context was updated; the final question-token separation grew later. This is **visual and descriptive**, not proof that the same linguistic operation or causal chain occurred.

We chose Qwen 7B and Llama 3B because they are **distinct but not wildly unrelated**: both are decoder-only Transformer-family systems. That makes them a tractable first comparison while leaving architecture/training confounds unresolved.

## Primary research questions

- When a sentence changes in a precisely labeled linguistic way, what happens to the *entire path of processing* from input through successive Transformer blocks?
- Which factors (name identity, semantic role, grammatical voice, word order, verb identity, factuality, question demand, distractor) influence which sites, layers and internal sub-operations?
- Do those factors produce recurring temporal/depth patterns across instances within a model, and do any corresponding patterns appear across models despite differing magnitude/timing/coordinates?
- Which aspects of the response belong to a shared architecture/training/readout background, and which track **linguistic distinctions**, independently of the final answer?
- Does studying attention/MLP contributions, token interactions and labeled perturbations reveal relationships that final residual-path summaries cannot?
- Can an observation-based representation ever predict a *new* linguistic or causal effect beyond depth, lexical order, answer identity and patch strength baselines?

## Linguistic contrasts: small but rich

Use examples with explicit labels, including:
- “Zorva rescued Mikel.” ↔ “Mikel rescued Zorva.” — meaning-changing Agent/Patient reversal, but confounded with name order unless controlled.
- “Zorva rescued Mikel.” ↔ “Mikel was rescued by Zorva.” — semantic roles preserved, surface order and syntax changed.
- “Zorva rescued Mikel.” ↔ “Zorva did not rescue Mikel.” — factuality/scope change.
- Same sentence, different **Agent**, **Patient**, **first-mentioned** or **event-identity** questions — can expose answer preparation without changing source content.

The made-up names are examples only, not unverified tokenization-clean stimuli; validate their subtokens, positional alignment and the model's task competence. Crossed designs must permit semantic roles and surface positions to disagree.

## What “development” means (keep distinct)

1. **Within an inference pass:** sequence of transformations across blocks and sublayers.
2. **Across token positions / autoregressive decoding steps:** interaction history and generation unfolding.
3. **Across training checkpoints:** parameters and representations evolve with learning.

These are different axes. We begin with (1) and token-site relations, later considering (2); (3) is a separate long-term project. Transformer blocks are different maps, not automatically iterations of a stationary dynamical system.

## Multiple forms of proximity

Absolute hidden-state proximity, internal step-direction proximity, multiscale curve shape, labeled perturbation interactions, component-level information flow and *causal functional consequences* can disagree. We want to understand what each reveals, not collapse them into one obligatory score.

Compare curves of separation and step norms while retaining true high-dimensional states: norms may look close even when directions are different; generic high similarity can be driven by depth. A model's internal coordinate direction cannot directly be compared with another's without valid assumptions.

## Scientific success can take many forms

A principled measurement of linguistic processing development; a robust empirical regularity; a falsified intuitive measure with a useful new control; a predictive compositional technique that makes old methods work together; or a new causal account. “Absolute breakthrough” is **not** the only worthwhile outcome.

The research objective is **to explain how linguistic relationships develop inside the black box**, not to get a flattering similarity coefficient. Failure of one metric is not failure of the question, and a beautiful shared curve is not yet evidence of a shared algorithm.
