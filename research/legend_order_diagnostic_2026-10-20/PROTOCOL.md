# Frozen legend-order diagnostic —20 October2026

## Question and scope

The completed factorial panel emitted the code for not-entailed on727/768 trials, but not-entailed was always the last legend entry. Category preference and last-option following were confounded. This one local Qwen7B pilot separates their operational predictions using32 fresh cases. It is not a binding mechanism experiment or population replication.

## Fixed design

Two AI-authored sentence families: Clara/Mateo with arrested/arrest; Hector/Nadia with thanked/thank. Each has2 role orientations ×2 question directions ×2 A/B mappings ×2 legend orders=16 cases;32 total. Active sentences and questions, same system instruction/background, no prefix, one unchanged128-token canvas. Only the order of the two complete legend clauses is reversed within a matched pair. Assert identical source/question token prefixes, token count, complete token multiset and measured end positions. Both mappings cross both orders. Unsupported converse means not entailed, not known false. No human annotation. Names and predicate vary together at family level.

Execution order is shuffled once with fixed seed20261020 and frozen in INPUTS.json. No model outputs influence cases or order. Full native vocabulary argmax is the first-token outcome; A/B and leading-space aliases map to classes. Off-format output matches no prediction. No fitted threshold, calibration or answer exclusion.

## Frozen rival predictions and decision

1. negative_category: emit the symbol assigned to not-entailed regardless of truth or legend order.
2. last_option: emit the last legend symbol, regardless of truth or which class it means.
3. relational_answer: emit the symbol matching the clause/question entailment, regardless of legend order or mapping.

Qualification requires≥28/32 matching decisions,≥14/16 in each family, and≥14/16 in each legend order. Predictions agree pairwise on exactly16/32 trials, making these qualification regions disjoint. CPU fixtures must recover each and reject constant-A and constant-B nulls. None qualifying means mixed/unclassified; no closest-pattern winner. No p values.

Independently, causal-task readiness requires32/32 native-correct decisions and8/8 complete four-answer units. A qualifying relational pattern at28–31 correct does not pass this competence gate. No causal intervention is run or automatically authorized here.

## Instrumentation, resources and technical gates

Existing local Qwen2.5-7B-Instruct4bit checkpoint and installed MLX stack; no model download, training, cloud or paid API. Native unwrapped forward, batch1, fixed128 tokens with x-padding after the original assistant end; read logits at that original end. Save full152064-dimensional float32 vectors losslessly and every first-token decision/top12 tokens.

**33 forwards maximum:1 exposed equivalence control before fresh cases, then32 diagnostic calls.** The control reproduces the previously saved unwrapped128 output from the completed study; full-logit error≤.05. Any technical/resource gate failure stops collection and preserves all partial data, with no automatic retry or tolerance change. Count attempts before dispatch. Maintain disk>4GiB, global swap<2GiB; use the prior6GiB MLX memory guideline (not a hard OS guarantee),256MiB cache, explicit post-call GC/cache release, and one short-lived model process. Do not close unrelated apps or alter system settings. The previous OS-panic cause remains unknown.

Freeze source, inputs, predictions and model hashes on the existing research branch before new inference. Independently replay hashes/scores/argmax/journals after collection. Preserve every unexpected answer and all raw vectors, technical control, source/annotation uncertainty and resource records. Save report and append to Issue35, then stop.

## Interpretation limits

Last-option following is evidence for an order-dependent output policy, not unique proof of literal copying; recency-weighted class interpretation or other instruction behavior can reproduce it. Negative preference does not prove absence of relational information. Correct behavior does not identify a neural algorithm. Two authored families and one checkpoint cannot establish generality or novelty. No failed gate from previous investigations is repaired retroactively.
