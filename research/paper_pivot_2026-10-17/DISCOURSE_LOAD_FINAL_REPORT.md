# Irrelevant participants and status-conditioned retrieval: a falsification and discriminator

17 October 2026. Local Qwen2.5-7B-Instruct MLX 4-bit; greedy answers, no paid API or model download. All prompts and scoring are in the frozen protocols and raw JSONL files. These synthetic passages are **development diagnostics**, not natural-corpus or mechanistic confirmation.

## First matched test: a confounded signal

The [first protocol](DISCOURSE_FRONTIER_PROTOCOL.json) paired 16 base event pairs across four forms: bare, named bystanders, embedded record, and both. The 64 variants each had two single-sentence questions and four joint questions. [Script](discourse_frontier.py), [items](DISCOURSE_FRONTIER_ITEMS.json), [384 raw responses](DISCOURSE_FRONTIER_RAW.jsonl), [full machine report](DISCOURSE_FRONTIER_REPORT.json).

All **128/128** single-sentence answers were exact. Joint answers were **64/64 bare**, **54/64 with bystanders**, **64/64 embedded**, and **62/64 embedded with bystanders**. All 12 joint errors selected the other event's target participant, and all occurred when the requested participant was in sentence two. No bystander was output.

**Critical design fault:** the bystander-only formatter lowercased the target's first letter after the introductory clause, whereas the embedded-bystander formatter retained a capitalized name. Bystander presence and capitalization were not independently manipulated. The frozen prediction required at least a 15-point drop in *embedded plus bystanders* relative to bare and a 15-point increase in target-second disadvantage; the observed joint drop was only **2/64 = 3.1 points**. The prespecified gate failed. We do not attribute the bystander-only 10/64 errors to bystanders from this test.

## Fresh 2 × 2 discriminator

Before any new model output, the [second protocol](CASE_BYSTANDER_PROTOCOL.json) froze **new names** and a complete 2 × 2 manipulation: bystander absent/present × target name proper/lowercase. It held the action, status pattern, question, and target names fixed within each of 16 base items. Each variant had the same two single-sentence questions, both sentence orders and both target questions. [Script](case_bystander_discriminator.py), [items](CASE_BYSTANDER_ITEMS.json), [all 384 raw responses](CASE_BYSTANDER_RAW.jsonl), [machine report](CASE_BYSTANDER_REPORT.json).

| Condition | Single answers | Joint answers | Target first | Target second | Wrong-event participant | Bystander output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare, proper names | 32/32 | 64/64 | 32/32 | 32/32 | 0 | 0 |
| Bare, lowercased names | 32/32 | 64/64 | 32/32 | 32/32 | 0 | 0 |
| Bystanders, proper names | 32/32 | **57/64** | 32/32 | **25/32** | **7** | 0 |
| Bystanders, lowercased names | 32/32 | 54/64 | 31/32 | 23/32 | 9 | 0 |

The proper-name contrast shows that capitalization **does not fully explain** the new bystander effect. The matched target-second deficit is **7/32 = 21.9 percentage points**; a cluster bootstrap over 16 base items gives an exploratory 95% interval of **9.4–34.4 points** (10,000 draws, seed 20261017). Seven of 16 base items contributed one extra error. Five of seven proper-name errors were negated-event queries and two were unverified-report queries. All seven copied the other event's participant; none copied a bystander. The model still answered every individual sentence correctly and every target-first joint query correctly.

**Prespecified decision:** the second protocol required at least **8/32** target-second errors in the proper-name bystander condition before treating this as a strong naturalistic-replication lead. It observed **7/32**. The competence gate passed (64/64 variants), but the practical effect gate missed by one. We do not relax it after seeing output, run a causal patch, or claim a validated mechanism. The lowercase condition is intentionally unnatural and serves only as an artifact diagnostic.

## What remains and what does not

The robust descriptive result is narrower than an event-status circuit: adding irrelevant named participants to short, otherwise easy synthetic passages produced **wrong-event participant copying on some second-target questions**. That pattern appeared on fresh proper-name items and was absent in both bare conditions. This is a promising *failure mode*, but position and syntax-aware answer-source shortcuts remain complete explanations. The embedded first test also showed that one extra framing sentence can mitigate rather than amplify the error, so the effect is not a monotone complexity law.

A paper-level contribution would require a predeclared natural-language replication with independently validated Agents, status scope and matched predicates; multiple models of comparable competence; matched position/lexical/length baselines; and a causal prediction that separates a restricted status-sensitive process from a query-aware compound-key copier. The earlier [NO-GO certificate](../decisive_witness_gate_2026-10-17/REPORT.md) still blocks a unique event-selector inference from a full-residual patch. This investigation supplies a concrete behavioral failure category and a counterexample to the capitalization-only account, **not** a paper-ready mechanism.

Prior work already treats [negated discourse entities](https://aclanthology.org/2022.naacl-main.71/), [cross-event argument confusion](https://aclanthology.org/2024.findings-acl.328/), [position bias in event argument extraction](https://aclanthology.org/2024.findings-acl.487/), and [mixed positional/lexical binding routes](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf). “Bystanders cause errors” or “first mention wins” alone is not a novelty claim. The only plausible increment would be a reproducible **conditional switch to wrong-event copying despite intact single-event retrieval**, with a predictive and causally constrained account that beats those baselines.
