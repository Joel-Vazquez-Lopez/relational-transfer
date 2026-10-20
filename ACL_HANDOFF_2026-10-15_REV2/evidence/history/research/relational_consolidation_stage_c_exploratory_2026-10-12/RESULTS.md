# Exploratory development-only synthetic pilot — 12 October 2026

**Status:** one seed, one development graph, human grammar review pending, no confirmation items or natural-language challenge accessed. This is an engineering screen, not a publishable test of artificial sleep or LLM relational abstraction.

## Why it ran

The user did not have time for the Stage B blind grammar review and asked the research to continue. The Stage B generator has formal role-map and leakage checks but only AI inspection of grammatical naturalness. This run therefore tests whether the synthetic benchmark has headroom and whether a consolidation contrast is even distinguishable. It cannot validate authentic language claims. The [protocol](PROTOCOL.json) fixed architecture, seed list, steps, arms and screen before training.

## Set-up

Family 0 only. The model is a two-layer, width-64 Transformer encoder with a name-classification output. It saw 324 observed training events in both symbolic and canonical form, with actor and patient questions. Its development set contains 90 events over 15 withheld unordered participant pairs, three verbs and both directions. Each event is evaluated in canonical, passive, object-cleft, subject-cleft and role-reversed form, with both actor and patient questions. A whole event is correct only when all ten answers are right.

After 300 wake updates, every continuation arm starts from the *same* saved wake weights. The extra-ordinary arm gets 200 ordinary batches. The augmentation arm gets 100 ordinary plus 100 transformed-pair batches with answer cross-entropy. S2 offline gets the same 100+100 batches with additional symmetric role-distribution consistency and transformed batches last. S2 interleaved uses the **same batch IDs and losses** as offline, alternating ordinary and transformed batches. All use 64-row batches and a constant learning rate. Extra-ordinary uses fewer input tokens per batch than pair arms, so it is a compute/step diagnostic rather than an exact token-matched control. Augmentation, S2 offline and S2 interleaved each make two forwards on transformed-pair batches.

## Observed development outcomes

| Arm | Correct whole events / 90 | Canonical role answers | Passive | Object cleft | Subject cleft | Reversal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake/stop | 0 | 180/180 | 0/180 | 6/180 | 12/180 | 180/180 |
| 200 extra ordinary batches | 0 | 180/180 | 0/180 | 6/180 | 8/180 | 180/180 |
| Ordinary transformed-data augmentation | **90** | 180/180 | 180/180 | 180/180 | 180/180 | 180/180 |
| S2 offline consistency | **90** | 180/180 | 180/180 | 180/180 | 180/180 | 180/180 |
| S2 interleaved consistency | **90** | 180/180 | 180/180 | 180/180 | 180/180 | 180/180 |

The wake headroom screen passed: canonical accuracy was 100% and joint L2 accuracy 0%. **The next gate failed:** offline S2 did not beat augmentation or interleaved S2; all three hit ceiling on the same development items. The improvement over wake is therefore fully compatible with exposure to correctly labeled transformed examples. There is no evidence from this pilot that a distinct offline phase adds value.

The role-reversal score was already perfect after wake because both directions of each *observed* pair were included in canonical training. The reversal test on *unseen* pairs still checks compositional transfer, but it was not the hard form here. The hard forms were unseen sentence templates; once examples of those templates were supplied for observed pairs, transfer to development pairs became easy. A finite construction-aware rule also solves this benchmark perfectly (see [Stage B card](../relational_consolidation_stage_b_2026-10-12/BENCHMARK_CARD.md)). These facts make the result a **benchmark limitation**, not a negative theorem about consolidation.

## Decision

Stop this V1 schedule comparison after the first seed. The protocol required an offline advantage over both augmentation and interleaving in both development seeds to proceed; a three-way ceiling tie in seed 4101 gives no informative positive contrast. Do not test confirmation families or run Qwen on this benchmark. The next synthetic task must withhold a **combination of structure and lexical/contextual cues** so ordinary augmentation does not trivially solve every development event, while retaining deterministic semantic proofs and simple shortcut baselines. Freeze that next design before model training. Human review remains necessary before a natural-language claim.

## Reproducibility

[Run config](runs/seed_4101/RUN_CONFIG.json), [wake trace and score](runs/seed_4101/WAKE_RESULT.json), [full arm results](runs/seed_4101/RESULT.json), the saved wake and arm weights, and [runner source](train.py) are preserved. The run config records protocol, source-data and code hashes; each model result records the checkpoint hash. The source files read were only Stage B family 0 `TRAIN_EVENTS.jsonl`, `DEVELOPMENT_EVENTS.jsonl`, and `S2_TRAIN_DERIVATIONS.jsonl`. The script explicitly rejects other family IDs. PyTorch 2.13.0 ran on CPU. No reserved active/passive evaluation or human-review answer file was read.
