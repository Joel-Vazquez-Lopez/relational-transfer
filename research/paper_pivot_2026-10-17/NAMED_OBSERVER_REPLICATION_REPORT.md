# One candidate behavioral law under a serious rival

17 October 2026. All three models were existing local MLX 4-bit weights. No paid API, model download, activation patch, or natural-corpus confirmation. [Frozen protocol](NAMED_OBSERVER_PROTOCOL.json), [72 item variants](NAMED_OBSERVER_ITEMS.json), [script](named_observer_replication.py), [all 1,296 raw responses](NAMED_OBSERVER_RAW.jsonl), and [machine report](NAMED_OBSERVER_REPORT.json) are preserved.

## Test

The preceding [case-by-bystander discriminator](DISCOURSE_LOAD_FINAL_REPORT.md) found 7/32 wrong answers when Qwen7B had to retrieve the second event's participant from passages with a preposed named bystander, versus 0/32 without bystanders. Every single-sentence query passed. The specific alternative tested here was **added clause/length**, not the extra named entity: if that alternative is sufficient, a clause containing an unnamed observer should be similarly disruptive.

Before inference, the new protocol froze 24 fresh base event pairs using six new action/object combinations, two epistemic conditions, and two clause frames. Each base had a bare passage, a same-position *unnamed observer* clause, and a *named observer* clause; all three received identical single-sentence questions and paired questions in both orders. The first frame put the observer clause before the target action (`While [observer] examined..., [agent] acted`), the second put it after (`[agent] acted while [observer] watched`). Qwen7B was the primary model; Llama3B and Qwen1.5B were capacity/family checks. Gold strings, generation and exact scoring were locked before any model output.

## Full scores

| Model | Condition | Single exact /144 per model, by condition /48 | Joint exact /96 | Target first /48 | Target second /48 | Wrong-event answers | Observer answers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen7B | Bare | 48/48 | 96 | 48 | 48 | 0 | 0 |
| Qwen7B | Unnamed observer | 48/48 | 96 | 48 | 48 | 0 | 0 |
| Qwen7B | Named observer | 47/48 | **91** | 48 | **43** | 4 | 1 |
| Llama3B | Bare | 48/48 | 96 | 48 | 48 | 0 | 0 |
| Llama3B | Unnamed observer | 48/48 | 92 | 46 | 46 | 4 | 0 |
| Llama3B | Named observer | 40/48 | 85 | 43 | 42 | 3 | 8 |
| Qwen1.5B | Bare | 48/48 | 71 | 40 | 31 | 25 | 0 |
| Qwen1.5B | Unnamed observer | 47/48 | 59 | 36 | 23 | 37 | 0 |
| Qwen1.5B | Named observer | 45/48 | 51 | 42 | 9 | 34 | 11 |

The Qwen7B named-observer condition has 23/24 item variants passing both single-sentence probes. Its one single failure was on a *postposed* report item and did not produce any of the five joint failures. All five Qwen7B joint errors were on **component-competent preposed** variants with the requested event in **sentence two**. The same model got **24/24** target-second queries in the postposed named-observer frame and **24/24** in the preposed unnamed-observer frame. Four of five wrong answers copied the other event's participant; one copied that event's named observer. Both negated and unverified-report items failed (three and two errors respectively).

## Decision against frozen predictions

The primary frozen practical gate required at least **8/48** more wrong Qwen7B target-second answers for named than unnamed observers, and a positive effect in *both* clause frames. The result was **5/48**, entirely preposed. **The broad named-entity-competition law fails that gate.** We do not pool models to rescue it: Llama's named condition has eight single-probe failures and mostly observer answers, while Qwen1.5B is already weak on bare joint prompts. They do not provide the same error mechanism.

One narrower phenomenon survives descriptively. On the previous fresh proper-name set, preposed named bystanders yielded **7/32** Qwen7B target-second errors versus 0/32 bare; on this new action/name set with a different preposed clause and an unnamed-observer control, the corresponding score was **5/24** versus **0/24 unnamed**. That is **12/56** errors across two synthetic panels with similar rates (21.9% and 20.8%). This comparison is **post hoc at the level of the narrowed claim**: the second protocol predicted both preposed and postposed effects, and the postposed prediction failed. The panels are not independent natural-language samples, so 12/56 must not be presented as a population rate or a preregistered replication statistic.

## What the competing explanation survived

The unnamed-observer control shows that mere extra clause length is insufficient for Qwen7B in these particular preposed templates: it remained perfect. But a **position-and-entity cue mixture** can explain the observations without a discrete event selector. The preposed named clause adds an eligible proper name before each action; the model may fall back to the first event's salient participant when the query requests the second. Postposing the observer removes the Qwen7B failure. We have no internal intervention that distinguishes this account from a semantic event representation, and the earlier [identifiability NO-GO](../decisive_witness_gate_2026-10-17/REPORT.md) still applies.

The scientific next step is a *single prospective natural-language study* of this **preposed named-observer, second-event retrieval** boundary: source-valid event pairs, independently reviewed role/status scope, generic and named clause controls matched within item, two clause placements, multiple competent models, and a locked cue-mixture baseline trained outside the final documents. A strong paper would require that this prediction generalize beyond our synthetic templates and beat the matched baseline. The current data are a lead, not a publishable law.

Prior work already treats [cross-event argument confusion](https://aclanthology.org/2024.findings-acl.328/), [position bias in event extraction](https://aclanthology.org/2024.findings-acl.487/), [negated entity tracking](https://aclanthology.org/2022.naacl-main.71/), and [mixed positional and lexical binding routes](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf). The narrower preposed interaction is not automatically new relative to that literature.
