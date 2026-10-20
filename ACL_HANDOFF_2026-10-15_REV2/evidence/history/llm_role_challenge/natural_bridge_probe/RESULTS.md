# Natural-language Transformer bridge: first diagnostic

**12 October 2026. Exploratory, not a repair result.** The [protocol](PROTOCOL.json) was written before model inference. This run used only the [official UD English EWT](https://github.com/UniversalDependencies/UD_English-EWT) train and development splits. It did not read the reserved 32-pair human-review challenge or EWT test split.

## Why this run

The existing natural Qwen adaptation used 256 QA-SRL Bank training events. In that selected set, the actor precedes the other participant in **244/256** events. The separate selected development and Bank-test sets have actors first in **74/75** and **126/130** events. Their role scores can therefore be influenced heavily by word order. This is a property of the selected data, not proof that a model used the shortcut.

I tested whether a smaller pretrained Transformer has usable signals for role order in natural sentences when its training examples contain both active and passive voice. I froze DistilBERT and trained a regularized linear probe on its contextual states at two annotated participant heads and the target verb. The probe selects which participant is the actor. It does not generate an answer span or change the Transformer weights.

## Data and comparison

The restricted extraction requires exactly one subject and object dependent of the target verb for an active case, or one passive subject and an explicit `by`-agent for a passive case. It also requires a sentence of at most 45 tokens. This yielded **4,873 active and 120 explicit-agent passive** training events, plus **632 active and 16 passive** development events. These are syntactic annotations. An `obj` is not automatically a semantic patient.

Three fixed samples compared equal-sized training sets: **240 active examples** versus **120 active plus 120 passive examples**. Both conditions used the same frozen DistilBERT and probe method. The development items were fixed before inference.

| Development cases | Active-only training | Voice-balanced training | Simple first-participant rule |
| --- | ---: | ---: | ---: |
| Active, 632 items | 617–623 correct; mean **98.05%** | 599–607; mean **95.52%** | 588/632 |
| Passive with explicit agent, 16 items | **0/16** in all three samples | **14–15/16**, mean **91.67%** | 0/16 |
| All 648 items | mean **95.63%** | mean **95.42%** | 588/648 |

The balanced probe also improved the **44 active development cases where the actor is the second participant**: 42–44 correct versus 29–37 for active-only training. But it lost accuracy when the actor was first. Its overall score did not improve.

## What this establishes, and what it does not

The passive change shows that the frozen encoder states contain information a small probe can use once it sees both voices in training. It does **not** establish that the probe learned a general role relation. Every restricted passive development item puts the actor second and has an explicit `by` phrase. A post-hoc surface rule that chooses the second participant when `by` appears between the target verb and that participant gets **16/16 passive** and **588/632 active** items correct. That diagnostic was devised after the run, so it is only a warning about the test's ease.

The passive sample has just 16 items. The two training conditions also have different voice mixes, so the run tests exposure to passive examples, not the controlled ordered-role auxiliary objective. DistilBERT and Qwen use different architectures and answer interfaces. Their scores should not be interpreted as a matched causal comparison.

## Next experiment implied by this result

The final evaluation needs correct answers that are not predictable from `by`, first mention, grammatical subject, or a fixed answer position. For each human-verified reversible event, test meaning-preserving active/passive forms **and** a form that puts the patient first without a passive `by` cue, such as a grammatical cleft. Add a true role-reversed event. Score both roles, both question wordings, and whether all forms of one event are right. Keep the 32-pair active/passive challenge untouched until independent review, and develop any additional forms from source-only text before running the existing checkpoints. Compare shortcut rules alongside models.

Only after that evaluation should a new role-aware training objective be judged. A positive result on active/passive alone would be too easy to mistake for learning a surface rule.

## Audit trail

- [Frozen protocol](PROTOCOL.json) gives source and model hashes, sample IDs, analysis rule, and limitations.
- [Per-item results](RESULTS.json) contain all three training samples and both voices.
- [Probe source](../natural_bridge_probe.py) parses the annotated corpus and runs the frozen encoder and linear probe.
