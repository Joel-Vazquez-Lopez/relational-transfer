# Output-field-order discriminator: exploratory development results

**Completed 16 October 2026.** The [protocol](FROZEN_PROTOCOL.json), [288 source sentences and exact prompts](FROZEN_ITEMS.json), parser tests, model/checkpoint hashes, and [runner plus analysis code](run.py) were locked before inference. Six local Qwen2.5-0.5B units used three worlds, each with the untouched base and its ordinary adapter at seed 830211. This reuses **unsealed synthetic development items** from the prior adapter diagnostic; it is not an independent replication or a test of internal representations. All 576 original `Actor; Patient` replays matched the archived generated token IDs exactly, including padding after EOS. No training, downloads, paid calls, or closed natural datasets were used.

## Main result: reversed order disrupts the output format

`AP` requested `Actor: <name>; Patient: <name>`. `PA` changed only the requested field sequence to `Patient: <name>; Actor: <name>`. Both used identical sentences and a 20-token greedy cap. “Valid” requires both labeled fields exactly once, in either order; order compliance is scored separately. Correctness below requires **both** names under their correct labels; malformed outputs are wrong under the frozen rule.

| Arm | AP both correct | PA both correct | PA labeled/valid | PA obeys requested order | PA unlabeled two-name outputs |
|---|---:|---:|---:|---:|---:|
| Untouched, 3 worlds (288) | 177/288 (61.5%) | 19/288 (6.6%) | 53/288 | 51/288 | 235/288 |
| Ordinary, 3 worlds (288) | 215/288 (74.7%) | 55/288 (19.1%) | 108/288 | 57/288 | 178/288 |

The PA prompt commonly produced an unlabeled pair such as `Emma; Ryan`; every one of the 235 untouched and 178 ordinary outputs in that category contained exactly the two sentence participants, but **no actor/patient labels**. They were not assigned roles or credited. Another 51 ordinary PA responses supplied both labels in the old AP order despite the requested reversal. There were no responses cut off at 20 generated tokens. Total CPU model inference time was 8.35 minutes for 2,304 new generated responses.

| World | Base AP / PA both correct (of 96) | Ordinary AP / PA both correct (of 96) | Base PA valid / compliant | Ordinary PA valid / compliant |
|---|---:|---:|---:|---:|
| fresh1 | 58 / 8 | 67 / 34 | 18 / 18 | 58 / 22 |
| fresh2 | 63 / 4 | 72 / 5 | 9 / 9 | 10 / 10 |
| fresh3 | 56 / 7 | 76 / 16 | 26 / 24 | 40 / 25 |

The [locked analysis](ANALYSIS.json) reports AP→PA mapping transitions, actor and patient accuracy, first emitted name position, exact-format rates, token counts, and all paired gain/loss counts by world, arm, and construction. Relative to the one base in each world on the same sentences, ordinary AP both-role accuracy gained/lost 11/2, 12/3, and 22/2 sentences in fresh1–3; PA gained/lost 28/2, 2/1, and 10/1. These are descriptive paired comparisons with three graph worlds, not independent-sentence significance tests.

## Conditional order-following sensitivity

The [post-lock sensitivity plan](SECONDARY_ANALYSIS_PLAN.md) was declared before inference. Its [read-only results](SENSITIVITY.json) separate two denominators so that **ignoring the PA order request cannot masquerade as role-label invariance**:

| Arm | Both outputs valid | Both valid **and** obey their different requested orders | Invariant role assignments among compliant pairs | Same emitted name sequence among compliant pairs |
|---|---:|---:|---:|---:|
| Untouched | 53/288 | 51/288 | 11/51 | 40/51 |
| Ordinary | 108/288 | 57/288 | 19/57 | 38/57 |

Among the small jointly compliant subset, the two emitted names usually stay in the same sequence when field labels reverse. That flips the actor/patient assignments in **40/51 untouched** and **38/57 ordinary** pairs. This is more consistent with a local output-slot sequence tendency than with invariant semantic labels **within this selected subset**. The selection is severe: most PA outputs are unlabeled, so the conditional pattern cannot be projected onto all 576 comparisons. The all-output result chiefly shows an instruction/format failure after changing field order.

## Same event across active, passive, and cleft

`FLL` means actor/patient assignments are correct under the three forms of one pair × direction × verb event; `LLL` means a fixed last-mentioned-name actor mapping across forms. There are **32 dependent events per world/arm**. The primary strict complete-event gate requires FLL in **both** output orders.

| World | Base AP FLL / PA FLL / both orders | Ordinary AP FLL / PA FLL / both orders |
|---|---:|---:|
| fresh1 | 10 / 0 / 0 | 15 / 2 / 2 |
| fresh2 | 8 / 0 / 0 | 17 / 0 / 0 |
| fresh3 | 3 / 0 / 0 | 17 / 0 / 0 |

Across worlds, that is **0/96 untouched** and **2/96 ordinary** events correct under both orders and all three forms. The low PA format validity limits what this metric can say about roles. Requested field order was independently crossed **within each fixed sentence**. Construction and gold actor mention position were still linked by these templates (active actor first; passive/cleft actor last), so this is not a fully crossed construction × actor-position factorial. Name identity and actor direction were balanced.

## Verb-specific direct-question companion

The fixed questions were `Who praised someone?` / `Who criticized someone?` for actor and `Who was praised?` / `Who was criticized?` for patient. The system still required exactly one name. Strict one-name correctness, with malformed outputs wrong:

| Arm | Actor QA | Patient QA | Actor QA active / passive / cleft | Patient QA active / passive / cleft |
|---|---:|---:|---:|---:|
| Untouched | 157/288 | 116/288 | 12 / 62 / 83 of 96 | 59 / 44 / 13 of 96 |
| Ordinary | 253/288 | 99/288 | 95 / 91 / 67 of 96 | 25 / 53 / 21 of 96 |

The actor and patient question rankings move in different directions with ordinary adaptation; this is another interface-sensitive behavior, not an isolated role-storage result. Two direct QA calls and one joint extraction call are **not compute matched**. The QA results should not be used as a causal cost-controlled comparison to extraction.

## What this rules in and leaves open

The experiment demonstrates a large **behavioral dependence on requested output order**: reversing field order often removes field labels altogether, and among the small fraction that obey both orders, emitted name sequence frequently persists while role assignments flip. It does not establish that the model lacks semantic roles internally. The PA instruction changes visible token order and the decoder's local answer contract; a competent parser of the sentence could coexist with a brittle output formatter. The earlier high patient-cleft extraction score remains vulnerable to the fixed-last-name shortcut identified in the [shortcut audit](../adapter_shortcut_audit_2026-10-15/RESULTS.md). Further conclusions require a design with better PA format compliance, frozen before looking at results, and preferably a fully crossed construction × actor-position panel.
