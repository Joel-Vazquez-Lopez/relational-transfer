# Read-only shortcut audit of the adapter development diagnostic

**Status:** post hoc audit of the completed [frozen development outputs](../adapter_readout_diagnostic_2026-10-15/). No model inference, training, prompt revision, or change to the frozen scores occurred. [AUDIT.json](AUDIT.json) gives counts, valid-only denominators, paired gains/losses, and every world/seed unit. [audit.py](audit.py) records the rules and source hashes.

## Fixed predictions before inspecting each construction

Each sentence contains two named participants exactly once. In active sentences, actor is first; in passive and patient-fronted clefts, actor is last. Thus a **fixed last-mentioned-name → Actor** rule predicts wrong active answers but correct passive and cleft answers. A true role-binding rule predicts first → Actor in active and last → Actor in passive and cleft. The event-clause grammatical-subject rule predicts actor for active and cleft, but patient for passive; for the cleft, the matrix `It` is excluded and the lexical subject of `whom A praised/criticized` is A. These rules were applied unchanged to every form and arm, including incorrect and malformed responses in the all-output denominator.

## Observed extraction mappings

`F` means `Actor = first-mentioned name; Patient = last`; `L` is the reverse. All-output counts are shown as F / L / invalid-or-other. Correctness and valid-only denominators are in AUDIT.json.

| Arm | Active | Passive | Patient cleft | Global L agreement |
|---|---:|---:|---:|---:|
| Untouched (96/form) | 21 / 71 / 4 | 26 / 69 / 1 | 7 / 87 / 2 | 227/288 (78.8%); 227/281 valid (80.8%) |
| Ordinary (192/form) | 98 / 94 / 0 | 23 / 169 / 0 | 27 / 165 / 0 | 428/576 (74.3%); 428/576 valid |
| Ordered-role (192/form) | 80 / 101 / 11 | 52 / 136 / 4 | 25 / 163 / 4 | 400/576 (69.4%); 400/557 valid (71.8%) |
| Participant-only (192/form) | 73 / 108 / 11 | 50 / 136 / 6 | 22 / 165 / 5 | 409/576 (71.0%); 409/555 valid (73.7%) |

For the untouched model, this one fixed positional rule explains the high patient-cleft extraction score **and** its active-form reversal: it gives L on 87/96 clefts and 71/96 active sentences. It is stronger than the event-subject mapping (134/288 all outputs) on this panel. The ordinary adapter reduces the active-position failure, but none of the arms uses a single error-free role policy.

Extraction outputs were almost always one of the two complete orderings; malformed outputs account for most exceptions. Both-role correctness among **all** outputs is therefore exactly F on active and L on passive/cleft. For example, untouched active is 21/96 correct and 21/92 valid, while untouched cleft is 87/96 correct and 87/94 valid. The analogous all/valid denominators are preserved for each construction, interface, and arm in AUDIT.json.

## Same event across all three forms

Each underlying pair × direction × verb event has an active, passive, and cleft sentence. `FLL` is the position pattern of correct role binding across all three. `LLL` applies the fixed last-mentioned-name shortcut across all three. Denominators are **32 events per world/seed**, not three independent sentences.

| World / arm | Untouched FLL / LLL | Ordinary 830211 / 830212 FLL / LLL | Ordered 830211 / 830212 FLL / LLL | Participant 830211 / 830212 FLL / LLL |
|---|---:|---:|---:|---:|
| fresh1 | 10 / 9 | 15 / 8; 15 / 10 | 15 / 5; 13 / 4 | 12 / 10; 11 / 4 |
| fresh2 | 8 / 14 | 17 / 8; 17 / 7 | 13 / 4; 15 / 7 | 12 / 4; 16 / 10 |
| fresh3 | 3 / 19 | 17 / 11; 17 / 12 | 14 / 8; 9 / 14 | 15 / 9; 6 / 16 |
| **Pooled descriptive** | **21/96 FLL; 42/96 LLL** | **98/192 FLL; 56/192 LLL** | **79/192 FLL; 42/192 LLL** | **72/192 FLL; 53/192 LLL** |

This complete-event check changes the simple “ordinary fine-tuning damages role ability” story. Although ordinary training lowered the archived patient-cleft actor score, it increased all-form extraction-correct events from 21/96 untouched to 98/192 across two matched seeds, with gains in every world/start. The baseline is evaluated **once per world** and reused only as the within-world comparator; these pooled counts are descriptive and should not be treated as independent replications or a new confirmed training benefit.

## Direct questions, paired changes, and format

The exact direct patient question was **“Who received the action described in the sentence?”** For `praised` and `criticized`, the recipient is the praised or criticized person, but “received the action” is abstract and less natural than “Who was praised/criticized?” It may invite a different reading or encourage copying the actor-answer pattern. We did not revise it. On patient clefts, the direct patient response was validly formatted in **all** outputs but named the actor on **87/96 untouched, 164/192 ordinary, 159/192 ordered, and 161/192 participant-only**. Thus the cleft patient error is primarily a semantic/output-selection error under this wording, not a strict-parser artifact.

On the same cleft sentences, extraction versus direct actor gained/lost **13/5** untouched, **48/17** ordinary, **54/13** ordered, and **40/15** participant-only. For direct patient, gains/losses were **80/2**, **141/4**, **135/5**, and **136/2**. Joint correctness gained **84/0**, **161/0**, **160/0**, and **160/0**, respectively. The joint gain is dominated by the weak direct patient interface and the extraction position policy; it does not isolate hidden relational recovery.

Compared with the one world baseline on the same 96 sentences, all-form extraction joint accuracy changed by **+29.2 pp ordinary, +19.8 ordered, +16.1 participant** on active; **+16.1, −1.0, −1.0 pp** on passive; and **−4.7, −5.7, −4.7 pp** on clefts. Those differences are averages over two seeds per world; AUDIT.json reports every matched world/seed gain and loss. The cleft-only view would miss the larger active improvement and the fact that a fixed last-name mapping already performs well on clefts.

Malformed direct patient answers often included a single participant name plus an explanatory sentence. A separate sensitivity inventory finds one uniquely identifiable pair name in **117/117** malformed untouched responses (81 gold patient, 36 actor), **12/12** ordinary (4/8), **18/18** ordered (9/9), and **84/84** participant-only (52/32). Extraction malformed responses contained both participant names in **7/7**, **0/0**, **19/19**, and **21/21**, so that simple unique-name salvage is ambiguous. These are not retrospective rescoring: all frozen primary metrics and invalid-output rules remain as originally reported. The raw examples and all-output/valid-only rates for every interface are in AUDIT.json.

As a conservative **format sensitivity only**, counting the unique participant name in those malformed direct-patient responses would change all-form patient-name matches from **47/288 to 128/288** untouched, **124/576 to 128/576** ordinary, **203/576 to 212/576** ordered, and **165/576 to 217/576** participant-only. A full sentence can still be pragmatically ambiguous despite one identified name, so these are possible name matches, not corrected primary scores. On patient clefts, every direct-patient response was already validly formatted; this sensitivity adjustment changes **none** of the cleft counts. The frozen “semantic accuracy” label therefore combines name correctness with compliance to its specified answer format, especially on active and passive forms.

## Conclusion and next discriminating test

The strongest observation is the **same-event crossover**: untouched extraction is more often `LLL` (42/96) than semantically consistent `FLL` (21/96), even while its patient-cleft extraction score is high. Ordinary fine-tuning shifts many events toward FLL, so its cleft-only regression and all-form extraction improvement coexist. The audit supports a simple position/output-allocation shortcut as an important explanation of the apparent cleft recovery. It cannot show that any arm has or lacks an internal role representation.

A single next development experiment should balance construction with **requested output-field order** (`Actor; Patient` versus `Patient; Actor`) on identical fresh sentences and compare those extraction responses with natural verb-specific direct questions (“Who was praised?” / “Who was criticized?”). Pre-freeze the full all-form event score and positional-versus-semantic predictions. This would distinguish a fixed input-last-name policy from output-order copying or semantic binding more sharply. No such experiment was run here.
