# Real-data event-pair feasibility audit — 16 October 2026

## Decision

**There are genuine natural-language event pairs for a development pilot, but no ready-made confirmatory panel yet.** MAVEN-Arg is the stronger source. Its public validation set contains 39 automatically eligible Agent/Patient pairs across 23 documents under the broad screen below. Requiring each of the four entity arguments to be mentioned within 100 characters of its own event trigger leaves 12 pairs across 10 documents. Manual review finds clear examples, but also annotation/ontology and event-granularity problems. The numerical counts are candidate ceilings, not accepted gold evaluation items.

WikiEvents provides an independent source, but its public development split has only 20 documents; four pairs across three documents pass a broad same-type, two-role, four-distinct-coreference screen with triggers within 200 tokens. Its training pool is larger. Neither corpus should be called an untouched confirmation set after this audit.

No model inference, training, or hidden-test inspection was performed.

## Sources and reproducibility

- [MAVEN-Arg authors' repository](https://github.com/THU-KEG/MAVEN-Argument) and [paper](https://aclanthology.org/2024.acl-long.224/). The repository links a public `MAVEN-Arg.zip`; only `train.jsonl` and `valid.jsonl` were opened. Its test labels are hidden by the authors.
- [WikiEvents authors' repository](https://github.com/raspberryice/gen-arg) and [paper](https://arxiv.org/abs/2104.05919). The public `train.jsonl`, `dev.jsonl`, and corresponding coreference files were read. The test split was not opened.
- `audit.py` produces the broad MAVEN-Arg funnel in `AUDIT.json`. `focused_audit.py` produces the focused MAVEN-Arg and WikiEvents counts in `FOCUSED_AUDIT.json`. The latter records SHA-256 hashes for the downloaded source files; those files were held in temporary storage and are not redistributed here.

## MAVEN-Arg funnel

The authors' archive has 2,913 train documents with 64,923 events and 710 validation documents with 15,556 events. A broad pair requires the same event type, at least two shared filled roles, exactly one filler per selected role per event, four distinct entity IDs or normalized spans, and two triggers at most 1,000 characters apart. Pair counts are dependent because one document can contribute several pairs.

| Criterion | Train pairs / documents | Validation pairs / documents |
|---|---:|---:|
| Same event type | 43,263 / 2,741 | 11,473 / 673 |
| At least two shared filled roles | 34,379 / 2,663 | 9,332 / 652 |
| Two single-filler roles with four distinct keys | 13,981 / 2,020 | 3,564 / 485 |
| Above, triggers ≤1,000 characters apart | 10,369 / 1,980 | 2,494 / 476 |

The broad funnel is only an upper bound. It mixes different kinds of roles, non-entity spans, distant arguments, and sometimes related mentions of what is effectively one larger event.

The focused screen fixes **Agent** and **Patient**, requires exactly one entity ID per role and four distinct IDs across two same-type events, and places their closest triggers within 1,000 characters. A local variant additionally requires a mention of each argument within 100 characters of its own event trigger.

| Focused criterion | Train pairs / documents | Validation pairs / documents |
|---|---:|---:|
| Agent/Patient, four distinct entity IDs, nearby triggers | 250 / 121 | 39 / 23 |
| Plus all four arguments locally mentioned (≤100 characters) | 88 / 56 | 12 / 10 |
| Conquering type, four distinct IDs, triggers ≤500 characters | 38 / 27 | 7 / 5 |

Entity IDs reduce obvious alias duplication, but distinct IDs do not guarantee semantically distinct referents or correct labels. A local mention window reduces long-distance ambiguity but is a design choice, not an annotation-quality guarantee.

## WikiEvents cross-check

This screen requires two events of the same type, two roles each with exactly one *coreference cluster* filler per event, four distinct clusters, and the stated trigger distance. The source uses token offsets rather than MAVEN-Arg's character offsets. Roles are not restricted to Agent/Patient, so these counts are not directly comparable to the focused MAVEN-Arg row.

| Criterion | Train pairs / documents | Dev pairs / documents |
|---|---:|---:|
| Same type, two roles, four distinct coreference fillers | 496 / 66 | 11 / 6 |
| Plus triggers ≤200 tokens apart | 186 / 50 | 4 / 3 |
| Plus triggers ≤100 tokens apart | 104 / 36 | 1 / 1 |

## Manual spot check of MAVEN-Arg validation candidates

I inspected all 12 candidates in the focused 100-character validation subset. These are informal judgments by one reviewer, not adjudicated annotations or a new certified subset.

- **Clear examples:** `2014 Men's Hockey World Cup` has “Australia ... defeating the Netherlands” and “Argentina ... defeating England” in adjacent sentences. `Extreme Rules (2012)` contains three separate wrestler-vs-wrestler defeats. `2009 Men's Asia Pacific Floorball Championships` contrasts Japan–India with Poland–Ukraine. These have obvious event triggers and distinct role fillers.
- **Other usable-looking examples requiring review:** `Brabant Revolution` has two support events; `2010 IAAF World Cross Country Championships` has two “led [team]” events farther apart; `First Battle of the Isonzo` has two nearby “retook [place]” events. The target question and passage span need careful wording.
- **Reasons to reject or adjudicate:** `Battle of Rastatt (1796)` pairs a clash with an invasion under one broad type; `Jin–Song Wars` includes a pronoun and a questionable Patient link; `2013 World Junior Ice Hockey Championships` has a “win” annotation whose Agent is hard to reconcile with the nearby sentence; `Action of 18 September 1810` spans distant, differently scoped military events. The type label and distinct IDs alone do not make these clean contrasts.

The 12 pairs are therefore **not** 12 verified test cases. Multiple pairs from `Extreme Rules (2012)` share one document and should not be treated as independent observations.

## Research implication

This audit establishes a viable source for **constructing** a real-data pilot. It does not establish enough independent, reviewed, same-schema event pairs for a high-confidence held-out mechanistic result. The next data step is to manually adjudicate candidate passages, exclude documents already inspected here from any blind confirmation set, and predefine event identity, role questions, acceptable answer aliases, and rival model predictions. A proper paper-level test may need a larger review sample, a second source, or a bespoke annotation layer on top of existing corpora.
