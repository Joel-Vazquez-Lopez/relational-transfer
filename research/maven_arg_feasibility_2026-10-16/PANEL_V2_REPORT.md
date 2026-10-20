# Gemma event × role replication on 17 new documents — 16 October 2026

**Follow-up:** The [open-weight Qwen 7B gate](OPEN_MODEL_GATE.md) and [next-study memo](NEXT_COOK.md) use this panel to choose a causal research direction.

## Main result

I built a second, document-disjoint panel from **17 public MAVEN-Arg training documents**. Each passage has two marked events of the same annotated type and one annotated Agent and Patient per event: **68 questions**. These documents were absent from the six-document validation pilot. One reviewer screened the source text and annotations before inference, excluding conspicuously mismatched event/role pairs. This is a stronger **development replication**, not an independently adjudicated or blind confirmation set.

Gemma-4-31B-it gave the exact annotated local string on **46/68** questions. Under a **whole-token, unique gold-alias containment rule frozen before these requests**, it matched **63/68**. It answered all four questions on **13/17 documents** under that rule. The two small local Qwen models were run on the same frozen items and wording. Their comparison protocol was prepared after the Gemma outcome was visible, so the cross-model analysis remains exploratory.

| Model | Exact local string | Exact or unique target alias | Same-event wrong-role | Identical Agent/Patient answer |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct, local | 20/68 | 27/68 | 30/68 | 31/34 events |
| Qwen2.5-1.5B-Instruct, local | 37/68 | 42/68 | 21/68 | 21/34 events |
| Gemma-4-31B-it, Berget | **46/68** | **63/68** | **3/68** | **3/34 events** |

These are task results, not a controlled scaling curve: the models differ in family, size, template, and serving stack. Questions within one document are dependent. No p-value or confidence interval is claimed from 68 question rows.

| Annotated event type | Documents | Gemma alias-rule score | Qwen 1.5B | Qwen 0.5B |
|---|---:|---:|---:|---:|
| Conquering | 8 | 31/32 | 22/32 | 12/32 |
| Military_operation | 3 | 12/12 | 5/12 | 3/12 |
| Bringing | 1 | 4/4 | 3/4 | 2/4 |
| Control | 5 | 16/20 | 12/20 | 10/20 |

The `Military_operation` items here contain sporting defeats; the table retains the corpus ontology rather than relabeling them after the run.

## What the five Gemma misses were

- **Three same-event wrong-role answers:** For Battle of Ostrach, Battle of Ruona, and Fall of Saigon, a generic Patient question returned the commander rather than the commanded force.
- **One unmatched answer with plausible target meaning:** In Battle of Ruona, Gemma answered `5,000 men` where the annotation says `Swedish army`. This may refer to the same force, but the frozen string rule does not accept it.
- **One annotation-granularity conflict:** The passage says Kuchlug's `force of 30,000` was defeated; MAVEN-Arg marks `Kuchlug` as Patient. Gemma answered `his force of 30,000`. The model's answer tracks the surface sentence, while the gold label uses the commander as the affected entity. I retain this as an error under the frozen rule.

The other 17 non-exact Gemma outputs accepted by the alias rule use phrases such as `the West Indies`, `Captain James Lawrence`, or `the Central Powers' combatants`. The predeclared unique-alias rule prevents accepting an answer that also names another one of the four marked participants. It is still a string heuristic, not independent semantic adjudication.

The same heuristic also accepts one verbose, token-truncated Qwen 0.5B response because it contains `Serena Williams`. Thus its 27/68 alias score includes one answer that violated the requested copy-only format; the exact-string score remains 20/68.

## Adaptive wording diagnostic

After inspecting the main results, I froze and ran a separate diagnostic on **all 20 Control questions**. It asked for the *commander* or the *group, army, or vessel under that command*. Gemma moved from **16/20 to 20/20** under the same alias rule. Three wrong-role answers and one unmatched answer became target-alias matches; no previously accepted answer became wrong. This supports the interpretation that the generic “acted on or targeted” Patient question was a substantial part of the Control difficulty on this panel. Since the wording was designed after seeing the errors and retested on the same items, **20/20 is adaptive development evidence, not confirmation**. It must be tested unchanged on fresh Control documents.

## Construction, scoring, and reproducibility

The source is [MAVEN-Arg](https://github.com/THU-KEG/MAVEN-Argument). The public `train.jsonl` was read from the authors' archive; hidden test labels were not opened. A deterministic structural screen found 88 eligible Agent/Patient pairs across 56 training documents. It required two same-type events, one entity filler per role per event, four distinct entity IDs, nearby triggers, and a local mention of each argument. I inspected candidate passages and selected one pair each from 17 distinct documents before model inference. The [selection manifest](PANEL_V2_SELECTION.json), [builder](build_panel_v2.py), and [items](PANEL_V2_ITEMS.json) preserve that choice. Selection by one investigator, and omission of difficult or noisy candidates, limits generalization.

The [main protocol](PANEL_V2_PROTOCOL.json) was frozen before the 68 Gemma requests (SHA-256 `e4ecf529a111637418d31761adb547bd86858e4a26fc04c3226aabc02d547f0d`). It locked the items, fixed generic questions, greedy generation, and both scores. The primary score is normalized exact match to the local annotated mention. The secondary score accepts an answer containing a whole-token gold alias only when no alias of the other three event-role entities is contained; otherwise it classifies the answer by wrong event/role or as unmatched. The [local comparison protocol](PANEL_V2_LOCAL_PROTOCOL.json) was frozen before local Qwen inference, but after Gemma results were seen. The [adaptive Control protocol](CONTROL_WORDING_PROTOCOL.json) was likewise frozen before its calls, but after main-panel inspection.

Raw outputs and scoring: [Gemma predictions](PANEL_V2_GEMMA_PREDICTIONS.jsonl), [Gemma analysis](PANEL_V2_GEMMA_ANALYSIS.json), [Qwen 0.5B analysis](PANEL_V2_QWEN05_ANALYSIS.json), [Qwen 1.5B analysis](PANEL_V2_QWEN15_ANALYSIS.json), and [adaptive analysis](CONTROL_WORDING_ANALYSIS.json). The scripts are [main Gemma runner](panel_v2_run.py), [local runner](panel_v2_local.py), and [adaptive runner](control_wording_check.py).

Berget's protected key was read process-locally and not copied into outputs. The main 68 requests all returned `finish_reason=stop`; estimated token cost was **€0.00271**, with **€0.00542** conservatively accounted under a separate €0.10 ceiling. The adaptive 20 requests cost an estimated **€0.000753**, conservatively accounted as **€0.001506** under a separate €0.03 ceiling. These are token estimates and local safeguards, not provider invoices. The provider alias was available at the prior EUR token prices but exposed no pinned weight revision or system fingerprint.

## Scientific reading

The result removes one concern: **stronger LLMs can keep the two natural events and their roles distinct in this setup**. It also sharpens a confound: residual errors on command relations can arise from the question interface. This does not show that Gemma contains a novel event × role circuit, nor that smaller models lack one. Hosted Berget inference exposes text behavior only.

The next decisive step is to freeze the commander wording and alias rules on a **fresh, independently adjudicated Control set**, alongside clear defeat and leadership events from another source or untouched MAVEN-Arg documents. For a neural mechanism claim, the same task must first be solved by an open-weight model whose activations can be intervened on, with event-cue, role-cue, answer-frequency, and position controls. Until those gates pass, the defensible claim is a reproducible behavioral capability difference and a prompt-sensitive error mode.
