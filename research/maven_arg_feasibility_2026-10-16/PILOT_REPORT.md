# Event × role natural-passage pilot — 16 October 2026

**Follow-up:** The same items were subsequently run on hosted Gemma-4-31B-it; see the [Gemma result](GEMMA_RESULT.md). The initial Qwen results below remain unchanged.

## Decision

**We found a workable research object, but this pilot is not a mechanistic discovery.** Six authentic MAVEN-Arg validation passages contain two marked events of the same type, each with an annotated Agent and Patient. On 24 questions (six passages × two events × two roles), untouched local Qwen2.5-0.5B-Instruct answered 7 with the exact annotated local mention; Qwen2.5-1.5B-Instruct answered 12. An adaptive, more event-specific wording check on the 1.5B model answered 13. The latter change is too small and inconsistent to explain the failures as merely a generic-question wording problem.

The clearest behavioral pattern is **role confusion within the selected event**. The 1.5B model returned the opposite annotated role on 7/24 initial questions and 9/24 adaptive questions. It gave identical answers to both Agent and Patient questions for 9/12 event pairs initially and 7/12 with adaptive wording. That is a concrete phenomenon to investigate. It does **not** demonstrate a specific internal event representation, its absence, or a novel circuit.

## Materials and protocol

The source is the public [MAVEN-Arg authors' repository](https://github.com/THU-KEG/MAVEN-Argument). Only training and validation annotations were inspected; the hidden-label test split was not used. The six validation documents were manually screened by the investigator, so every result here is **development evidence**, not an independent held-out estimate. Passages retain the natural sentence text with two trigger markers added. Four requested fillers per passage were taken from the dataset annotations. The [feasibility audit](FEASIBILITY_REPORT.md) explains the larger candidate pool and its quality limits.

The [initial protocol](PILOT_PROTOCOL.json) was frozen before model inference (SHA-256 `d57377920a074d8436248f0d976d761a3671118752e38806f74b8acff3e38e02`). It specified fixed generic Agent/Patient questions, greedy generation of up to 24 tokens, and exact normalized matching to the annotated local mention as primary score. Both models ran locally on CPU in float32. No model training, paid inference, or activation intervention was performed. The [adaptive wording protocol](PILOT_V2_PROTOCOL.json) was frozen **after** the initial outputs were inspected (SHA-256 `c1b8ef63909452876841ef490c574c8192ade1d90c324d1a7964ae38c3e0548d`), so it is a diagnostic, not confirmation.

| Condition | Exact local mention | Same-event wrong role | Other / unmatched | Identical Agent–Patient outputs |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B, fixed generic wording | 7/24 | 7/24 | 10/24 | 12/12 |
| Qwen2.5-1.5B, fixed generic wording | 12/24 | 7/24 | 5/24 | 9/12 |
| Qwen2.5-1.5B, adaptive event-specific wording | 13/24 | 9/24 | 2/24 | 7/12 |

The 0.5B “other” cell contains two wrong-event same-role, two wrong-event wrong-role, and six unmatched/ambiguous outputs. In the initial 1.5B condition, the five unmatched outputs include shortened or longer expressions for apparently correct referents (for example, `Gebremariam` for `Gebregziabher Gebremariam`). The two adaptive unmatched outputs also appear to refer to the target (`Gebremariam`; `United States` for `the United States`). Exact match therefore **understates plausible referent accuracy**, but changing the acceptance rule after seeing outputs would bias the declared score. The raw answers and the frozen scoring categories are in [initial analysis](PILOT_ANALYSIS.json) and [adaptive analysis](PILOT_V2_ANALYSIS.json).

The paired adaptive comparison is mixed: four unmatched answers became exact, three exact answers became wrong-role answers, and one wrong-role answer became unmatched. Notably, on the hockey passage the 1.5B model initially answered `Australia` correctly as the Agent of “defeating the Netherlands”; with the direct “Who defeated?” wording, it answered `Netherlands`. The model also repeatedly returned `Japan` for both roles of Japan defeating India, `Prussia` for both roles of Prussia's support, and `Church` for both roles of the Church supporting the Statists. These outputs warrant investigation, but the prompt changes alter other factors too, including phrasing and answer tendency.

## Interpretation and limits

The model often identifies *a participant of the right event* but fails to keep the role straight. That motivates an event-and-role binding test more sharply than generic extraction accuracy. However, this six-document panel cannot distinguish an internal binding failure from answer-generation bias, lexical priors, instruction-following limits, or item-specific ambiguity. The 0.5B and 1.5B models belong to one family and both perform too poorly for a clean causal localization test. A failed intervention on them would be especially hard to interpret.

The items are not uniformly ideal: “support of Prussia” has a nominal trigger, “led” treats a team as Patient under the annotation schema, the IAAF passage separates its two events by several sentences, and the wrestling passage includes a third unmarked defeat. Sentence cropping and trigger markers also modify the original reading task. These issues should be resolved during adjudication before any stronger claim.

## Next experiment that could produce a real result

1. **Build an independently reviewed panel.** Write acceptance rules for event boundaries, role semantics, span aliases, extra unmarked events, and passage length. Exclude all documents read in this audit and pilot from the final held-out set. Use training documents for development and an untouched, separately adjudicated partition for confirmation. Count documents as the independent units.
2. **Establish competence first.** Run several stronger, accessible open-weight models on the frozen panel. Require high accuracy on both event and role questions, with verified referent scoring and a matched single-event control. A candidate model should clearly exceed this pilot before internal interventions are interpretable.
3. **Cross event and role interventions.** For a competent model, construct four matched prompts with the same passage and target event/role combinations. Separately alter the event cue and role cue; compare their effects on the output and on candidate internal states. The key test is whether an event-directed intervention changes *which event* is answered while preserving role, and a role-directed intervention changes *which role* is answered while preserving event. Include same-token, position, and answer-frequency controls and preregister the expected crossover pattern.
4. **Make the claim conditional on the control results.** If the crossover is absent, report that. If it appears in one model, replicate on held-out documents and another model family. Only then argue for a reusable event × role factorization, rather than treating a suggestive activation plot as discovery.

## Files

- [Curated pilot items](PILOT_ITEMS.json) — modified natural excerpts and gold roles; development data only.
- [Initial protocol](PILOT_PROTOCOL.json), [inference/scoring script](pilot.py), [initial analysis](PILOT_ANALYSIS.json), and raw predictions for [0.5B](PILOT_PREDICTIONS_qwen05.jsonl) and [1.5B](PILOT_PREDICTIONS_qwen15.jsonl).
- [Adaptive protocol](PILOT_V2_PROTOCOL.json), [inference script](pilot_v2.py), [scoring script](analyze_v2.py), [adaptive analysis](PILOT_V2_ANALYSIS.json), and [raw predictions](PILOT_V2_PREDICTIONS_qwen15.jsonl).

The temporary downloaded dataset archive is not copied into this workspace. The script-generated hashes and records make the local run auditable while respecting the dataset source.
