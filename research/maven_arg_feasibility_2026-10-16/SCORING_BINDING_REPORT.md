# Different-relation scoring calibration

17 October 2026. This is an **adaptive follow-up**, designed after the defeat-relation patch studies. It is not a preregistered independent confirmation or a novelty claim.

## Locked design

Three new MAVEN-Arg `train.jsonl` documents have two selected `Getting` events described as scoring points or goals. Their source document IDs do not overlap the earlier behavioral, stress, or fresh defeat panels. A one-reviewer source screen fixed the [selection](SCORING_SELECTION.json), [items](SCORING_ITEMS.json), and [protocol](SCORING_BINDING_PROTOCOL.json) before local model inference. The protocol fixes Qwen2.5-7B-Instruct MLX 4-bit, layer 16, 12 generated tokens, the same-letter A/B recipient versus B/A donor, and two controls. The entire 30-row run is in [raw results](SCORING_BINDING_RAW.jsonl), with a [machine summary](SCORING_BINDING_SUMMARY.json).

The questions were: “who scored the points or goal?” for Agent, and “which opposing team did that player score against?” for Patient. The Patient answer requires match context; it is less locally explicit than the Agent answer. The Ajax passage also has an unmarked third score. These limitations were recorded before running the model. On audit after the run, I also found a prompt defect: the reused system instruction says to copy a **person's name**, while the Patient question requests a **team**. This conflict makes the Patient results especially hard to interpret and must be fixed in a new, separately labeled study.

## Complete-answer results

| Passage | Role | First gold | Second gold | Unpatched A/B | Direct B/A | Patched A/B | Adjacent EVENT control | Identity control |
|---|---|---|---|---|---|---|---|---|
| 1995 Rugby World Cup Final | Agent | Andrew Mehrtens | Joel Stransky | Andrew Mehrtens | Joel Stransky | Joel Stransky | Andrew Mehrtens | Andrew Mehrtens |
| 1995 Rugby World Cup Final | Patient | South Africa | New Zealand | Springboks | Springboks | Springboks | Springboks | Springboks |
| 2012 Football League Cup Final | Agent | Joe Mason | Martin Škrtel | Joe Mason | Martin Škrtel | Martin Škrtel | Joe Mason | Joe Mason |
| 2012 Football League Cup Final | Patient | Liverpool | Cardiff | Cardiff | Liverpool | Cardiff | Cardiff | Cardiff |
| Ajax 5–1 Liverpool (1966) | Agent | Klaas Nuninga | Chris Lawler | Klaas Nuninga | Chris Lawler | Chris Lawler | Klaas Nuninga | Klaas Nuninga |
| Ajax 5–1 Liverpool (1966) | Patient | Liverpool | Ajax | Ajax | Chris Lawler | Liverpool | Ajax | Ajax |

The **Agent result is clean on this small panel**: all three unpatched answers match the first scorer, all three direct reversed answers match the second scorer, all three label-patched answers match the second scorer, and both control conditions preserve the first scorer in all three documents. This extends the observable scorer-selection effect beyond the defeated relation in one model.

The **Patient result does not support a two-role transfer**. All three unpatched Patient answers miss the locked local-mention exact score; all three direct reversed answers miss second-event gold. The rugby `Springboks` answer is semantically South Africa, the first-event opponent, but that alias was absent from the source entity's listed mentions. Even granting it as correct, the direct reversed rugby answer remains wrong. Consequently none of the three documents meets the prespecified strict success criterion. The summary's 4/6 patched-second exact count includes three Agent cells and one Cardiff Patient cell, but that Cardiff answer is *identical to the unpatched answer* and gives no evidence of a patch-induced Patient switch. The opposing-team prompt did not establish reliable baseline competence; in two football passages the model returned the scorer's team for the first event. We should not treat these failures as causal evidence for or against an internal Patient binding.

## Interpretation and next gate

The robust, limited observation is a layer-16 event-label intervention that transfers **which scorer** is retrieved across three new sports documents. A generic first/second event cue, passage context, and answer-bearing information remain rival explanations. Earlier [cross-document results](FRESH_BINDING_REPORT.md#adaptive-cross-document-diagnostic) lean toward a transferable order cue rather than a literal event record. The present study does not resolve that distinction or prove novelty; prior activation-patching work already studies binding and order.

Before another two-role causal claim, build a relation family whose two participants are explicit in each local clause, use a system instruction compatible with both answer types, freeze unambiguous role questions, and require correct first and reversed-second baselines for both roles. A separate corpus or independent annotation adjudication would strengthen the next confirmation. Do not silently drop these three Patient failures or tune wording on this panel and call it held out.
