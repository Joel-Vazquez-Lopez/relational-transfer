# More object-cleft exposure does not repair the transfer failure

**12 October 2026.** Prospective Qwen2.5-0.5B exposure control on a new lexical bank. The [protocol](PROTOCOL.json), thresholds and 36-event test were fixed before the new training or inference on this bank. The reserved human-review challenge remained sealed.

## Why this control was necessary

In the preceding [role-coupled pilot](../role_prompt_equivariance_pilot_2026-10-12/RESULTS.md), object-cleft patient answers were poor even on training events: ordinary supervision got **17/48** and **16/48** in the two starts; the coupled objective got **22/48** and **23/48**. The new-lexical test was worse. This made underexposure a plausible explanation before proposing a more complex semantic-role model.

The new schedule keeps the same natural source adapters, 48 directed training events, 96 optimizer updates, nine role-answer targets per update, questions, optimizer and LoRA setup. It shows each event in active and object-cleft form twice, and in passive and subject-cleft form once. The old schedule showed active and passive twice and each cleft once. We trained ordinary SFT and the already defined coupled loss under the new schedule; both old checkpoints were also evaluated on the same **new** name/verb bank.

## Frozen outcome

Values are event–forms where **both held-out role questions** were correct. Broad is out of 144 (36 events × four forms), and each form is out of 36.

| Start | Arm | Broad | Object cleft | Active | Passive |
| --- | --- | ---: | ---: | ---: | ---: |
| Fresh1 | Old SFT | 91 | 12 | 24 | 26 |
| Fresh1 | Old coupled | **94** | 12 | 26 | **27** |
| Fresh1 | Boosted SFT | 76 | 10 | 22 | 17 |
| Fresh1 | Boosted coupled | 91 | **15** | **26** | 21 |
| Fresh3 | Old SFT | 86 | 10 | **18** | 28 |
| Fresh3 | Old coupled | **96** | 12 | **22** | **30** |
| Fresh3 | Boosted SFT | 65 | 10 | 18 | 14 |
| Fresh3 | Boosted coupled | 80 | **13** | 18 | 22 |

The **simple exposure screen fails**: boosted SFT changes object-cleft joint accuracy **12→10/36** and **10→10/36**, far below the required +8 in both starts. Passive joint accuracy drops **26→17** and **28→14**, beyond the permitted loss of four. This is an exposure trade-off, not a repair.

Under the boosted schedule, the coupled loss improves broad held-out joint accuracy **76→91/144** and **65→80/144**. It exceeds the provisional +6 threshold and does not regress on object clefts, but these improvements do **not** establish that the boosted combination is best: old coupled still scores **94/144** and **96/144** overall. Thus the objective helps relative to its data-matched SFT arm, while the training distribution still determines where performance is lost.

For the verb-specific object-cleft patient question, boosted SFT gets **15/36** and **14/36**; boosted coupled gets **25/36** and **19/36**. This is a real within-schedule improvement, but it does not yield reliable joint role interpretation and remains below the easy forms. All 5,760 outputs are valid candidate names.

## Interpretation

Simply repeating the hard construction is insufficient under a fixed budget. The result does not prove that the model lacks a reusable event representation: the old and boosted schedules differ in passive exposure, and the objective may change optimization dynamics as well as role binding. It does rule out the simple claim that *one more object-cleft exposure per event* solves this measured failure.

The next useful distinction is whether the checkpoint can identify the actor and patient **together** from the sentence without a separately worded actor/patient question. If it can, some failure may lie in question-to-role routing; if it cannot, explicit event-role supervision becomes more plausible. Either finding would remain output-level evidence, not proof of an internal mechanism.

The literature already contains structural supervision and argument-structure transfer tests, including [Wilson et al. (2023)](https://aclanthology.org/2023.tacl-1.78/), and prompt-consistency training such as [Hejabi et al. (2026)](https://aclanthology.org/2026.acl-long.71/). Our candidate contribution must be a demonstrably better role-binding intervention under matched data and held-out transformations, not exposure or consistency alone.

## Audit

- [Frozen protocol](PROTOCOL.json), [training and evaluation code](boost.py), four new checkpoints and training traces, eight raw EVAL prediction files, [analysis code](analyze.py), and [analysis](ANALYSIS.json).
- Verified hashes; each new checkpoint has 96 updates and 864 answer targets; all **5,760** test answers are unique valid candidate names. Prior checkpoints and frozen tests were not modified.
