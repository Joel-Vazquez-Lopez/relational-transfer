# Local language-model role pilot (7 October 2026)

## Question and scope

Do two locally available Qwen2.5 instruction models reliably identify actor and patient across active/passive versions of the same event, or can answer position and question wording change their answers? This is an exploratory pilot on constructed English sentences. It does **not** test whether a pretrained model has seen a name pair before, and it does not measure internal representations. The two models are sizes within one family, not independent model families.

The [fixed initial protocol](PROTOCOL.json) specified four new name pairs, three verbs, two directions, two voices, two roles, and two question wordings: 24 semantic events and 192 prompts per model. Greedy output was counted correct only when it gave exactly the intended name. The meaningful units are the 24 events; the 192 prompts are related measurements, not independent examples. The [joint-role follow-up](FOLLOWUP_PROTOCOL.json) and [matched-wording check](WORDING_PROTOCOL.json) were designed **after** seeing the first pilot and are exploratory diagnostics.

## Fixed first pilot

| Model | Correct one-name answers | Valid one-name answers | Chose final name in sentence | Both voices correct across matched role/wording cases |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 105/192 (54.7%) | 192/192 | 183/192 (95.3%) | 9/96 |
| Qwen2.5-0.5B-Instruct | 104/192 (54.2%) | 192/192 | 124/192 (64.6%) | 35/96 |

The gold answer is the final name in exactly 96/192 prompts. Thus the 1.5B result is dominated by a final-name shortcut under these prompts. For this model, generic active actor questions were correct 1/24, while generic passive actor questions were correct 24/24. Generic active patient questions were correct 24/24, while generic passive patient questions were correct 0/24. These four cells show why the aggregate 54.7% is misleading. The 0.5B model showed a weaker, less uniform position bias.

## Adaptive joint-role diagnostic

The same 48 sentences were then asked for both roles at once, with definitions of actor and patient. A second condition also included one passive example.

| Model and condition | Both roles correct | Active | Passive | Strictly formatted |
|---|---:|---:|---:|---:|
| 1.5B, definitions | 35/48 | 22/24 | 13/24 | 48/48 |
| 1.5B, definitions plus example | 40/48 | 20/24 | 20/24 | 48/48 |
| 0.5B, definitions | 1/48 | 1/24 | 0/24 | 6/48 |
| 0.5B, definitions plus example | 0/48 | 0/24 | 0/24 | 1/48 |

The strict 0.5B format score is particularly harsh: many outputs add a terminal period to an otherwise parseable `actor=<name>; patient=<name>` answer. In a **post-hoc punctuation-tolerant check**, 32/48 definition outputs and 22/48 example outputs were parseable; 15/48 and 11/48 respectively had both roles correct. This secondary check retains the original strict numbers and prevents a formatting error from being described as a role error. The 1.5B results were unchanged by punctuation tolerance.

## Adaptive matched wording check

Finally, the actor question was changed from “Who did the action?” to “Who performed the action?” on exactly the same 48 sentences. The model, system instruction, output format, generation settings, and scoring were unchanged.

| Model | Active actor, old → new | Passive actor, old → new |
|---|---:|---:|
| Qwen2.5-1.5B-Instruct | 1/24 → 14/24 | 24/24 → 24/24 |
| Qwen2.5-0.5B-Instruct | 14/24 → 13/24 | 11/24 → 16/24 |

For 1.5B, all 13 changed active answers moved from wrong to correct under that single phrase change. This is direct evidence of wording sensitivity **for this model and these items**. It is an adaptive test selected after earlier prompt results, so it is not a confirmatory significance claim.

## Interpretation and next step

The local models can give unstable role answers. For the 1.5B model, some one-name prompts nearly always elicit the final name, while explicit joint-role instructions recover many correct answers. That supports an evaluation-validity question: one prompt and one aggregate accuracy score do not reliably show what the model can do on these events. The joint follow-up still leaves 8/48 incorrect for 1.5B even with an example, including passive reversals, so prompt repair is incomplete.

The pilot does not show that these models failed to *learn* an unseen pair, that their internal role representations are absent, or that larger current LLMs behave the same way. A stronger next study would use a larger independently selected item set, more model families and sizes, separately frozen prompt variants, and authentic sentences with verified roles. Any internal-state claim requires a separate intervention or representation analysis.

Raw responses and exact scoring are in `qwen15_result.json`, `qwen05_result.json`, `qwen15_followup_result.json`, `qwen05_followup_result.json`, `qwen15_wording_result.json`, and `qwen05_wording_result.json`. The executable methods are `run.py`, `followup.py`, and `wording_check.py`. The separate 72-model synthetic confirmation was still running during this pilot.
