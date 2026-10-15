# User-requested interim read

At 2026-10-08 13:21:14 UTC, 15/72 models were complete. The user requested a
provisional look. Only the first **fully completed graph**, `overlap0`, was
inspected; it has all three paired starts and all three training arms (nine
models). The next graph was incomplete and was not inspected. The frozen
[protocol](PROTOCOL.json), manifest, stopping rule, remaining training plan,
and analysis code were not changed.

For `overlap0`, nine unseen pairs have low shared-partner overlap (0–1) and
four have high overlap (3–4). Mean binary actor/patient accuracy across three
starts was:

| Training arm | Low overlap | High overlap | High − low |
|---|---:|---:|---:|
| Ordinary baseline | 95.47% | 72.74% | −22.73 points |
| Unordered identity auxiliary | 95.53% | 78.42% | −17.11 points |
| Ordered role auxiliary | 99.66% | 97.59% | −2.07 points |

The ordered-minus-unordered attenuation of the high-overlap penalty was
+15.04 points. This first graph matches both predicted directions, but one
graph cannot establish either claim. The planned eight-graph tests and
guardrails remain pending. No interim significance test is used.

## Second user-requested status read

At 2026-10-08 13:38:27 UTC, 24/72 models were complete. The first **two**
fully completed graphs (18 models) were inspected in frozen plan order. The
third graph was incomplete and was not inspected. The high-minus-low overlap
role-accuracy penalties were:

| Graph | Ordinary baseline | Unordered identities | Ordered roles | Ordered − unordered penalty |
|---|---:|---:|---:|---:|
| overlap0 | −22.73 points | −17.11 points | −2.07 points | +15.04 points |
| overlap1 | −24.72 points | −21.38 points | +0.09 points | +21.46 points |

The two-graph provisional means are **−23.73 points** for the baseline penalty
and **+18.25 points** for ordered-minus-unordered attenuation. Both graphs
match the predicted directions, but six graphs and all frozen tests remain.
No training, selection, stopping, or analysis choice changed after this read.

## Third user-requested status read

At 2026-10-08 14:06:23 UTC, 36/72 models were complete. The first four fully
completed graphs (36 models) were inspected. Their ordinary-baseline
high-minus-low overlap penalties were **−22.73, −24.72, −30.11, and −9.73
points**. The corresponding ordered-minus-unordered penalty attenuation was
**+15.04, +21.46, +23.05, and +16.95 points**. All four graphs currently match
both predicted directions. Provisional four-graph means are **−21.82** and
**+19.13 points**, respectively. These are not final tests; four graphs remain.
All completed checkpoint hashes verified. No change was made to the protocol,
remaining training, or analysis.

## Fourth user-requested status read

At 2026-10-08 14:46:45 UTC, 57/72 models were complete. Six graphs had all
three starts and three arms complete (54 models); the seventh was incomplete
and not inspected. All six ordinary-baseline high-minus-low penalties were
negative: **−22.73, −24.72, −30.11, −9.73, −12.44, and −29.11 points**.
All six ordered-minus-unordered attenuations were positive: **+15.04, +21.46,
+23.05, +16.95, +11.41, and +21.04 points**. The provisional six-graph means
were **−21.48** and **+18.16 points**. All 57 saved checkpoint hashes verified.
No analysis, selection, stopping, or training choice was changed; two graphs
and the frozen final tests remain.
