# Ability-matched ordered-role development result

Completed 14 October 2026. This is **development evidence on eight previously examined graph families**, not an independent confirmation. The [protocol](PROTOCOL.json), [calibration](CALIBRATION.json), [full ability gate](ABILITY_GATE.json), [focal results](FOCAL_DEVELOPMENT.json), checkpoints, and [run ledger](RUN_LEDGER.jsonl) preserve the design and outputs.

## Frozen decision sequence and integrity

Four nonzero ordered-role loss weights (0.001, 0.005, 0.02, 0.08) were trained on the first two families, both sham/high graph conditions, and two paired seeds: 32 calibration models. The selection rule used only eight prespecified *other unseen pairs* and four familiar pairs per graph condition. It chose weight **0.001**, whose calibration nonfocal binary-role accuracy was **88.55%**, versus **87.55%** for the existing participant-only arm: a **+1.00 percentage-point** signed gap, within the frozen two-point tolerance. Minimum familiar four-choice accuracy was **99.80%**. The other weights missed the ability tolerance.

The selected weight was then run on all eight families. There were 24 additional models, with eight calibration models reused. All 32 calibration jobs and 24 new full-stage jobs finished with exit code zero. The resumed full-stage runner retried two interrupted, unfinished configurations without changing their graph, seed, weight, training budget, or code. Ledger entries for the interrupted starts remain visible. All 32 selected-weight metric files match their 6,000-step model SHA-256; frozen source, runner, selector, evaluator, and protocol hashes were verified.

The full **aggregate ability gate passed**: ordered-role **85.69%** versus participant-only **85.01%** nonfocal unseen binary-role accuracy, a **+0.68-point** gap. Minimum familiar four-choice accuracy was **99.61%**. The evaluator opened the focal pair only after this gate passed. Importantly, the family-level ability gaps were **+2.34, −0.34, −4.22, +7.57, −1.76, +2.27, −3.00, +2.59 points**. Thus the global match is not a close match in every family or condition.

## Prespecified focal outcome

The focal outcome is the difference between the high-minus-sham binary-role accuracy changes for ordered-role and participant-only training. Positive values would mean ordered-role training attenuates the focal graph penalty. At the ability-matched operating point, the mean was **−6.10 points**, with a descriptive family-level 95% t interval **[−14.70, +2.49]** and exact two-sided family sign-flip **p = 0.140625**. Two of eight families were positive. The eight attenuation estimates were **+4.20, −17.48, −9.96, −1.27, −18.55, −1.95, −13.28, +9.47 points**.

| Final 6,000-step focal binary-role accuracy | Sham | High rewire | High minus sham |
|---|---:|---:|---:|
| Ordered-role, weight 0.001 | 99.88% | 66.94% | −32.93 points |
| Participant-only, original weight | 98.49% | 71.66% | −26.83 points |

The high-rewire focal result for weak ordered-role supervision is lower than participant-only despite their similar **aggregate** nonfocal ability. Ordered-role focal event accuracy was **99.61% sham / 98.65% high**; its focal four-choice accuracy was **99.50% sham / 66.30% high**. These secondary scores show that the failure concerns role assignment rather than general event recognition, but they do not isolate a unique mechanism.

## Interpretation and decision

This fixed ability-matched test **does not support** the claim that ordered-role supervision reduces dependence on the tested relational-evidence structure beyond its broader transfer gain. It does not prove the absence of such an effect at every possible operating point: the interval includes a small positive attenuation, family effects vary, and family-specific ability differences remain. The earlier high-weight Phase 2 result of **+22.57 points** remains a real measured absolute-scale interaction; it must not be described as an ability-controlled interaction. Neither result identifies a unique graph property, because the rewire changes a package of local relations.

These eight graph families were previously examined, and the calibration used two of them. The exact p value and interval are descriptive, not independent confirmatory evidence. The original role-swap consistency objective also showed no distinct gain in Phase 2. A fresh-graph confirmation of the *strong* ability-independent claim is not justified on this development outcome alone.

The next research decision should keep the controlled effect as a bounded finding, investigate its pair-specific and ability-dependent character if useful, and prioritize a **direct pretrained-LLM study with independently verified natural-language role labels** for the proposed paper. Existing Qwen synthetic and QA-SRL/EWT runs motivate that study but do not establish the graph interaction or authentic-language repair. Any new objective or evaluation protocol must be frozen before new confirmation data are examined.
