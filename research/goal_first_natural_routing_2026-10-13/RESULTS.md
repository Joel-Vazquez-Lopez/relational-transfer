# Goal order and learned question routing on natural QA-SRL items

## What was tested

Two Qwen2.5-0.5B continuations per training start used the same 160 training items, 640 prompts, 160 updates, and **11,676 supervised target tokens**. Each had to output the requested role, actor, patient, and final answer. Only the order changed: `Need; Actor; Patient; Answer` versus `Actor; Patient; Need; Answer`. The [protocol](PROTOCOL.json) fixed a new 40-item, 40-sentence QA-SRL Bank evaluation before either arm was trained. Actor/patient labels come from a deterministic mapping of human QA questions and answer spans; that mapping is not independently human reviewed. This is exploratory development, not a final natural-language test.

## Matched training result: goal-first order failed

| Start | Output order | Exact answers /160 | All four answers /40 | Both role fields right /160 | Requested-role field right /160 |
|---|---|---:|---:|---:|---:|
| fresh1 | Goal first | 83 | 2 | 134 | 94 |
| fresh1 | Goal after roles | **93** | **6** | 131 | **106** |
| fresh3 | Goal first | 82 | 2 | **135** | 88 |
| fresh3 | Goal after roles | **91** | **6** | 122 | **103** |

The frozen goal-first positive screen **failed in both starts**. Its all-four score was four items lower in each, with 9–10 fewer correct direct answers. This is evidence against **this output-order implementation**, not against goal-conditioned decomposition in general. The two arms have identical target-token counts, and their field accuracy remains substantially higher than their direct-answer accuracy. Among prompts where the generated requested role **and both role fields** were correct, the goal-first final answer was correct on 74/75 and 72/75; the goal-after-roles answer was correct on 77/86 and 77/80. Much of the remaining breakdown is identifying the requested role in the model output and preserving the role fields, rather than merely copying a correct role once all fields are correct.

## Learned query routing: system-level positive result

After the matched-arm result, a separate [routing protocol](ROUTER_PROTOCOL.json) fixed a character-gram logistic classifier trained only on the 320 actor/patient question texts from the 160 training items. It identified the requested role on **80/80** questions in the same 40 fresh items, including **54/54 exact question strings absent from its training questions**. It then selected the corresponding actor or patient from each model's already generated fields. No gold requested-role label was supplied at test time.

| Start | Model whose printed fields were routed | Own exact /160 | Routed exact /160 | Own all-four /40 | Routed all-four /40 |
|---|---|---:|---:|---:|---:|
| fresh1 | Previous role-first | 82 | **138** | 1 | **28** |
| fresh3 | Previous role-first | 80 | **131** | 2 | **26** |
| fresh1 | New goal-first | 83 | **137** | 2 | **27** |
| fresh3 | New goal-first | 82 | **140** | 2 | **28** |

The existing answer-only checkpoints scored 94/160 and 95/160 direct answers, with 3/40 all-four in each start. The prespecified exploratory routing screen passed. However, these QA-SRL question patterns are extremely regular: the simple rule *“`someone` is the last word”* also identifies all 80 roles. The learned classifier does not establish a novel language-understanding ability. The gain is a **modular answering result** showing that role fields can contain useful information that the free final answer fails to use.

## Limits and next step

The question bank was new for the matched training test, but the routing follow-up was designed after seeing the matched result and uses that same bank. The 40 items share one source distribution, are automatically mapped to actor/patient labels, and have participant overlap with training. The two starts share all items and one model family. The routed system adds a separate trained classifier and a deterministic answer selection rule, so it is not a matched-compute improvement over direct generation. This result cannot yet support a conference claim or a general natural-language repair.

The next test should train a model **with the requested role supplied as an input condition** while preserving actor/patient extraction, compare against an equally formatted constant-tag control on a new sentence bank, and use the classifier's predicted tag rather than gold at inference. The independent human-reviewed challenge remains untouched.
