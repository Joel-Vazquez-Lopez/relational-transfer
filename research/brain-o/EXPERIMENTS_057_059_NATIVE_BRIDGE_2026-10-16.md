# Brain-O E057–059 — Context-dependent native causal influence is not sufficient for correct relational answers
**16 October 2026 | Status: EXECUTED and exact frozen-checkpoint replay verified | Synthetic open development only**

## Motivating falsifier
E045–E056 established near-perfect auxiliary compound event-index classification (from a contextual query-tag readout) without correct final actor/patient answers. The auxiliary event index **never directly feeds the answer head**; E051's engineered grammar/name copier produced near-perfect answers even with randomized/zeroed source representations. Therefore a no-parser, **model-native** residual intervention and parameter-matched lexical/capacity controls were required before any stronger claim.

## E057: fixed-backbone trainable residual bridge
Freeze *all* weights of two successful 3-layer ordinary causal Transformer event selectors, **including native final answer layer and all attention layers**, from E045–050 (seeds 1040/1041 at 750 updates; generator-supervised event auxiliary). Add an identically initialized **12,673-parameter MLP** to the native answer-token residual *after layer 2, before original layer 3*. Train only bridge parameters for 260 updates using 900 synthetic training worlds and score **200 new generated worlds × 6 dependent questions** per model.

Arms: (a) **contextual query-key states** after layer 2 (verb and context-tag tokens), (b) equal-parameter **self-role-state** sham, (c) equal-parameter **raw lexical token embeddings** without source context. No candidate/voice/name parser, gold event label, explicit participant pointer, or event-classifier output at inference.

| Seed | Frozen native | Contextual route | Self-token sham | Raw lexical control | Whole correct worlds |
|---|---:|---:|---:|---:|---:|
| 1040 | **387/1200** | **414/1200 (34.50%)** | 393/1200 | 395/1200 | 0/200 in all |
| 1041 | **400/1200** | **491/1200 (40.92%)** | 399/1200 | 386/1200 | 0/200 in all |

Contextual-vs-self absolute gain +1.75 pp for 1040, +7.67 pp for 1041. Exploratory bootstrap over 200 **world clusters**: **[-0.42,+3.92]** and **[+5.50,+9.83]** pp, respectively; conditional on these fixed seeds, not a confidence interval over models or language families. The first seed does not show a convincing material rescue. No complete six-question worlds are solved.

## E058: identical question + word-multiset-preserving **source-event reassignment**
Generate **300 new worlds × 2 roles per seed**. Holding the entire question token string and full source-token multiset identical, swap the two relevant source records' compound (verb,context) keys, preserving each record's participant names, voice and position; original/donor correct participant is necessarily different. Transplant **only** donor layer-2 query-key contextual states into base world before frozen layer 3, for each already trained bridge arm.

| Fixed-query counterfactual outcome (600 pairs/model) | Seed 1040 | Seed 1041 |
|---|---:|---:|
| Contextual bridge answer **changes** after source patch | **190** | **254** |
| Self-token sham answer changes | 3 | 2 |
| Raw lexical control answer changes | 2 | 0 |
| Contextual patched answer equals **new gold participant** | **180 (30%)** | **219 (36.5%)** |
| Native auxiliary event-index from *same patched state* equals new donor record | **408 (68%)** | **598 (99.7%)** |
| Full donor native event head correct | 598 | 600 |

Seed 1041 provides especially clear evidence of **source-conditional task-relevant event readout transfer without good answer utility**. But this remains a whole-token state patch, not isolation of a semantic binding feature. Other source-sensitive lexical/copying processes could generate this influence. The model still chooses the wrong donor participant most of the time.

**Critical metric caveat:** patched output agreeing with the full donor output at ~95%+ is misleading because even the full donor's *answer* is often wrong. Source-sensitive answer changes and task-aligned correctness must be reported separately.

## E059: world-cluster paired audit
On same 200-world held-out panel, bootstrap paired accuracy differences resampling entire 6-question worlds; on E058, bootstrap difference in patch-change rate resampling 300 two-role worlds. Event-minus-self patch switching gap: **+31.17 pp** [27.17,35.17] (seed1040), **+42.00 pp** [38.00,46.17] (seed1041). High sensitivity does not imply a correct answer. These are *exploratory conditional generated-world* intervals, not general population claims.

## Scientific interpretation / decision
**Supported:** a small no-parser internal bridge can convey **source-context-sensitive information through the existing final Transformer attention layer into its native frozen answer head**. It outperforms both equal-parameter self-role and no-context lexical controls in changing answers when the source-event assignment changes. A very accurate event-classifier output can coexist with consistently wrong final-role retrieval, even when the bridge makes the native answer function source-sensitive.

**NOT supported:** a reusable abstract semantic binding variable; a general computational principle proved across model families; realistic natural language or pretrained LLM transfer; an accurate answer rescue; a uniqueness claim over generic copying or better-trained attention backbones. The native event classifier is **separate**, not literally routed into native answer; this bridge uses hidden states. Effects are heterogeneous by seed and underpowered for architecture-wide conclusions.

**Next threshold:** successful *whole-world* answering and ability/compute-matched generic neural-copying rival with native-path necessity cuts and heldout 3–5-event structures. In parallel, existing-human-question→span QA-SRL Gold and MuSiQue bridge belongs to Nova/Comput/Mira, not Brain-O. No paid inference/sealed evaluation/other branch modification.

## Reproducibility
Exact self-contained **conversation artifact** `BRAIN_O_E057_059_NATIVE_BRIDGE_2026-10-16.zip`, SHA256 `9912ea0c0372ea6b4df0d67d4f7a1b77628576d225bfdef82c44baee0b97ab56`, **32 ZIP members, 31/31 payload SHA checks, CRC pass**. Includes source from E040/E041, both original frozen selector weights, 6 trained native bridge weights, all six evaluation metrics, E058 paired per-example CSVs, E059 grouped bootstrap JSON, scientific report, manifest and `verify_saved.py`. **Verified from clean extracted temporary directory: all 6 frozen bridge evaluations replay exact counts, no retraining.** Binary ZIP is conversation attached; not on GitHub and not yet independently received by a second researcher. Use `torch.load(...,weights_only=True)` after source review.

**Signal: POSITIVE source-sensitive causal influence + NEGATIVE accuracy-rescue result. No EUREKA.**
