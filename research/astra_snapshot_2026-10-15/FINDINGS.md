# Findings, limitations, and next decision

All numerical findings below are **ARCHIVAL-REPORTED**, not independent replication in this branch. The central subject remains generalization of actor–patient relations. Different tasks and models are not interchangeable evidence of one mechanism.

## 1. Controlled graph effects exist, but the stronger explanation failed

The prospective rewiring comparison reported a −24.45 percentage-point focal penalty under participant-only supervision and +22.59-point attenuation with ordered roles. However, the high-weight role objective also improved general transfer. The subsequent ability-matched development comparison reported **−6.10 points attenuation**, interval **[−14.70, +2.49]**, with only two of eight families positive. Thus an ability-independent reduction in structural dependence was not established. The strict factorial success gate and the original role-swap consistency repair also failed.

Sources: [checkpoint](archive/RESEARCH_CHECKPOINT_2026-10-14.md), [ability match](archive/role_transfer_phase01/ability_match_v1/RESULTS.md), [Phase 2](archive/role_transfer_phase01/phase2_development_v1/RESULTS.md).

## 2. Auxiliary supervision partly protects against fine-tuning regression

On the constructed Qwen2.5-0.5B cleft evaluation: untouched **86.49%**, ordinary fine-tuning **51.51%**, ordered-role auxiliary **75.53%**, participant-only auxiliary **83.82%**. The auxiliary benefits do not exceed the untouched baseline on this test and cannot be uniquely attributed to ordered role labels. Familiar-form performance differs between objectives. This motivates studying adaptation, output specialization, and preservation of capabilities with appropriate controls.

Sources: [matched confirmation](archive/llm_role_repair_confirmation/RESULTS.md), [participant control](archive/llm_role_unordered_control/RESULTS.md).

## 3. Synthetic routing gains did not establish natural-language repair

Slot-first training improved the new-lexical-bank synthetic screen. Deterministic routing from generated fields was a strong baseline. Subsequent natural-sentence tests did not establish the corresponding repair; candidate position, answer formatting, and unverified semantic-role mappings limited interpretation. EWT subject/object labels must not be silently treated as verified semantic agent/patient labels. Human review remains necessary.

Sources: [synthetic confirmation](archive/research/role_slot_routing_fresh_confirmation_2026-10-13/RESULTS.md), [natural evaluation](archive/research/role_slot_routing_qasrl_gold_2026-10-13/RESULTS.md), [checkpoint](archive/RESEARCH_CHECKPOINT_2026-10-14.md).

## 4. Prompt and construction sensitivity has a capability boundary

On the reused 12-event constructed panel, object-cleft joint actor/patient correctness under lexical versus explicit questions was Qwen7B **3/12 versus 8/12**, Phi3 **9/12 versus 1/12**, and Llama3B **2/12 versus 0/12**. A no-prefill Llama control yielded **4/12 versus 0/12**. Hosted Gemma31B answered all **1,728** evaluated prompts correctly. Those are repeated variants, not 1,728 independent events. The strong panel has six name-pair clusters; model families, sizes, quantization and inference interfaces differ. This is not a pure scaling experiment or evidence that all LLMs fail roles. Fixed positional heuristics can explain some failed cells.

Source: [cross-family report](archive/research/llama_gemma_cross_family_2026-10-15/RESULTS.md) and its analysis/protocol files.

## 5. Mechanism remains unresolved

Selected activation patches changed answers, but position and question effects were not isolated into an abstract semantic-role mechanism. Better answer interfaces reduced earlier extraction advantages. Neither question interference nor unstable internal representations is established. Generated role fields, direct answers, and format compliance must be scored separately.

Sources: [same-sentence patch study](archive/llm_role_internal_v2/RESULTS.md), historical frontier diagnostic reports under `archive/research/`.

## Next bounded investigation — not a discovery yet

Compare matched untouched, ordinary, participant-only and ordered-role Qwen checkpoints to ask whether adaptation changes extraction versus direct-answer behavior. A read-only loader and two-event technical preflight exist. Full evaluation is pending protocol freeze. Competing explanations include conventional overfitting, auxiliary regularization, output specialization, role-binding errors and answer selection. Behavioral differences alone cannot identify an internal mechanism.

The scientific ambition remains a strong LLM/natural-language contribution. No conference acceptance, universal learning principle, novel repair, or independently verified natural-language generalization is established by this snapshot.

## Completed development update: interface-dependent adapter rankings

The formerly pending diagnostic is complete: 21 model evaluations, 2,016 sentence-model comparisons. On clefts, ordered-role versus ordinary actor accuracy changed from 77.1% vs 62.5% under the archived interface to 63.5% vs 69.8% under the fixed explicit direct question. Joint extraction was 90.6% untouched and 84.9–85.9% for adapted arms. Direct patient QA was poor, and extraction also failed frequently on active sentences. This supports interface-dependent behavioral rankings, not a preserved internal mechanism or a general repair. [Full results and uncertainty](completed_adapter_diagnostic/RESULTS.md). All outcomes are synthetic development evidence; the large prompt count does not provide independent natural-language replication.
