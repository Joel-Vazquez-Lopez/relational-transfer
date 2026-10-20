# Research-to-paper audit — 14 October 2026

## Recommendation

Develop the LLM paper around **the interaction between event-role extraction and role-conditioned answering under linguistic variation**. First test whether the failures occur in untouched pretrained models or are introduced by adaptation. Then test a targeted intervention. Keep the controlled graph findings as a bounded companion contribution; combining them with the LLM study requires direct evidence that the same phenomenon is involved.

Candidate question: **Does conditioning on a requested role improve answer selection while destabilizing event-role extraction, and can separating these computations improve transfer to unfamiliar language?** This is a new hypothesis suggested by our existing outputs, not an established finding or a verified novelty claim.

This audit reviewed the accumulated research checkpoint, prior paper audits, major branch reports, selected implementation/ancestry code, and nearby primary literature. It recomputed output-level diagnostics from 12 cached evaluation files. It did not independently rerun every historical model, inspect sealed outcomes, or launch training. Historical numbers below come from the corresponding reports; new diagnostics are explicitly post hoc.

## What the research currently supports

| Evidence | What it supports | Limit on the paper claim |
|---|---|---|
| Fresh controlled rewiring: unordered focal penalty −24.45 pp; ordered attenuation +22.59 pp | High-weight ordered-role supervision improves transfer under a controlled rewiring package | The package changes several structural properties; general transfer ability also improves |
| Strict factorial and ability-matched follow-up | Serious tests of the stronger explanation | Strict common-partner interval includes zero. Ability-matched attenuation −6.10 pp, interval [−14.70,+2.49], does not support the proposed ability-independent effect |
| Original role-swap consistency: +0.33 pp attenuation, interval [−5.31,+5.97] | A useful comparison of two distinct learning signals | No demonstrated benefit of the original consistency objective |
| Synthetic Qwen patient-cleft: ordinary 51.51%, ordered 75.53%, participant-only 83.82%, untouched 86.49% | Auxiliary training preserves some capabilities lost during ordinary adaptation | Ordered roles are not uniquely effective; trained arms do not beat the untouched model here |
| Subject/role-decoupling across two starts and two lexical banks | More passive exposure substantially helps passive transfer while active performance falls | Construction trade-off in a small synthetic setting, not general natural-language repair |
| Token-matched typed cues with candidates | Large answer-use improvements with limited extraction improvement | Candidate support and question interpretation simplify the problem |
| Candidate-free two-start pilot | Typed cues improve final answers, unevenly across starts | Frozen success screen failed; one start loses role-field accuracy; all 100 dev sentences actor-first |
| Natural-language structured output and verifier branches | Correct printed roles can coexist with incorrect final answers | Existing labels, source-adapter ancestry, candidate support, format effects and source restrictions limit generality |
| Early goal-decomposition branches | Some controlled diagnostic tools and hypotheses | Early pad-mask invalidity; later oracle/grammar baselines and overlap prevent a broad novel-planning claim |

The strong version of the original graph hypothesis has not survived its most informative controls. That is evidence for changing the next question, not erasing the earlier absolute performance gains. The graph paper remains a possible bounded analysis paper, but it is a weaker match to the user's LLM-and-natural-language objective.

## Important corrections to the research plan

1. `NEXT_DECISIVE_LLM_STUDY.md` proposed an event-bound transformation loss. We already tried a close version in `research/event_role_transport_pilot_2026-10-12`: joint success fell 31→28 and 38→35 out of 144. Any new version needs an explicit mathematical and diagnostic distinction; simply repeating it at greater scale is unjustified.
2. Many recent LLM branches start from natural-role adapters which themselves followed synthetic participant training. They are branches from these sources, not independent experiments from pristine pretrained weights. A failure in these checkpoints cannot establish a general failure of untouched LLMs.
3. The latest candidate-free trial matches updates and target-token counts, but its NONE versus typed input cues are not exactly input-token matched. Do not inherit the stronger matching claim from the earlier token-control experiment.
4. Printed fields are generated outputs. Calling them internal representations or proof of an internal mechanism would overstate the evidence.
5. Public QA-SRL answers have annotation provenance. Our automatic actor/patient remapping, filtered sample, and generated rewrites require their own validation. Original QA-SRL question-answer evaluation can proceed without claiming that every original label needs new annotation.
6. Earlier paper rankings from 13 October predate the strict and ability-matched results. Use this audit when choosing the next direction.

## New read-only diagnostic from existing outputs

See `CACHED_OUTPUT_AUDIT.json` and `COMMON_PAIR_AUDIT.json`. All units below are development data. The question comparison changes requested role and wording together; the typed arm also changes its cue. It does not isolate a causal effect.

| Candidate-free start | Neutral: correct requested field but wrong final answer | Typed: correct requested field but wrong final answer |
|---|---:|---:|
| fresh1 | 39/118 | 13/119 |
| fresh3 | 44/118 | 16/105 |

The denominator is the number of responses with a correct requested-role field. These are descriptive conditional rates; conditioning on correctness selects different items across arms.

Now restrict to identical events for which both question responses parse in both arms and both inferred role tags are correct:

| Start | Common eligible events | Neutral: role tuple changes between questions | Typed: role tuple changes between questions |
|---|---:|---:|---:|
| fresh1 | 88 | 25 | 40 |
| fresh3 | 83 | 19 | 43 |

A sentence's event-role tuple should be stable when the target event stays fixed and only the queried role changes. These counts suggest that better answer-field agreement can coexist with less stable extraction. However, semantic remapping remains provisional, query wording differs, and this was discovered after examining results. It is a reason to design a test, not a confirmed interference mechanism.

## Novelty boundary: what nearby work already covers

Primary sources inspected on 14 October 2026; this is a targeted comparison, not an exhaustive proof of absence.

- [Press et al., Measuring and Narrowing the Compositionality Gap](https://arxiv.org/abs/2210.03350): correctly answered subproblems can coexist with a wrong composed answer. A generic extraction-versus-answer gap is insufficient novelty.
- [Focus on Your Question! (ACL 2024)](https://aclanthology.org/2024.acl-long.499/): studies correct rationales followed by wrong answers and mitigation. Question information and answer generation already have direct precedents.
- [Legible Failures (September 2026 preprint)](https://arxiv.org/abs/2609.11216): decodable in-context bindings despite wrong answers, with detection and steering. It is particularly close to a generic binding-is-present-but-unused claim. Our current printed-field evidence is weaker than hidden-state interventions.
- [Mixing Mechanisms](https://arxiv.org/abs/2510.06182): investigates retrieval of bound entities and multiple mechanisms. Merely locating a binding or retrieval mechanism is insufficient.
- [Emergence and Localisation of Semantic Role Circuits](https://aclanthology.org/2026.findings-acl.1964/): directly studies semantic-role circuits. Do not claim that investigating role information in LLMs is novel.
- [Emergent Semantic Role Understanding (May 2026 preprint)](https://arxiv.org/abs/2605.09187): uses QA-SRL and probes to study role understanding. Plain QA-SRL probing is already established.
- [Flip-Flop Consistency (ACL 2026)](https://aclanthology.org/2026.acl-long.71/): trains robustness across prompt perturbations. A generic consistency loss needs this kind of comparator; agreement alone cannot establish correct relational transfer.
- [QA-SRL Bank 2.0](https://aclanthology.org/P18-1191/) and [MRC-based SRL](https://aclanthology.org/2022.coling-1.191/): question-conditioned argument extraction is established. Explicit roles and pipelines are not inventions here.
- [Joint semantic-role and NLI learning](https://aclanthology.org/2020.repl4nlp-1.11/): role supervision for generalization has precedent.

Potential distinctive contribution: **a demonstrated, reproducible trade-off between question-conditioned answer use and event-role stability, its dependence on syntax and adaptation, and a controlled intervention that improves joint correctness on independently evaluated natural language.** Each component remains to be established. Separating extraction from answering is an established pipeline idea; the contribution would be the new empirical account and evidence for when separation helps or hurts.

## Sequential programme and decision gates

### 1. Establish whether the phenomenon survives a clean baseline

Use an openly documented pretrained instruct checkpoint without our adapters, the current adapted source, and the current typed/neutral branches. Begin on development data, using ordinary natural questions and a small prompt-development budget shared across conditions. Include the already available small Qwen and a larger checkpoint; later test a genuinely different family. Record ancestry, precision, decoding, formatting, and prompt selection.

Target a specified predicate so changing the requested role cannot silently change the event. Compare direct answers, role-first answers, typed cues, and extraction supplied with sentence plus predicate only. Use the same sentences and no candidate spans in the primary task. Supplied candidates remain a diagnostic condition.

Gate: if the problem is confined to adapted checkpoints, pivot explicitly to adaptation-induced relational regression. If clean models are near ceiling on the balanced task, do not manufacture difficulty by selecting their failures; predefine linguistically motivated harder strata. If no meaningful phenomenon remains, stop this repair branch.

### 2. Validate the natural-language measurement

Retain original QA-SRL questions and accepted spans as an initial source-grounded task. For the actor/patient subset, review predicate identity, semantic applicability, alternate valid spans, and any ambiguity. Keep grammatical subject/object separate from semantic roles. Independently verify every newly transformed test event. Existing inspected sets remain development data.

Build event families covering natural active and passive cases, varied participant order and reversible events. Score role-preserving rewrites separately from genuine role swaps. Use document-disjoint splits; audit entity-pair, verb and template overlap separately. Start with a modest reviewed development sample; determine confirmation size using its paired disagreement and desired precision. Adding a second dataset is useful only when its annotation scheme genuinely tests the same claim.

Metrics: original-QA exact match and span F1; actor and patient accuracy; both roles correct; all required variants correct; role-tuple stability; final-answer agreement with the requested field; invalid output frequency. Report correctness with stability, never stability alone. Cluster uncertainty by source document/event family, retain seed-level variation, and avoid treating every prompt variant as independent.

Include sentence-omission, first/last participant, subject-as-actor, candidate-order and alternative-answer-span controls. Human review must be prediction-blind. AI-assisted labels remain provisional until reviewed.

### 3. Isolate the suspected trade-off before inventing a new loss

Use a factorial comparison: requested-role information visible during extraction versus withheld until answering; typed-role cue present versus absent at answer selection. All extraction conditions receive the target predicate. Keep final task information equivalent. A two-stage system may receive more compute or reset context, so compare with a two-stage sham and matched generation budget.

Evaluate predicted extraction with deterministic routing as a strong baseline. Compare gold versus predicted role selectors and gold versus predicted event fields only as diagnostic interventions. Gold fields supply answer information and cannot count as ordinary model performance. Corrupt or swap fields under fixed sentences to distinguish following the representation from correcting it; score both behaviors separately.

Only if these diagnostics identify a repeatable benefit, train direct-QA, structured-role, typed-role, and extraction-separated arms from the same source. Match data, labels, updates, token exposure and loss accounting; control additional passes and output order. Include ordinary transformation augmentation before crediting any additional consistency objective. Reuse or implement a conventional SRL-plus-router comparator where feasible.

Gate: proceed when the intervention improves joint correct answers on development constructions and does not simply sacrifice another construction, exploit supplied candidates, or repeat the old transport loss. A null result is retained. Do not tune until a chosen p-value appears.

### 4. Independent confirmation and generalization

Freeze one method and its strongest controls, training seeds, primary contrast, exclusions and scoring. Evaluate new documents/event families once after development. Replicate across a second pretrained family and a meaningful scale difference, subject to hardware. Model counts alone are not evidence of generality; report exact scope and heterogeneity.

A stronger result would show improvement on held-out constructions and question forms with comparable ordinary performance. If hidden mechanisms become part of the claim, add targeted activation interventions with position, token identity, event and random-direction controls. Visible-field intervention alone supports an interface claim.

Do not launch this stage before the earlier gates. Do not fold the graph experiments into a unified mechanism without a matched bridge.

## What a strong paper could contain

1. A validated natural-language diagnostic exposing a specific failure hidden by ordinary accuracy.
2. A controlled account of whether the failure originates in adaptation, extraction, query interpretation, or answer use.
3. A simple targeted intervention, compared with credible pipelines, augmentation and consistency controls.
4. Independent natural-language and cross-model evidence, with failures and trade-offs retained.

An analysis paper can be valuable even if a new objective fails. The current portfolio does not yet establish this package, and no audit can guarantee ACL acceptance. Its strongest opportunity is to turn the repeated extraction/use dissociation into a precise, independently tested linguistic finding. More pilot variants without stronger measurement will not close that gap.

## Reproduction of this audit

Run `python3 research/paper_audit_2026-10-14/audit_cached_outputs.py` and `python3 research/paper_audit_2026-10-14/audit_common_pairs.py` from the project root. Both consume existing development predictions only. See the artifact index for historical report paths and hashes. Historical frozen protocols and results are unchanged.
