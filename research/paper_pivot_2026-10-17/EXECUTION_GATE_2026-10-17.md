# New paper track: execution gate after source and prior-method audit

17 October 2026. **Decision: no new Qwen run yet.** This is a source-only and literature audit, not a null model result. The user authorized renewed work after the earlier one-study stop; the present design has not passed the prerequisites for an informative held-out intervention.

## What was completed

- Rechecked the earlier [five-document causal protocol](../maven_arg_feasibility_2026-10-16/FRESH_BINDING_REPORT.md). Its layer-16 patch was at the question label *before* Agent/Patient wording, and one Agent donor was used for both roles. Thus the proposed late-bound full-vector experiment would substantially duplicate the prior observation, not supply new evidence by itself.
- Compared the nearest method claims. [Dai et al. 2026](https://aclanthology.org/2026.acl-long.2194/) already identify a low-dimensional entity–relation binding subspace, transfer it across contexts, and use perturbation and patching for causal tests. [RAVEL](https://aclanthology.org/2024.acl-long.470/) tests causally disentangled representations. [Semantic Role Circuits](https://aclanthology.org/2026.findings-acl.1964/) uses role-cross minimal pairs and causal circuit discovery. [Mixing Mechanisms](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf) explicitly constructs counterfactuals separating positional, lexical and reflexive retrieval (§3.2), then separates a pointer from answer copying (§3.4). This is a first method-level boundary check, not proof of novelty or of duplication in every detail.
- Reproduced a deterministic source-only sample from the prior MAVEN-Arg 88-pair training census. The [script](source_gate_audit.py), [source candidate census](SOURCE_CANDIDATE_CENSUS.json), and [12 reviewed passages](SOURCE_GATE_SAMPLE.json) preserve IDs, original marked passages, annotations, judgments and reasons. The local source archive SHA-256 matches the earlier pinned audit. No official TEST data was read.

## Source audit result

After excluding prior experiment documents and requiring trigger separation ≤350 characters and role-mention distance ≤180 characters, the old census has **35 candidate pairs in 24 documents**. A seeded, type-stratified sample of 12 distinct documents yielded **0 clean acceptances, 3 conditional cases and 9 rejections** by one reviewer. This is an intentionally broad feasibility sample, not a random estimate of the entire corpus's acceptance rate or an independent annotation agreement measurement.

A subsequent [source-only follow-up](SOURCE_GATE_SECOND_PASS.md) read the shortest candidate pair in each of the other 12 eligible documents: 0 unambiguous accepts, 2 conditionals and 10 rejections. Across the 24 selected pairs, the tally is **0 clean, 5 conditional and 19 rejected**. The 11 other candidate pairs in those same documents remain unaudited; this does not prove the corpus lacks valid pairs.

| Document | Type | Judgment | Main issue |
| --- | --- | --- | --- |
| Operation Retribution (1941) | Attack | Conditional | Distinct bombings, but many intervening attacks and long excerpt. |
| Attack on Mers-el-Kébir | Attack | Reject | First roles distant from the marked trigger; many other attacks. |
| Yokozuna Memorial Show | Bodily_harm | Reject | `wrestled` and `beat` do not provide comparable directed roles. |
| Wisconsin gubernatorial recall election | Conquering | Reject | First Agent is unresolved `She`. |
| Bloodymania IV | Conquering | Conditional | Two explicit defeats but further unmarked defeats. |
| Siege of Pondicherry (1793) | Control | Reject | Annotated governing Agent conflicts with `governed from Pondicherry`. |
| 1986 Men's Hockey World Cup | Earnings_and_losses | Conditional | Two outcomes recoverable; second event is indirect and needs review. |
| 2015 World Men's Handball Championship | Getting | Reject | Spain is not the Patient of `captured the bronze medal`. |
| Byzantine–Bulgarian war of 894–896 | Getting | Reject | `sought` aid and `secured` assistance have different factual status. |
| Dardanelles Operation (1807) | Military_operation | Reject | An unsuccessful operation and a war differ in scale and role explicitness. |
| Tuareg rebellion (2012) | Military_operation | Reject | `struggled to reconcile` and `clashed` have different event semantics. |
| Air engagements of the Gulf War | Preventing_or_letting | Reject | Annotated Patient `Mirage` conflicts with protected `EF-111`. |

The sample exposes a real measurement problem: four different string fillers and matching dataset event types do not ensure two valid, comparable event–role records. Using these labels as automatic gold would make an activation result uninterpretable. The source audit also shows why a higher-capability model cannot repair bad targets.

I also reopened the already local WikiEvents training source as an independent rescue path. The earlier structural screen has **104 near-100-token pairs across 36 documents** with two shared roles and four distinct coreference fillers. A targeted text read found a plausible pair of separately located hotel/church explosions in `scenario_en_kairos_52` and two different purchase events in `0_VOA_EN_NW_2016.09.21.3519045`; it also found false positives where two triggers describe one attack or coreferent devices. These are *candidate leads*, not adjudicated cases or a new acceptance-rate estimate. WikiEvents uses event-specific roles such as Place/Target and AcquiredEntity/Giver, so it would require a separately frozen question contract rather than silently reusing Agent/Patient wording.

## Research decision

**Keep a narrow paper bet, but do not run the held-out model study now.** The meaningful increment over prior work would be an Agent-trained, rank-restricted causal edit that transfers to Patient on unseen, human-adjudicated natural passages, plus targeted removal/restoration and an order-versus-content witness. The old full-vector late-bound patch is already development evidence and cannot be sold as new confirmation. An unrestricted independent-reader program remains observationally equivalent to a shared-address program; the claim must name its restricted pathway rival.

Before a model run, construct a sufficiently large source-valid set with at least one independent reviewer and predeclared acceptance threshold; prioritize the already local WikiEvents candidates or create a carefully sourced annotation layer if MAVEN-Arg cannot supply it. Freeze the exact prefix boundary, site/rank selection method, matched shams, role-held-out predictions and document-level threshold. The current source audits cannot meet that gate. Do not recast a convenience sample of clean passages as a representative benchmark.

For the 15-credit course, the existing causal effect, the first no-witness result, and this identifiability/data-validity audit already support a rigorous research narrative. A publishable mechanistic claim still requires a source-valid held-out study that beats these prior-method baselines. No success is promised.

## Factuality-binding pivot (source-only follow-up)

The [factuality/argument paper bet](FACTUALITY_BINDING_PAPER_BET.md) uses an exact-ID join of public MAVEN-FACT validation annotations with local MAVEN-Arg validation annotations. This is a newly checked feasibility avenue, not another intervention. The two-event, same-type, four-distinct-role factuality contrast has only two structural validation pairs, so it fails as a confirmatory design. A narrower question about whether factuality gates participant retrieval may be feasible with one role and separately adjudicated status/role questions. Its first gate is an independent source audit; no model result or paper claim follows from the join alone.
