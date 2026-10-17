# One-paper research strategy — 17 October 2026

**Prepared by:** Armaggedon, scientific coordinator
**Audience:** Research lead and all independent scientific investigators
**Status:** Strategy/triage only; no inference or agent-work approval; exam break remains until user reactivates on/after 23 October.
**Target:** A self-contained, defensible original-language + pretrained-model scientific result by **15 December 2026**, not a guaranteed ACL acceptance.

## The answer we owe the lead

We do **not yet** have a verified new mechanism or sufficiently replicated stand-alone paper result. We **do have** a specific empirical phenomenon worth one cautious, source-valid replication attempt. Repeatedly declaring an exciting mechanism on preliminary errors has been destabilizing. Stop proliferating main frontiers. Retain all independent branches but select one accountable primary paper bet.

## PRIMARY — Source-substitution under irrelevant entity load

**Candidate paper question:** When an irrelevant, non-answerable named entity is inserted into a two-event context, what determines **which event supplies the retrieved participant**? Can we predict participant-source substitutions, separately from bystander copying, target-order shortcuts, extra token load, lexical syntax, and published mixed positional/lexical/reflexive binding models?

**Grounded preliminary observation (other investigator's local report; original raw files not independently available here):** controlled corrected Qwen7B task **32/32** isolated sentences, **64/64** bare paired answers, **57/64** with irrelevant named bystanders. All **7** errors occurred for **second-sentence targets** and were copies of another **event's participant**, not the bystander. **Investigator's preregistered replication threshold was 8 target-second errors: 7 < 8, so that gate FAILED.** Original designer caught a first-condition confound and reported a lowercasing condition that did not fully explain effect. This is **not** a published discovery and no mechanism follows.

**Paper-level result that could stand alone IF observed:** a preregistered, source-cluster-replicated and multi-model **predictive law/error provenance map** of event-source substitution under controlled, non-answerable entity interference, outperforming competent position/proximity, syntax-aware compound-key, sentence-salience, random-by-candidate, length/token-load and [Gur-Arieh/Geva/Geiger ICLR2026 mixed retrieval](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2eeff35664016c7f0f8aa704f0d9a83e-Abstract-Conference.html) comparators, including novel authentic passages and a constructed counterbalanced testset with transparent provenance and human adjudication. A *selective native intervention* would be bonus/second-stage only; behavioral, explanatory, falsifiable empirical study must stand on its own.

**Priority P0 on 23 Oct+:** not GPU; independently obtain/review complete original report/scripts, both experiments/confound, all raw prediction outputs, Qwen checkpoint/version, prompt templates and heldout document identifiers. Then *source/intake* feasibility and strong-rival prediction certificate. A minimum independently reviewed and disjoint heldout panel and predeclared pass/fail/cluster analysis must be feasible before running models. This is not a claim that an exact sufficient N is already available. If no valid panel or no rival distinction, STOP this paper attempt.

**Methods to freeze before any test:** matched lexical identity and total token count; non-name bystanders, named-but-nonparticipant bystanders, participant-bearing extra event, equal-position insertion, variable entity number, cue salience; target-first/second and role-specific counterbalancing. Primary endpoint **wrong-event participant provenance** (not just overall accuracy); secondary explicit bystander pickup/invalid and correct recipient-local source. Derive specific *different* predictions from the mixed-position/lexical/reflexive model and a capable grammar-aware copying algorithm; if both explain the pattern, no new mechanistic or predictive novelty claim.

## BACKUP 1 — Factuality-conditioned argument retrieval (higher conceptual upside, weaker data gate)

Pilot on six *constructed cross-document* paired passages reported **13/24** answers, **0/6** whole passage groups, and target first **10/12** vs second **3/12**. Separate individual-source context probes gave status **10/12** and participant **11/12**, but do not establish integration failure in the paired context. Source review: **36 cards, 22 provisional, only 2 clear same-document factual-vs-other Agent collisions**; raw cross-document counterpart search not gold.

Potential original question: whether **narrator-asserted vs negated/reported/possible** event status controls argument retrieval separately from entity mention, recency and source framing, and whether a predictor transfers to factual original multi-event text. Position confound large, strict semantics/human gold missing. [MAVEN-FACT EMNLP2024](https://aclanthology.org/2024.findings-emnlp.651/) is close prior art. This is NOT the current primary and should not consume new compute during the primary source gate.

## BACKUP 2 — Native reuse vs independent event-role reconstruction (highest mechanism risk)

Qwen7B owner reported 4/5 complete events under one question-label patch, 8/10 cross-document recipient redirections, and failures under changed relation / two-role retrieval. [FRONTIER-002](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33#issuecomment-6232764849) correctly stopped when semantic event identity and capable syntax/content copier predict same outputs in 8/8 factorial cells. [FRONTIER-003](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33#issuecomment-6232931287) demonstrates that shared-prefix patching can affect both roles with fully independent readers. No isolated pathway-specific causal mediator known. Binding/index and causal-abstraction prior art very close. Preserve as independent conceptual/negative-control programme, but do NOT sell it as the December paper until a genuinely discriminating native intervention is specified.

## Other streams — retain as audit/calibration, not competing headline papers

- Nova/Mira/Brain-X/Sentry official QA-SRL Gold audit: original human question→span data and exposure/corpus validity; no novel two-family natural mechanism demonstrated yet. A 5-fold lexical baseline at **5616/6590** (~85.2%) on selected SemEval relation orientation in Comput R025 raises shortcut standards.
- Brain-O / Compass synthetic role/composition work provides valuable falsifiers, but source-specific native interventions did not deliver robust whole-world recovery (Brain-O E057–059 reported **0/200** fully correct worlds per seed), and simple parsers can solve some constructed benchmarks.
- Source study [MAVEN-Arg](https://aclanthology.org/2024.acl-long.224/) original event annotations useful, but bad type/argument cases demand human review and source rights. Original vs constructed pairs MUST be honestly labeled.

## One-paper pass condition (not yet met)

A standalone paper must have, at minimum:
1. **A real and reproducible effect** on a heldout, sufficient, independently adjudicated and source-disjoint original-language testset (report uncertainty at document cluster level), with synthetic/constructed causal manipulations labeled separately.
2. **A nontrivial explanatory result**: preregistered, out-of-sample wrong-event *source prediction* beating strong position, role, lexical/syntax, answer-format, named-entity/token-length and ICLR2026 mixed binding competitors, or a clear new controlled empirical law beyond those precedents.
3. **Two-family evidence** with identical answer contracts where possible; negative results and fail gates retained.
4. An independently reviewed **novelty map** against ICLR2026 Mixing Mechanisms, Shuang et al. 2024 cross-event confusion, MAVEN-FACT, Dai 2024/2026 binding, and new [Tang et al. ICML2026 entity-state tracking](https://proceedings.mlr.press/v306/tang26ah.html), which challenges naive “must accumulate state” explanations.
5. A paper narrative **complete without a lucky circuit**: falsifiable phenomenon + reliable measurement + predictive source-error model + practical limits. Native path interventions are a conditional additional contribution, never a mandatory foundation for the whole paper.

## Resumption: ONE bounded decision tree, no endless search

**First:** share/receiver-audit old source/output records and identify a feasible human gold testset. If missing, no new model inference.
**Second:** freeze the strongest rival prediction and an out-of-sample evaluation. If no distinguishing witness against published mixed-retrieval competitor, either change the precise research claim *before* touching heldout or stop.
**Third:** one independently approved test; report both successes and failures. If passes, commit to one manuscript. If fails, freeze this branch and select BACKUP 1 via a separate data+novelty gate. Do not run all branches in parallel as headline research.
**Governance:** #31 scientific approval gate; #33 evidence archive; #23 Dec 15 original-language deadline. No new runs or team assignments authorized by this note.
