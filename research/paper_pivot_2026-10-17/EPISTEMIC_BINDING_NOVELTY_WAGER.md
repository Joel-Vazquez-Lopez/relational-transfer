# Epistemic binding: a falsifiable novelty wager

17 October 2026. Research proposal and source-only audit. **No new model inference or causal result.**

## The computation worth asking about

When a passage says *A tried to capture X but failed* and *B captured Y*, it introduces participants for both described events but commits only to the latter capture. A reader can answer “who tried?” with A and “who captured?” with B. The interesting failure mode is **event instantiation leakage**: a model correctly recognizes the failed attempt as nonfactual when asked directly, but retrieves A as though A completed the event when asked for an actual participant.

The wager is that factuality information may be available yet **not reliably coupled to the event–participant lookup**. This is a prospective prediction, not an observed Qwen failure. If it appears selectively under matched distractors, it is a sharper result than a generic failure to understand “not.” If it does not appear, a successful compositional retrieval result is still meaningful, but the paper then needs a causal test beyond ordinary accuracy.

## Prior work boundary

- [MAVEN-FACT (2024)](https://aclanthology.org/2024.findings-emnlp.651/) provides event-mention factuality and examines whether supplied argument information improves *factuality classification*. It does not establish how a model uses factuality to gate retrieval of an event's participant in a competing-event question. Its CT−/PS+ labels are useful source annotations, not ready-made QA gold.
- [MAVEN-Arg (2024)](https://aclanthology.org/2024.acl-long.224/) supplies event arguments on overlapping documents. The exact-ID join is real, but the unit mismatch creates invalid derived questions; our [72-document source review](FACTUALITY_SOURCE_REVIEW_ROUND2_2026-10-17.md) makes this explicit.
- [Dai et al. 2024](https://aclanthology.org/2024.emnlp-main.967/), [Dai et al. 2026](https://aclanthology.org/2026.acl-long.2194/), [RAVEL](https://aclanthology.org/2024.acl-long.470/) and [Mixing Mechanisms (2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf) set the standard for causal binding and disentanglement. We cannot claim discovery of a binding vector, role circuit, or event selector alone.
- [Modality and Negation in Event Extraction (2021)](https://aclanthology.org/2021.case-1.6/) and earlier factuality work show that treating mentioned events as actual is a longstanding NLP problem. A new benchmark that merely counts such errors would be incremental. The target here is the **cross-task dissociation and its causal pathway** in a locally inspectable model.
- [Schuster and Linzen (2022)](https://aclanthology.org/2022.naacl-main.71/) already find that transformer models sometimes treat entities in negated sentences as available discourse referents, especially with multiple noun phrases. [Sulem et al. (2021)](https://aclanthology.org/2021.findings-emnlp.385/) study competitive unanswerable event questions. These sharply constrain novelty: neither “models ignore negation” nor “distractors cause wrong answers” is new. We would need event-specific source/status/participant conjunctions, out-of-sample prediction, and selective causal evidence.
- [QAEVENT (2024)](https://aclanthology.org/2024.findings-eacl.126/) already represents event extraction as QA pairs. Our joined annotation audit must not be presented as inventing event QA.

This is a scoped literature check, not a guarantee nobody has studied an equivalent experiment. The exact closest-method comparison must be repeated before submission, especially for work published after this audit.

## Testable design

**Development corpus:** two authentic source sentences, independently reviewed, placed together as an explicitly *constructed pair*. One describes an actual event; the other a failed, negated, planned, possible, or attributed event with a different participant. Prefer the same predicate lemma; require an explicit, local participant in each. Use unfamiliar names where possible, or test name-swapped versions separately to measure world-knowledge contamination. The source sentence and its transformation must never be confused.

**Questions per pair:**

1. Status of each target event, including whether the passage establishes occurrence.
2. Participant in each *described* event.
3. Participant in the *established actual* event, with `NONE/NOT ESTABLISHED` available.
4. A role-specific version on a subset with independently validated second roles.

**Counterbalances:** factual sentence first/second; failed/possible/alleged cue near/far from trigger; participant and event noun swapped where grammar permits; two prompt phrasings; same-lemma versus different-lemma strata. Score whole output, status, participant, and complete-pair conjunction separately. Document and source-sentence IDs define clusters; never split transformed variants of one source sentence across development and held-out sets.

**Prospective behavioral prediction:** Qwen should do better on direct status classification and described-participant retrieval than on the *joint* established-participant question when a nonfactual and factual event share a predicate. The largest gap is predicted for attributed or possible events, smaller for explicit `not` or `failed`. This prediction must be frozen before Qwen sees held-out pairs. A syntax-aware copier predicts no systematic gap once it has enough context; a proximity copier predicts failures that track order/distance rather than status class.

**Mechanistic follow-up, only if behavioral competence exists:** On development pairs, learn a low-rank status edit from a status-only task and an event/participant edit from a retrieval task at sites fixed by a priori layer bands or development data. Freeze before held-out pairs. Apply each separately and jointly to the same recipient prefix *before answer type is revealed*. Evaluate whether the status edit changes established-event answers without changing described-event participant identity, and whether the participant edit changes the named source without changing status. Compare to norm-matched random directions, neighboring-token, same-document identity, answer-token, and full-vector controls. A full-vector patch alone cannot establish independent variables. Report off-target damage and the complete 2×2 outcome table.

**Interpretation limit:** A sophisticated copier with a reusable status bit and event key belongs to the same functional equivalence class as a factorized retrieval architecture. Even perfect selective transfer would support a reusable causal factorization under the tested edit budget, not a uniquely “semantic” event object. If a strong syntax-aware copier reproduces the outputs and interventions, narrow the claim to a predictive model of behavior.

## Source state and stop rules

- Two disjoint, 36-document source samples yielded **44/72 provisional single-event accepts** (round one 22, round two 22). These were reviewed by one nonblinded reader; the number is not a benchmark or representative acceptance estimate. See [round one](FACTUALITY_SOURCE_REVIEW_2026-10-17.md) and [round two](FACTUALITY_SOURCE_REVIEW_ROUND2_2026-10-17.md).
- Only two round-one accepted cards had a structurally eligible same-document factual distractor. Thus the *untouched natural-document* primary design fails its own source gate. Cross-document sentence pairing is a new controlled-corpus design, with untouched documents reserved for external validity.
- The validation pool has raw same-type factual counterpart candidates for all 22 first-round accepted cards. That is an availability count, not a semantic match. Broad types such as `Getting` contain incompatible predicates.
- No held-out inference until at least 30 matched pairs from at least 30 independent source documents have explicit role/status gold, a second reviewer resolves disagreements, all question wording and status labels are frozen, and at least one condition yields different predictions for the rival accounts. Otherwise stop or present a descriptive course project without a mechanistic paper claim.

**Publishable positive outcome:** a reproducible benchmark with enough source-valid pairs, a preregistered cross-task dissociation, and a selective causal status/participant intervention that generalizes across cue families and event types beyond prior binding tasks. **Publishable negative outcome:** sufficiently precise evidence that local Qwen lacks an independent status gate despite direct status competence, or that the gate is robust and the feared leakage does not occur, with adequately powered controls. A few striking examples or one-vector patch is not enough.

## Development outcome and revision of confidence

After writing the prospective wager, a [frozen six-pair Qwen competence pilot](EPISTEMIC_COMPETENCE_DEV_REPORT.md) scored **13/24 individual answers and 0/6 complete pairs**. Target-first questions scored 10/12; target-second scored 3/12. This is a development result with a generic prompt and a convenience sample, but it **does not pass the competence gate** for the proposed causal intervention. The apparent first-sentence effect is an alternative to the status-leak interpretation. No activation experiment should be run on these items. The next experiment must first test status-only and described-participant competence separately on new, independently reviewed pairs, then make a frozen joint-task prediction against positional and syntax-aware copier baselines.

The stronger prospective paper question is now: **when the model individually knows an event's status and participant, does combining those facts fail in a predictable, order-sensitive way?** The pilot did not measure individual status competence, so it cannot answer this. The novelty would be a *compositional failure boundary and its causal mediation*, beyond earlier negated-entity and unanswerable-QA findings, if a clean new panel supports it. If the model fails the component questions too, call it ordinary task failure and stop.

A [post-pilot, frozen component diagnosis](EPISTEMIC_COMPONENTS_DEV_REPORT.md) on the same exposed sentences scored **10/12 status** and **11/12 Agent**, with three pairs passing all four component questions. This makes the composition question worth a **new, single preregistered test**, but it is not independent evidence for a mechanism. The joint prompt differs from the component prompts and the first-sentence pattern could account for the errors. No further adaptive patching on these six items.
