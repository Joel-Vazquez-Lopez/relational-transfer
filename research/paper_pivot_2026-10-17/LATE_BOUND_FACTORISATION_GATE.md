# Late-bound event–role factorization: a falsifiable paper bet

17 October 2026. **Design only.** No new model inference, training, downloads, or API calls were made. The previous one-experiment authorization ended at the [no-witness gate](../decisive_witness_gate_2026-10-17/REPORT.md). This is a candidate successor to [PAPER_BET_V2.md](PAPER_BET_V2.md), not a completed result.

## The claim we could actually test

In a natural passage with two events, a model may carry a **compact, causally used event-source variable** at a prefix boundary after the event cue and before the requested participant role. Downstream computations may use that variable to answer either Agent or Patient. This is a *functional* claim. A content-address key or an ordinal index would count if it meets the causal criterion. It would not establish a unique semantic algorithm, a human-like event concept, or a dedicated neuron.

The novel part, if any, would be **cross-role causal reuse under natural, human-adjudicated multi-event prose, with the source of the address identified and strong alternative pathways tested**. Merely finding a reusable index is too close to Dai et al. 2024/2026, RAVEL and other binding work. *Mixing Mechanisms* also explicitly constructs interventions that separate positional, lexical and reflexive retrieval, then separately tests pointer versus answer explanations (§3.2 and §4.4). A paper requires a method-level prior-art audit and held-out predictive advantage.

## Why the attractive simple test fails

Let the common prefix be `P` (passage plus event cue), and let the role suffix be `r`.

- Shared-address program: `z = select(P)`; `answer = read(P, z, r)`.
- Independent-reader program: `M = remember(P)`; `answer = read_A(M)` for Agent or `read_P(M)` for Patient. Each reader may independently resolve the event cue.

For **any** patch of the full prefix state `M`, the independent readers can produce the same two answers as the shared-address program. One construction is for `M` to contain `P`, with both readers computing the same `select(P)` internally. The fact that the model has not yet seen `r` only means the prefix is role-independent by causal masking; it does **not** imply a separately represented event address. An event-only/role-only/fourth-cell output grid after a full-vector patch still admits this construction.

Therefore, **full-residual late-bound patching alone has no separating witness against an unrestricted independent-reader rival**. Do not run that test as a decisive experiment.

**Important audit of the existing work:** the [fresh five-document Qwen experiment](../maven_arg_feasibility_2026-10-16/FRESH_BINDING_REPORT.md) already patched the `EVENT A` question-label state *before* the Agent/Patient wording. A causal Transformer state at that token cannot contain the later role suffix. The experiment then reused an Agent donor for both recipient roles. Therefore merely moving to an explicitly branched prompt is not a new test of late binding; the proposal's extra content must be the restricted, role-held-out subspace and selective mediation design below.

## Narrower, operational rivals

We can test a restricted, measurable pair of claims about a *predeclared rank-`k` subspace `Z` at one token and one layer*:

| Claim | Prediction when an event-switch edit is learned using Agent answers only |
| --- | --- |
| `H_shared`: `Z` is a common causal mediator of event source before role revelation | An edit that changes event source for Agent also changes source for Patient, while each retains its requested role; removing `Z` removes or greatly reduces the event-cue effect for both roles. |
| `H_disjoint`: Agent and Patient select events through disjoint causal coordinates at this site | The Agent-trained edit changes Agent source, but leaves Patient source near sham; removing Agent coordinates does not remove Patient's event-cue effect. |

These are **restricted pathway hypotheses**, not exhaustive models of a Transformer. If both readers use the same compact key, they belong to `H_shared` at this resolution even if their later computations differ. A negative result may mean this site/rank is wrong, not that no shared address exists elsewhere. If role-specific coordinates overlap or edits alter general passage memory, neither simple rival wins; report NARROW/NO-GO.

### Separating witness

Use an unseen passage where E1 and E2 each have distinct, locally explicit Agent and Patient referents and all four unpatched questions are answered correctly. The common prefix identifies an event without mentioning a role. Branch into an Agent suffix and a Patient suffix. Fit the event-switch edit only on **development Agent branches**; freeze the subspace, rank, layer, token, patch magnitude and score. On a held-out document, apply the same edit to the common prefix and read both continuations. A Patient switch to the target event, while remaining a Patient answer, separates the two *restricted* predictions. This must survive equivalent-rank random directions, sham and adjacent-token edits, and must not depend on donor participant names.

Even this witness alone is not enough: a patch that transfers the whole answer-bearing passage into `Z` could change both roles. Test whether the intervention works when donor and recipient have disjoint participant names; score donor-name intrusions; test whether the same edit transfers across unrelated documents. Do not claim that failure to decode names proves their absence. The strongest evidence is **necessity plus sufficiency**: the isolated edit redirects both roles, while targeted removal/neutralization of `Z` reduces both event-cue effects and restoring `Z` repairs them without generally degrading the task. This remains causal evidence for a functional mediator at a stated resolution, not a complete circuit proof.

## Experimental sequence and freeze points

1. **Prior-art gate.** Compare exact intervention sites, rank restrictions, cross-role training/evaluation and natural-text controls with Dai et al. 2024 and 2026, RAVEL, semantic-role circuits, and *Mixing Mechanisms*. If the same claim and design are already tested, retire this paper bet.
2. **Source gate.** Human-review candidate MAVEN-Arg passages blind to Qwen outcomes. Record original source IDs, trigger and role spans, coreference, disputed labels, extra events and exclusions. Require two event types and a predeclared minimum holdout count that survives review; do not count the previous five defeat documents or three scoring documents as fresh. Keep source TRAIN/VALID partition and document disjointness. Do not inspect official TEST.
3. **Prompt gate.** Use one fixed entity-neutral instruction, e.g. `Answer with the participant name or organization named in the passage.` The prefix ends immediately after an event-identifying cue and before any role word. Keep prefix byte-identical for both role continuations. First screen four event × role unpatched questions and direct reversed-label controls. Save every failure and exclusion.
4. **Development only.** Select a *new* prefix-boundary site; the old layer-16 question-label patch cannot be silently reused as evidence because its token context and question wording differ. Fix rank `k`, edit extraction, scaling and score using development documents only. Use Agent answers to choose the edit; no Patient outcome may tune it.
5. **Freeze predictions.** Record `H_shared`, `H_disjoint`, order-index, marker, lexical/content-key and answer-copy predictions per condition. Count disagreement cells before held-out interventions. If `H_shared` and `H_disjoint` have no valid disagreement under the actual edit and controls, STOP.
6. **One held-out run.** For every competent holdout document, run both role branches with unpatched, edit, same-site sham, random matched-rank edit, adjacent-token edit, cross-document donor, `Z` removal, and restoration conditions. Counterbalance event order and marker assignment where the text stays grammatical; pre-review transformed passages. Record all generated strings and scores, including unexpected answers.

## Outcome measures and decision rule

The primary unit is the **document**, not an individual answer. A document is a strict cross-role success only if both roles move to the other event, both retain their roles, all four untreated event × role answers are correct, and preregistered shams do not move either role. Report the number screened, competency failures, intervention failures, wrong-role answers, donor-name copies, ambiguous answers and control failures. Also report answer-level effects separately with uncertainty intervals; do not turn correlated answers into independent trials.

- **GO for the restricted functional claim:** held-out cross-role effect plus targeted necessity/restoration, counterbalanced source behavior, and a material advantage over the frozen disjoint-pathway and order/marker baselines. The exact numeric threshold and sample size must be frozen after source review and before any held-out inference.
- **NARROW:** reproducible answer-source or order steering, or cross-role transfer without selective necessity, but no supported compact common mediator.
- **NO-GO:** source or competence gate fails, shams move, predicted rivals are not separated, or held-out effect does not replicate.

An unrestricted independent-reader construction remains logically possible under finite interventions. The paper must name the restricted comparison it defeats and avoid the unqualified phrase “the model selects an event first.”

## What makes this worth a paper rather than another pilot

A positive study would give a held-out, role-transfer and targeted-mediation result in independently adjudicated natural multi-event language, plus an explicit boundary between ordinal and content-sensitive addressing. A negative study could still support a careful 15-credit thesis; a publishable negative paper would need a substantial validated benchmark or a generalizable methodological result beyond one Qwen site. Neither publication nor novelty is established today.

Primary sources to audit: [Dai et al. 2024](https://aclanthology.org/2024.emnlp-main.967/), [Dai et al. 2026](https://aclanthology.org/2026.acl-long.2194/), [RAVEL](https://aclanthology.org/2024.acl-long.470/), [semantic-role circuits](https://aclanthology.org/2026.findings-acl.1964/), and [Mixing Mechanisms](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf).
