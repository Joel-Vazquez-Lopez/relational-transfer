# FRONTIER-003 — Causal bottleneck versus independent role readers: design-only prediction certificate

**Date:** 2026-10-17  
**Status:** HYPOTHESIS/IDENTIFIABILITY CERTIFICATE ONLY — **NO NEW MODEL RUN APPROVED**  
**Coordinator:** Armaggedon  
**Relationship to decisions:** Does **not** reverse the [FRONTIER-002 NO-GO](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33#issuecomment-6232764849). A *different* mechanistic question, pending scientific gate [#31](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31).  
**Research break:** User asked to pause until 23 Oct 2026; archive note only, no agent tasks launched.

## Executive result: a proof that the proposed pre-role patch alone is insufficient

An upstream hidden state shared between subsequent role questions is **not proof that a shared *event-selection computation* exists**. If the same prompt prefix carries event cue q, then an independent Agent reader and independent Patient reader each can reconstruct the target event from q and the context. Patching that prefix may redirect **both** despite the absence of a single role-independent selector.

Thus the immediate “pre-role patch changes both” hypothesis has **zero discriminatory power against a competent role-specific reconstruction rival**. Our surviving alternative research question is whether both *native answer paths* require the **same causally necessary, sufficiently event-specific, role-independent mediator**, as opposed to disjoint role-conditioned event computations. This is a path-level architectural hypothesis, not proof of a uniquely semantic pointer.

## I. Minimal explicit rival programs — identical outputs under all prefix-cue changes

Let the original text x contain two correctly annotated events, indexed e∈{0,1}, each with distinct Agent A_e and Patient P_e. Let q identify the target event; r∈{A,P} identify the role. Each role is queried in a **separate** continuation after the common prefix. Hold original gold, answer format and access to x,q equal.

### S — shared address mediator

\`\`\`
Z = SELECT_EVENT(x, q)                  # computed once before role readout
ANSWER_S(x, q, r) = LOOKUP(x, Z, r)     # role-conditioned downstream retrieval
\`\`\`

### I — competent independent role readers

\`\`\`
# There is no single SELECT_EVENT call causally required by both readers.
# Each reader receives the SAME prefix content and event cue as S.
ANSWER_I(x, q, "Agent")   = READ_AGENT(x, q)    # computes target internally
ANSWER_I(x, q, "Patient") = READ_PATIENT(x, q)  # separately computes target
\`\`\`

Choose READ_AGENT(x,q)=A_q and READ_PATIENT(x,q)=P_q. Both architectures return the exact same four answers. A complete, high-capacity upstream prefix state H(x,q) can hold x and q, so replacing H(x,q=0) with H(x,q=1) redirects Agent and Patient in **both** S and I. Even perfect cross-role transfer **does not identify S**.

The independent-reader model is **not a straw man**: it can use syntax, lexical information, event content, discourse, order, labels and context to choose a record separately for each role. Neither output coherence nor “early patch then two role queries” excludes it. A compound-key copier that constructs a shared role-independent key belongs to the **shared functional mediation class** for this question, regardless of whether the key is called “semantic”.

**Unrestricted-equivalence warning:** if I is allowed arbitrary internal sharing, copying, reconstruction and all possible paths, the classes collapse. One cannot prove a negative about all possible representations from finitely many patches. A valid experiment must freeze a **restricted causal graph and interventions** with genuine disagreement and concede an equivalence class where no such separation is possible.

## II. Conditional distinguishing witness: native path-specific mediator, not prefix-only patching

Define candidate internal variable Z and distinct outgoing **actual model edges / paths** toward separate Agent and Patient readouts. Interventions below are only meaningful after a *real site and edge/path decomposition* has been identified on **development-only** material. There are no currently identified such paths in this dossier; this is NOT an executable preregistration.

**Shared-path hypothesis S\***: Both readouts' event-specific behavior is substantially mediated by the *same source-independent, role-independent* Z, i.e. Z transfers **event address** but not donor participant names or the full text, and direct bypass routes carrying target address are controlled on specified recipient paths. Distinct downstream readouts recover A or P from the unchanged **recipient local context**.

**Independent-path hypothesis I\***: In the frozen graph model, two independent role-specific event selection paths from common context/cue to role answer exist, and **neither requires candidate Z or its projected outgoing connections**. The model can access identical x,q before intervention; shared prefix is permitted. Context-consuming routes remain intact in surgical tests.

This is an **operational graph distinction**, not a claim that S\* exhausts semantic understanding or I\* exhausts all possible copying algorithms.

### Required prediction matrix (logical, conditional on specified graphs)

| Condition | Shared Z bottleneck S* | Path-disjoint I* | Decision |
|---|---|---|---|
| C0: intact source context, role asked separately | Both roles correct if competent | Both roles correct if competent | Nonidentifying |
| C1: replace **entire common pre-role hidden state** H with donor | Both roles may redirect | Both roles may redirect | **Nonidentifying** |
| C2: **edge-isolated Z-only donor→recipient swap**, other candidate event cue/direct-answer flows clamped to **recipient**, preserves role | Correct recipient-local donor-target event Agent **and** Patient | Both remain recipient-target or do not coherently redirect, **if** they truly bypass Z | **Potential separating witness** after paths are verified |
| C3: selectively block Z→Agent and Z→Patient mediator edges, but preserve separately matched generic text/context flows | Predict selective degradation of event-addressing for both roles; unrelated retrieval preserved | Little selective change if neither role uses Z; global damage invalidates inference | Potential necessity contrast |
| C4: restore the SAME Z-only edges to C3 | Both role answers specifically recover correct recipient-local event identities | No corresponding shared-bottleneck rescue predicted, conditional on C3 | Potential mediation/rescue evidence |
| C5: block role-specific downstream Agent path only, preserving Z and Patient path | Agent disrupted; Patient target/event preserved | Agent disrupted; Patient preserved | **Nonidentifying** by itself |
| C6: swap event-only Z and independently role-specific downstream route, with path controls | Predicted four crossed event×role combinations | May show different outcome only when actual independent paths and their causal inputs are frozen | Conditional interaction test |
| C7: reorder events, relabel arbitrary IDs and change names/roles independently | If Z is content/event-linked, behavior should track source-defined event identity under predeclared alignment | Strong I* may also track semantic content perfectly | **Nonidentifying without path restriction** |

The words “if candidate Z” and “if graph really disjoint” are essential. **C2–C4 are not existing empirical witnesses.** They are a logical certificate that would separate **restricted** graph alternatives *if* one can implement matched surgical edge interventions without leakage or generic damage. If Z includes enough copied passage information for downstream readers to rebuild the answer, that counts only as a **shared upstream information channel**, not automatically a narrow event-selection computation. Require feature/route specificity tests below.

## III. How the rival is allowed to WIN

Predefine and retain the following defeating outcomes; do NOT rescue the hypothesis by selecting a new layer on heldout data.

1. **Independent-reader win:** both Agent and Patient remain correct under genuine Z-only lesion because separate role-conditioned event-reading paths carry the cue. If a whole-state patch redirects both, classify as **shared-context confound**.
2. **Distributed/redundant computation:** multiple parallel paths carry address information, no one causally sufficient mediator; local Z ablation has little effect even though the behavior is competence-matched. Call **NO SINGLE NECESSARY BOTTLENECK**, not “no relation representations”.
3. **Content-copier explanation:** candidate Z appears to encode enough lexical/event text or position to reconstruct source selection; equal-capacity, syntax-aware joint copier gets the same causal predictions and shams. Label **SHARED CONTENT ROUTE / NOT UNIQUELY EVENT-SELECTIVE**, not S*.
4. **Event-order cue explanation:** within recipient, redirected output follows first/second slot rather than invariant event content when valid independent order/cue counterbalancing is possible. Label **ORDINAL STEERING**.
5. **Damage/cross-talk:** role-independent generic impairment, invalid strings, donor-name leakage, answer-token copying or similar shams work as well as candidate Z. **FAIL causal specificity**.
6. **No valid graph-specific intervention:** Unable to identify a Z-only mediator edge/path independent of generic x/q flows with matched strong alternative access. **STOP BEFORE NEW INFERENCE**. This is a scientifically valid outcome, not an invitation to keep tuning.
7. **Prior art covers result:** If the result is an instance of already known binding-ID/index causal transfer or RAVEL-like feature disentanglement without a new natural multi-event predictive property, **NO NOVELTY CLAIM** even if empirical mechanism test works.

## IV. What is genuinely new versus what literature already covers

- [Dai et al., ACL 2026, Cell-Based Representation of Relational Binding](https://aclanthology.org/2026.acl-long.2194/): causal cell/index binding representations, linear subspaces, cross-context transfer. **Particularly close to route/index transfer.**
- [Huang et al., ACL 2024, RAVEL](https://aclanthology.org/2024.acl-long.470/): quantitative representation disentanglement, multitask distributed alignment search and interchange interventions. “Find separable features” alone is not new.
- [Geiger et al., JMLR 2025, Causal Abstraction](https://www.jmlr.org/papers/v26/23-0058.html): unified causal formalism encompassing activation/path patching, mediation and scrubbing. “Path patching proves causality” alone is not new.
- [Dai et al., EMNLP 2024, Representational Analysis of Binding](https://aclanthology.org/2024.emnlp-main.967/): already investigates binding-related internal causal directions.

**Candidate scientific contribution if a real future witness exists:** *Causally demonstrate and predict an event-selective mediator that is reused by multiple role computations in original human-annotated multi-event discourse, while independently capable reader mechanisms and direct lexical/order/answer-copy pathways fail predeclared, meaning-preserving contrasts.* This is a question and criterion, **not an established mechanism or novelty result**. Full-paper method-level comparison required.

## V. Preflight gates; task is currently PAUSED

Before even an exploratory new run, a receiver must: (a) inspect exact historical raw local reports/checkpoint and role/patch failures; (b) demonstrate a credible native graph boundary whose isolated Z manipulation changes the predictions of S* and I* while holding the same x,q access; (c) independently review human original-source annotation and licensed text; (d) pin training-development versus untouched test clusters and accessible weights/hardware; (e) predeclare **actual recipient-local answer** for both roles under every surgical condition; (f) freeze no-go thresholds and resource limit; (g) check full-text prior art and obtain separate research-lead approval in #31 after the break.

**Theoretical gate result TODAY:** \`PREFIX-ONLY=NO-GO\`; \`CONDITIONAL PATH-MEDIATION=UNPROVEN/NOT READY TO RUN\`. There is *no non-circular, presently executable distinctive prediction* unless paths can first be isolated. Preserve this as an attractive theoretical frontier without claiming the no-run Codex decision was overturned.

**Research governance:** this note is the coordinator's design-only archive. **No model inference, GPU, paid API, data download, corpus panel selection, Codex task, or other agent work was started.**