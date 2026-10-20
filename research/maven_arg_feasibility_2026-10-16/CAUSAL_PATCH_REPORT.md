# Event-label state transfer: development result and failed full-answer follow-up

**Later 17 October follow-up:** The fixed layer/site was tested on [five new source documents](FRESH_BINDING_REPORT.md). Complete second-event answers appeared in 9/10 patched role queries and both roles passed in 4/5 documents, with controls correct. An adaptive foreign-document donor diagnostic redirected 8/10 answers to the recipient's second event and copied no donor names. This makes a transferable event-order cue plausible while retaining the documented Patient failures and the need for another relation family.

17 October 2026. This is a calibrated intervention on one local open-weight model. It does not identify a unique event variable or establish a novel mechanism.

## Why run it

The three-event [behavioral panel](STRESS_REPORT.md) was nearly perfect under explicit markers, leaving too few natural wrong-event answers to distinguish event-first from fieldwise retrieval. I tested whether a question-label hidden state could redirect the answer when the passage changed which event the label named. The source sentences are natural MAVEN-Arg excerpts; the A/B/C marker labels were added for this experiment.

## Probe 1: literal label replacement

The [first frozen protocol](CAUSAL_PATCH_DEV_PROTOCOL.json) swapped the output residual at the question's `B` label token into an `A` query, with an Agent donor used for both recipient roles. On three development passages, layers 0–16 moved the first-token preference toward the B participant for all **6/6** event-role queries; layers 20–27 did not. Same-label identity patches preserved the A preference. However, this transfers the literal B token representation. Directly asking for B produced nearly the same result. This probe mainly established that the forward and patching code works; it was not a discriminator of binding.

## Probe 2: same-letter passage binding

The [second frozen protocol](CONTEXT_BINDING_PATCH_DEV_PROTOCOL.json) held the literal question `EVENT A` fixed. In the recipient's A/B/C passage, A marked the **first** defeat; in the donor's C/B/A passage, A marked the **third** defeat. The donor question asked for Agent. At the question-label token, the donor state is computed before the role-specific words, so that same state was transferred into both Agent and Patient recipient questions. Three passages were used to examine eight fixed layers. The neighboring `EVENT` token and same-passage A state were patched as controls.

At **layer 16**, the third-event first-token score exceeded the first-event score in **6/6** development queries. At the other seven sampled layers it did so in **0/6**. Both controls stayed with the first event in **6/6** at layer 16. This layer was selected from these development outcomes. A later adaptive full-generation diagnostic confirmed the third event's complete annotated name in **6/6** patched development responses, including both roles in all three passages. [Development raw logits](CONTEXT_BINDING_PATCH_DEV_RAW.json) and [full-answer diagnostic](CONTEXT_BINDING_DECODE_DIAGNOSTIC.json) preserve the records.

| Layer-16 condition, development | Third-event complete answer |
| --- | ---: |
| Same-letter donor label state | 6/6 |
| Identity or adjacent-token control | First-token preference stayed with first event in 6/6 each |

This is a passage-specific effect beyond the literal query letter. It remains compatible with a general context-carrying state, a local pointer, or copied information about several participants. It does not isolate an abstract event representation.

## Fixed-layer check on the fourth passage

The fourth passage, SummerSlam, was not used to select the patch site or layer. The [follow-up protocol](CONTEXT_BINDING_HOLDOUT_PROTOCOL.json) locked layer 16 and required **both complete role answers** to switch to the third event, while the adjacent-token and identity controls stayed with the first. This is a small follow-up on a previously studied document, not an independently sampled confirmation set.

| Role | Unpatched A/B/C | Direct C/B/A | Patched question-label state | Third-event gold |
| --- | --- | --- | --- | --- |
| Agent | Daniel Bryan | Brock Lesnar | **Brock Lesnar** | Brock Lesnar |
| Patient | John Cena | CM Punk | **Christian** | CM Punk |

Both control patches returned the unpatched first-event names for each role. The predeclared **both-role primary outcome failed**: one patched answer named the second event's Patient. The third-event versus first-event first-token margin was positive for that Patient query, yet `Christian` won generation. This is why full-answer and all-candidate scoring matter; a two-candidate logit margin overstated the transfer. [All ten follow-up responses](CONTEXT_BINDING_HOLDOUT_RAW.json) are saved.

## Current inference

There is a reproducible development-stage causal effect at the question-label position: the state can carry passage-specific information that redirects answers across roles, with a narrow sampled-layer window and controls that do not reproduce it. The fixed-layer fourth-passage result shows the proposed complete role binding is **not reliable** even within this small event family. The evidence supports continued research on how event cues route to participants; it does not yet support the shared-event mechanism claim.

This also has a clear novelty boundary. Activation patching of in-context bindings and order information has prior art in [Representational Analysis of Binding in Language Models](https://arxiv.org/abs/2409.05448) and [Task Schema and Binding: A Double Dissociation Study of In-Context Learning](https://arxiv.org/abs/2512.17325). Event-relation interventions also exist in [UERLens](https://aclanthology.org/2026.acl-short.38/). A contribution here would need a precise, independently validated prediction about **natural event–participant role binding** that matched context-copy, lexical, position, and fieldwise rivals miss.

## Next gate

Build a larger, independently adjudicated set of natural multi-event passages, with all candidate participants scored during full generation. Lock a small number of patch sites from development documents, then measure new correct answers, wrong-event borrowing, wrong-role answers, and all-candidate probabilities on held-out documents and another relation family. A shared-event account should be dropped if a context-carrying cue mixture or fieldwise model predicts the same responses and intervention effects.

All three patch runs used the existing pinned Qwen2.5 7B Instruct 4-bit MLX model locally and made no paid calls. The protocols lock source/model/script hashes. The manual full-prefix patch recomputed the modified state at each generated token for the follow-up and diagnostic.
