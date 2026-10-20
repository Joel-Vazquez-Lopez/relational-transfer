# Fixed-layer event-label transfer on five new source documents

17 October 2026. This is a small, document-disjoint follow-up within the same corpus and event family. It tests a locked intervention; it is not a mechanism proof or an independently adjudicated benchmark.

## Design and source screen

After a three-document layer search and a one-document follow-up, layer **16** and the question's `A` label-token output residual were fixed. Five unused MAVEN-Arg training documents were selected before this run. Each contains two annotated `defeated` events with four distinct named participants. The [selection record](FRESH_BINDING_SELECTION.json) lists document and event IDs and records one candidate rejected before inference because its sentence named two defeated people but only one annotated Patient. The [builder](build_fresh_binding.py) verified the source archive hash, document disjointness, local mentions, and trigger offsets. The [items](FRESH_BINDING_ITEMS.json) retain the natural contiguous source excerpts and add A/B or B/A trigger markers. Some excerpts also contain other, unmarked match outcomes; those remain in the test.

The recipient reads the A/B passage and asks for EVENT A's Agent or Patient, which names the first selected match. The donor reads the B/A passage and asks for EVENT A's Agent; its A names the second selected match. The donor's question-label state was captured before role wording and patched into both recipient role queries at layer 16. Full greedy answers were decoded with the patch reapplied at every generated token. Unpatched, direct B/A, same-passage identity patch, and neighboring `EVENT`-token patch conditions were all run. The [protocol](FRESH_BINDING_PROTOCOL.json) fixed documents, source/model/script hashes, conditions, full-answer scoring, and the strict document criterion before inference.

## Fixed-layer results

| Full-answer result, 10 role queries in 5 documents | Exact count |
| --- | ---: |
| Unpatched A/B → first-event participant | 10/10 |
| Direct B/A → second-event participant | 10/10 |
| B/A label state patched into A/B → second-event participant | **9/10** |
| Neighboring `EVENT`-token donor patch → first-event participant | 10/10 |
| Same-passage identity patch → first-event participant | 10/10 |

The strict both-role and all-control criterion passed in **4/5 documents**. In Bad Blood (2004), Agent switched from Triple H to Chris Benoit, but Patient remained Shawn Michaels rather than switching to Kane. This failure is retained in the denominator. The [summary](FRESH_BINDING_SUMMARY.json) gives each document's result, and the [raw responses](FRESH_BINDING_RAW.jsonl) include every complete answer.

## Adaptive cross-document diagnostic

After seeing the above result, I froze a separate [diagnostic protocol](CROSS_DOCUMENT_BINDING_PROTOCOL.json). Each recipient was paired with a different article whose four annotated participant names did not overlap. A foreign donor state came from a question using the same literal `A` label. The donor's A/B passage mapped A to the first selected event; its B/A passage mapped A to the second. The recipient always read its own A/B passage. This examines whether the state carries only the donor article's people or can redirect a different passage by event order.

| Foreign donor condition | Recipient first-event exact | Recipient second-event exact | Donor-name exact |
| --- | ---: | ---: | ---: |
| A/B donor, A=first | **10/10** | 0/10 | 0/10 |
| B/A donor, A=second | 2/10 | **8/10** | 0/10 |

The two B/A misses were Patients in Bad Blood (2004) and WrestleMania 32; both stayed with the first-event Patient. [All 20 answers](CROSS_DOCUMENT_BINDING_RAW.json) are preserved. This diagnostic was adaptive, used the same five articles, and placed foreign states into the recipient computation, so it is not confirmation of a general code.

## What the result supports

The question-label state can transfer information about **which marked event to use** across roles and, often, across documents while leaving the recipient's participant names in the answer. An order-indexed cue is a plausible account: A names the first or second marked event. The two Patient failures and the earlier fourth-passage failure show that the cue does not reliably carry a complete winner–loser tuple. The evidence currently fits a transferable event-selection signal combined with role-sensitive downstream retrieval better than a rigid shared-event record, but the entire activation vector was transferred. A contextual cue mixture, positional shortcut, or other distributed mechanism remains viable.

Binding and ordering interventions already have relevant prior art, including [Representational Analysis of Binding in Language Models](https://arxiv.org/abs/2409.05448) and [Task Schema and Binding: A Double Dissociation Study of In-Context Learning](https://arxiv.org/abs/2512.17325). The possible contribution here is narrower: a predictive account of event–participant routing in natural multi-event text that beats matched position, lexical, and fieldwise explanations on held-out data. This five-document result does not establish that contribution.

## Next discriminating gate

Use a new relation family and independently reviewed passages, with fixed layer/site and full-answer scoring across every participant. Counterbalance event order and label placement separately. A position-only rival should predict where a transferable order cue works; a content-sensitive event account should predict cases where labels refer to different event orders but the same semantic event. The next analysis must state those rival predictions before inference and include both role failures and control failures.

All runs used the existing pinned local Qwen2.5 7B Instruct MLX 4-bit model. No Berget or other paid API calls were made.
