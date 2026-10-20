# Paper bet: do binding-route models generalize to authentic multi-event prose?

17 October 2026. **Prospective publication target, not an established finding or permission for another inference run.**

## The precise question

Can the positional, lexical and direct-pointer retrieval account established on controlled binding tasks predict **which event's participant** a pretrained LLM answers with under causal intervention in authentic, human-annotated documents containing multiple same-type events? If it cannot, is the residual explained by a predeclared syntax/discourse-sensitive route that makes new held-out predictions?

This replaces the attempted unique “semantic event selector” claim. It asks whether an existing, published causal model generalizes to a different setting and what additional computation, if any, is needed. A strong answer can be positive or negative. Calling a failed replication “new mechanism” without a predictive replacement would not be a paper.

## Closest work and the actual gap to test

- [Gur-Arieh et al., *Mixing Mechanisms*, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/2eeff35664016c7f0f8aa704f0d9a83e-Paper-Conference.pdf) explicitly designs counterfactual patches that distinguish positional, lexical and reflexive binding predictions (§3.2), fits a causal mixture, tests nine models and ten templatic tasks, and checks open-ended text **interleaved with entity groups**. Their paper already owns the general multi-route and witness idea. Our possible gap is **fully authentic event syntax, discourse and original human event-role annotations**, plus a new predictive failure boundary; natural wording alone is insufficient.
- [Dai et al. 2024](https://aclanthology.org/2024.emnlp-main.967/) and [Dai et al. 2026](https://aclanthology.org/2026.acl-long.2194/) already establish causal binding-ID/cell-style steering and cross-context transfer. Dai 2026 also examines Re-DocRED, so “first real text” is false.
- [TabEAE](https://aclanthology.org/2023.acl-long.701/), [EAESR](https://aclanthology.org/2024.findings-acl.328/) and [Shaar et al. 2025](https://aclanthology.org/2025.tacl-1.71/) cover multi-event extraction, cross-event confusion and explicit trigger provision. This proposal is about **out-of-distribution prediction of intervention outcomes**, not simply better extraction F1 or marker gains.

## What would count as a paper contribution

1. **Source-valid challenge set:** original human-gold event/role annotations with distinct participant referents, documented ambiguity, and source-document splits. Independent adjudication of selected events and any sentence-order or voice transformation. Report rejected candidates, source rights, and annotation agreement. Existing MAVEN-Arg is a starting point, not automatically valid gold for every role.
2. **Frozen replication target:** implement the published three-route account as faithfully as possible using its released code/specification. Establish on its own controlled regime that the implementation predicts the published intervention categories before testing natural text. This prevents a bad reimplementation from masquerading as a domain-shift result.
3. **Prospective natural-text prediction:** on development documents, fit only predeclared parameters. Freeze per-condition predicted output categories or distributions for unseen documents, including both role questions, event order/marker crossings where linguistically valid, and shams. Compare to a competent syntax-aware compound-key rival with matched access and fitting budget.
4. **New explanatory boundary:** If the published route model systematically fails, specify a concrete syntax/discourse feature or native pathway that predicts which failures occur, and verify it on a second source or relation. Without that, report a bounded replication failure, not a new mechanism.
5. **Complete outputs:** per-document and per-answer results, original→target acquisition, all other participant names, invalid answers, failed competence screens, and negative controls. Do not aggregate selected competent cells as if random samples.

## Feasibility facts and limits today

- The existing Qwen 7B question-label patch gives a credible pilot: 4/5 strict fresh defeat documents, 8/10 cross-document role redirections, and 3/3 scorer redirections in another relation. It does **not** distinguish event identity from a compound-key copier, and scoring's two-role baseline failed. These are development-exposed and cannot be reused as confirmation.
- A quick local, source-only MAVEN-Arg heuristic screen found **66 fresh TRAIN and 18 fresh VALID candidate documents** at a strict 180-character trigger-to-role-mention distance, four distinct strings, and two same-type Agent/Patient events. At 600 characters, 95 + 22. These counts precede human semantic review and may shrink substantially; Military_operation and Attack dominate, and prior source checks found some flawed role assignments. This is a feasibility warning, not a benchmark size claim.
- A nonrandom first-three-per-type read of Conquering, Military_operation, Attack and Control candidates found useful examples alongside serious confounds: a `defeat` by disqualification with multiple opponents; an event annotated as `take` despite an explicit failed attempt; broad `Military_operation` mentions mixing wars and localized actions; and `Control` pairs with mismatched role semantics. This spot check cannot estimate an acceptance rate, but it confirms why four distinct strings alone do not produce valid gold.
- An additional permitted source or independently annotated Wikipedia sample may be needed. A release must respect each source's text rights. No official hidden TEST should be opened.
- The existing user authorization covered one bounded decisive experiment and ended at its no-witness stop. This memo authorizes **no new Qwen, hosted API, data download, or training run**.

## Hard gates, in order

**Gate 1 — novelty:** Compare the full methods and appendices of Mixing Mechanisms and Dai 2026 with the exact proposed natural-event intervention. Kill the paper bet if the same syntax/discourse contrast and predictive endpoint are already tested.

**Gate 2 — data:** Blindly adjudicate a small source-only sample across multiple event types. Predeclare a minimum usable document count and at least one external-validity stratum before model outputs; if the corpus cannot support it, change source or stop. Do not call annotation auto-filters ground truth.

**Gate 3 — faithful baseline:** Reproduce the published three-route predictor on its original task. If not, no claimed natural-text failure.

**Gate 4 — discriminating prediction:** Show at least one naturally valid, held-out condition where the frozen three-route model and a specified syntax/discourse extension predict meaningfully different outputs. If a strong compound-key rival matches the extension, describe the equivalence class and keep the claim predictive, not uniquely mechanistic.

**Gate 5 — one preregistered run:** Only after 1–4 pass, make one bounded approved local-model run. Publish all outcomes; abandon a mechanism claim if the baseline predicts equally well or the extension fails independently.

## Course and paper decisions

For the 15-credit course, the work can report the existing experiments, the no-witness result and the preparation/validation of this generalization test. A paper requires the extra held-out predictive contribution above. We should ask the supervisor early whether that scope meets the course rubric; no acceptance or publication is guaranteed.
