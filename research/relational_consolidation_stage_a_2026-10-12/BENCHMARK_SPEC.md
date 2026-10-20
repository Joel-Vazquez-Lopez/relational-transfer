# Relational-generalization ladder: design contract, not an implemented dataset

## Latent semantics

Represent an event as `(world_id, event_id, predicate_id, actor_id, other_id, role_type, truth_source)`. `role_type` must say whether `other` is a semantic patient, recipient, experiencer, or merely another participant. Begin with deliberately **reversible transitive actions** for which actor/patient labels are unambiguous. A surface realizer produces `(sentence, construction_id, mention_order, token_spans)` from the same event. The realizer never chooses gold roles from sentence position. An event with actor A and patient B is **different** from an event with actor B and patient A, even if they use the same verb and people.

There are two task types, and they must never be pooled as one transfer score:

1. **Role interpretation:** given a new *sentence* for a withheld participant combination, identify the roles using a learned linguistic rule. The sentence itself states the event; the model need not infer that the event happened from a relation network.
2. **Factual relational inference:** infer an unobserved fact from observed facts plus an explicit rule, e.g. `parent(A,B) ∧ parent(B,C) ⇒ grandparent(A,C)`. Without such a rule, an unobserved `chased(A,B)` edge is not entailed by other chase edges. This is a distinct L4 track.

The graph in the original controlled task records **which participants co-occurred in training examples**, not an objective knowledge graph that entails missing facts. No S3 generator may create labeled examples for the focal withheld pair in the training or consolidation phase merely because it knows the hidden evaluation event.

## Ladder and minimal pilot

| Level | Input / output | Generalization isolated | First pilot? |
| --- | --- | --- | --- |
| L0 symbolic | `ACT[fox,dog]` and a neutral scene/role query | New pair under a learned role schema and controlled co-occurrence network | Yes |
| L1 canonical text | “The fox chased the dog” → actor fox, patient dog | Surface lexicalization of L0 | Yes |
| L2 transformations | Passive equivalent “The dog was chased by the fox”; grammatical patient-fronted cleft; true reversal “The dog chased the fox” | Meaning-preserving invariance **and** meaning-changing equivariance; patient-first without a `by` cue | Yes, if generator validity passes |
| L3 lexical transfer | Held-out verb–noun combinations, new entities, verified paraphrases | Composition across lexical identities and constructions | Later; separate split for new verbs versus new entities |
| L4 composition | Two-event reference and explicit rule-based inference | Cross-event reasoning and genuinely new entailed relations | Later; needs a new rule engine and proof audit |

For each event, score a **joint quartet**: canonical, passive equivalent, cleft equivalent, and true actor/patient reversal. For the first three, the actor answer stays the same; for the reversal, it changes. Evaluate actor **and** patient. Agreement with a wrong answer receives no credit. Preserve item IDs linking the forms and score an event correct only if all required answers are right.

## Independent design axes and balancing

- Relation network: generate multiple graph families; match entity degree, number of edges, training-example count, predicate frequencies, and rendering before varying a chosen structural contrast. Record all changed graph statistics rather than naming a contrast by one proxy.
- Pair split: training, development, confirmation pairs are disjoint as unordered participant pairs, so reversing the same pair cannot leak across partitions. Both directions of a withheld pair stay out of training and consolidation.
- Construction split: when testing an unseen form, all lexical renderings of that form are held out for the relevant test condition. If the model saw the same syntax with other pairs, label the split “unseen pair in familiar construction,” not “unseen construction.”
- Lexical split: separate unseen *combination* from unseen *verb* or *entity*. New token embeddings make the latter a different difficulty.
- Actor order: balance actor-first and actor-second within each construction where grammar permits; do not let a passive marker determine every actor-second case. Patient-fronted clefts provide a second-order cue without `by`.
- Answer format: candidate A/B in both orders; candidate likelihood with length/prior controls; exact generated actor and patient spans; one or two predeclared wordings. Keep candidate scoring and generation separate.
- Supervision: ordinary answer target, ordered-role auxiliary target, participant-only auxiliary target, and any S2/S3 objective receive explicit information-budget labels.
- Event count: choose exact counts only after a source-only feasibility enumeration; freeze in a manifest before model training. A starting engineering target is **8–12 graph families, 12–20 entities per family, 2–4 verbs, and at least 80–120 genuinely withheld pair-events per family**. These are design targets, not claimed feasible counts; a generator test must prove them or revise the target *before* results are seen.

## Deterministic partitions and leakage rules

Use a named PRNG with fixed seed; serialize generation config, graph adjacency, event table, split assignment, realization grammar version, and SHA-256 digests. Assign graph families to development versus confirmation before training. Assign unordered pair keys, predicate keys and construction templates to partitions according to the intended split. Hash normalized full sentences, event tuples, and `(unordered_pair, predicate)` keys to catch exact and role-reversed overlap. For any generated consolidation example, log its proof/source event IDs and confirm all are in the **training** partition. Test items and their latent tuples must be inaccessible to the consolidation generator. Document lexical paraphrase checks separately; exact-string nonoverlap alone is insufficient.

For L4, a deterministic rule engine must emit a proof trace whose premises are all training facts and whose conclusion is evaluated by an independent rule checker. Do not label a merely plausible edge as entailed. If there are no logically entailed new facts, S3 cannot claim to create new relational knowledge in that world.

## Pre-training validity gate

Before any model run, execute generator tests: uniqueness, split nonoverlap, graph matching tolerances, balanced actor positions, balanced target names, active/passive semantic equivalence, reversal answer change, and S3 proof closure. Human-read a blinded stratified sample of L2 realizations for grammar and role correctness. Run first-participant, last-participant, grammatical-subject, `by`-phrase, lexical-memorization, and majority-answer heuristics. If any intended hard stratum is near solved by a shortcut, revise the **generator** before opening model results. Freeze a short benchmark card with counts, all heuristic scores and known limits. This gate uses synthetic data only and does not open the reserved natural-language challenge.
