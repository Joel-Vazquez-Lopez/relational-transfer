# Frozen execution specification: role-demand transfer

**Authorization:** Joel reviewed the proposed design and said “go for it mate.” This specification completes that one bounded experiment. It supersedes only the proposal's unspecified execution choices; the proposal, older studies and failed gates remain preserved. No new model has yet been initialized for this study when this document is frozen.

## Scope

One controlled synthetic acquisition experiment. Fixed source/target message slots are ordered arguments, not demonstrated natural-language semantic roles. One abstract six-identity graph is used, with identity relabelings and independent initializations. No claim of nonisomorphic-world replication, pretrained-LLM generality or novelty. No adaptive second task or rescue experiment.

## Six arms and exact data controls

Canonical identities H={0,1,2}, G={3,4,5}; all48 observed ordered meanings have at least one G endpoint. Twelve H–H meanings are absent from every training query, candidate and foil. Focal identity pair1/2.

For each G anchor, the other five identities form a five-cycle. The complementary five-cycle has the same degree at every vertex. One contains the focal1–2 edge; the other does not. Pair each edge with both event values to form a four-candidate source-comparison block, or its target-comparison counterpart. The degree identity makes each positive meaning occur equally often across treatments.

Use two replicas of this block family. In the four factorial arms, focal source/target cycle appears in zero/one replica as specified by the two binary factors. Source-double or target-double uses that role's focal cycle in both replicas. The combined focal comparison count in both equals either double-dose control. Off slots always use the complementary cycle, preserving total comparisons. There are no missing-example controls.

Add the same twelve reversal/event blocks per replica to every arm. They force an ordered-argument distinction on familiar data; unordered query matching has ceiling6/7, not100%. Every block is rendered once per nuisance setting and reused for all four selected queries and four cyclic candidate orders. Candidate-only and query-only ceilings are25%; dropping query event gives50%.

Each arm has84 blocks and96,768 indexed training cases:18 geometry settings, four named-distractor settings, four candidate rotations and four selected queries per block. Geometry is rendered with the existing source-balanced B-D15 function, keyed by meaning and nuisance setting, not treatment, block or selected answer. Each meaning's actual positive scene and candidate appearance frequencies are identical across arms. Named distractors cycle through all four legal identities. All six object permutations are exactly represented. Query, answer label and positive-scene indices match at every training array position, enabling coupled minibatches.

The preflight checks legality, pairwise comparison spectra, zero direct off-factor focal comparisons, positive and candidate marginals, answer balance, all720 identity relabelings, three complete encoded layouts, and exact finite Bayes ceilings under source/target query coarsening and full-input coarsening. Same-input/incompatible-label witnesses persist for all nuisance realizations because the unchanged four-candidate tensor is reused for each selected query. No scientific interpretation depends on an approximate symbolic collision.

## Model and optimization

Reuse the existing unmodified Block/Encoder/Agents classes: width48, four heads, two message blocks and two candidate blocks. The unused speaker/emit modules remain frozen. Input scenes remain3×14. Oracle query is[source,target,6+event,PAD]. **PAD=8 has its own vocabulary entry**, making vocabulary9; this removes the historical identity0/PAD alias in every arm. This is a declared common interface change, not a treatment or a replication of the old exact configuration.

CPU only, one thread, deterministic algorithms. AdamW learning rate.0008 and its installed default betas/epsilon/weight decay. Batch128, gradient norm clip1,3,000 updates, final model as primary. No early stopping from performance, no hyperparameter search, no checkpoint selection. Coupled initial weights and batch indices across all six arms per seed; stable cyclic arm order determined by seed reduces fixed execution-order bias.

One development initialization seed7284101, six arms. Only if the complete development gate passes, run eight fresh confirmation seeds7284201–7284208, all six arms. Seed-derived identity maps relabel the same graph. Maximum54 trained models plus one untrained technical reference. Historical38.7seconds/3,000-update run is a rough feasibility reference, not a promised runtime.

## Resources

No paid APIs, model downloads, GPU jobs or large pretrained weights. CPU runtime cap5,400seconds for this collection, own peak resident memory cap1.5GiB, run-artifact cap1GiB, minimum free disk reserve5GiB. Check before each model, during training and at checkpoint writes. Resource failure stops collection and preserves partial work. No deletion of old evidence. No reuse of protected seeds1704–1706 or reserved examples/checkpoints. Seeds here are new synthetic generation/initialization choices, not protected material.

## Measurements

Final held panel: for each event, candidates[(1,0),(2,0),(0,1),(0,2)] using unseen H–H relations,24 fresh geometries, four distractor ranks and four candidate rotations. All four queries are asked. Source-role success requires correct answers to both1→0 and2→0; target-role success requires both0→1 and0→2. Primary is the all-case fraction of complete pairs per role, then averaged within model. Also retain individual answers and all-four completeness. Never treat target-after accuracy alone as switching.

Familiar gate: independent rendering of all observed reversal/event panels,1,152 answers per model. Required-comparison gate: fresh nuisance rendering of the arm's training comparison contexts, retaining all explicitly focal source/target cases plus a fixed evenly spaced ordinary sample. Each explicit trained-role contrast must reach95%. No conditioning/selection of the primary held cases on native competence.

Technical reference uses observed/familiar scenes only. Direct versus traced forward scores, identity input copies and inverse-mapped candidate permutation must agree within1e−5, with identical argmax. No model training if this fails. Final identity and object-order controls are retained per model; object-order agreement is a reported boundary, not a way to discard unfavorable primary cases.

Checkpoints0,100,500,1500,3000. Save full model/optimizer/CPU and NumPy RNG states. Save every held/familiar score vector at these steps and fresh-context final scores. Capture all query/candidate token states at embedding and both block outputs for eight preselected held cases: geometry0,distractor0,rotation0,all four queries,both events. Developmental traces are secondary; no fitted direction, PCA selection or mechanistic site selection changes primary outcomes.

## Development stopping rule

After all six development models, require:

- Familiar accuracy≥95% in every arm.
- Fresh explicitly required comparisons≥95% for every trained role in every relevant arm.
- Identity controls maximum score error≤1e−5.
- Neither-arm mean source/target paired accuracy≤80%, leaving improvement headroom.
- At least one treatment improves that mean by≥10percentage points over neither.

If any fails, **stop; no confirmation models and no adaptive repair**. Report all development outcomes and exact failures. This is a valid feasibility stop, not a confirmed causal absence claim. Development results never enter confirmation statistics.

## Confirmation predictions and analysis

Unit: paired initialization/identity relabeling, n=8. Repeated geometries, rotations, object positions and candidate labels are not independent units.

Primary selectivity contrast per seed:

`D = .5 × [(Y_source(source-only) − Y_target(source-only)) + (Y_target(target-only) − Y_source(target-only))]`.

Role-selective pattern requires meanD≥.15, one-sided exact sign-flip p≤.05, and source-trained/source-tested and target-trained/target-tested gains each≥.15 versus neither. This is a bounded transfer interaction, not a dedicated neural circuit claim.

Broad-transfer-compatible pattern requires all four single-role-training × tested-role gains≥.15 and the90% seed-bootstrap interval forD entirely within[−.05,.05]. A nonsignificantD is not evidence of equivalence.

Both-demand-compatible pattern requires: each both-arm role gain≥.15; all source-only,target-only,source-double,target-double role-gain90% bootstrap intervals inside[−.05,.05]; both-arm mean paired accuracy exceeds each double-dose mean by≥.15; both one-sided sign-flip comparisons pass Holm correction at.05. This avoids calling additive role-specific gains “joint necessity.” Even success remains compatible with additional optimization explanations.

Use10,000 seed-level bootstrap draws, RNG7284991, and all2^8 sign flips. Signed-exchangeability under the null is an assumption, not guaranteed by repeated scene counts. Primary selectivity is one test; the two secondary dose tests receive Holm correction. All other values are descriptive. No universal or topology-level inference.

If any confirmation model misses familiar/required learning or identity gates, retain every result and classify the mechanistic comparison INCONCLUSIVE_TASK_LEARNING. Otherwise report the matching restricted pattern, MIXED_PATTERNS, or NO_SUPPORT_FOR_FROZEN_PROTOTYPES. Do not force a winner or change thresholds after outcomes. A ceiling or null is reported without another adaptive study.

## Preservation

Freeze file hashes, preflight output, seeds, installed runtime versions, thresholds and limits before technical model initialization. Publish that lock on the existing research branch before inference. Save scripts, compact training inputs, full evaluation inputs, raw scores, training traces, checkpoints, model-level endpoints, gate decisions, failures and compute accounting. Keep all old audit and experimental decisions intact. Link Issue35 and the historical Issue33 workstream. Stop after reporting this bounded study.
