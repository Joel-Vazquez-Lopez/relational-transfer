# Brain-O E056 — Readout gradients and forward-graph correction
**16 October 2026 · Executed synthetic checkpoint audit, no new training.**

## Motivation
Earlier E043–055 findings emphasized that the network could achieve near-perfect *auxiliary event-index classification* while answering ~33% of the compound event/role questions. E056 checks a crucial architectural alternative: **the event head is a separate readout and does not feed its predicted index to the answer head.** The native forward graph returns `ans(norm(z[:,24]))` and `event(norm(z[:,23]))`, where `z` is a causal three-block Transformer. Under causal masking, the event readout at token 23 is mathematically independent of the later role token at position 24, even at random initialization. Zero sensitivity along that path is a design constraint, not a learned mechanism. Earlier native intermediate activation patches still matter as tests of shared-state influence; do not equate them with passing an event-head output value into the answer.

## Executed gradient assay
On **96 new open-development generated worlds × 6 questions each**, frozen E045–050 checkpoints were compared with three random initialized 3-layer backbones. For each example, compute gradients of an answer **gold-vs-other-event-name logit margin**, and of an event **correct-slot vs other-slots logit margin**, with respect to *input embeddings* for query verb at position 22, context tag at 23, and requested role at 24. Within each head, calculate the fraction of mean position norms attributable to each of the three query tokens; do not compare absolute magnitudes across two differently scaled heads.

| Model | Event correct /576 | Answer correct /576 | Role share of answer query-gradient | Verb+tag share of answer query-gradient | Verb share of event query-gradient |
|---|---:|---:|---:|---:|---:|
| Margin 1040, 750 steps | 576 | 188 | 97.63% | 2.37% | 53.17% |
| Margin 1041, 750 | 576 | 190 | 96.66% | 3.34% | 42.47% |
| Margin 1040, 500 | 562 | 186 | 96.79% | 3.21% | 56.90% |
| Margin 1041, 500 | 394 | 182 | 97.16% | 2.84% | 11.52% |
| Standard paired CE 1041, 750 | 576 | 191 | 95.75% | 4.25% | 39.93% |
| Random 3 seeds, mean | ~185 | ~26 | ~91.09% | ~8.91% | ~4.68% |

**Supported:** event-head query gradients depend on two query keys in the successful trained selectors, while answer-head gradients overwhelmingly emphasize the role token; some of the role preference is intrinsic to last-token residual/readout geometry, and its relative dominance grows in trained examples. Previous E044 finite-query swaps support weak final-answer dependence on event keys. **Not supported:** uniquely semantic cached relation, mechanistic information bottleneck theorem, natural-language transfer, or a diagnosis of where *native* complete event information appears. Gradients are local and uncalibrated as semantic causal effects.

## Cross-team implication
- Compass I17/18 finds instruction changes matter more than matched operand changes across synthetic program tasks but weak whole-world composition and strong two-step shortcuts remain. This is complementary, not an identical proven mechanism.
- Brain-X BX006, Comput R020, Nova QA-SRL Gold adapter: initial authentic-language route now uses original published human **predicate question→answer spans**, including multiple answers, **not fabricated universal actor/patient gold**. Mira's EWT 25+24 candidate group effort is optional pending rights and real human adjudication, not a blocker for published human annotation.
- Astra's output-field-order × source-position diagnostic is different from our query-readout gradients.

## Reproducibility
Self-contained conversation artifact: `BRAIN_O_E056_READOUT_GRADIENT_AUDIT_2026-10-16.zip`, SHA-256 `1ab38f88d343188a871f9c49c3fde56e51fc1bdec23532d804b4416f965d6c49`, 12 payload entries + SHA manifest, ZIP CRC and each payload hash verified. Includes five trusted `.pt` checkpoints, the exact tokenizer/model classes, `gradient_audit.py`, raw JSON/STDOUT and methodological report. After safe review of the supplied code/weights, `python gradient_audit.py` regenerates metrics. The weights are NOT pushed to GitHub and should not be redistributed casually.

**Scientific priority next:** competence-matched generic neural lookup/copying rival and model-native attention-path interventions; in parallel support QA-SRL Gold two-family pretrained development evaluation under source/license and format gates. No new inference, no sealed set, no merges, no other investigator branch edits. **No EUREKA.**
