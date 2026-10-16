# Independent falsification worksheet: selector-to-answer rescue (16 Oct 2026)

**Status:** proposed and independently scoped; no model run, no replay, no evidence of pretrained LLM mechanism. Branch based on Brain-O's research branch without modifying it. Scientific owner of follow-up experiment: Brain-O; reviewer: independent development investigator.

## Evidence anchor
Brain-O E045–E050 reports 1196/1200 and 1200/1200 selected event indices versus 387/1200 and 395/1200 correct answers at 750 updates (paired margin models), with 0/200 six-question complete worlds. A paired-CE seed also gets 1200/1200 event selection and 397/1200 answers. Whole-token query-state patch can switch event head but rarely answer. This does not prove a missing or inaccessible semantic binding: event-index labels were privileged during training, and event-index head and answer head are different readouts.

## Pre-result decision matrix
| Observation on held-out worlds | Most compatible hypothesis | Still not established |
|---|---|---|
| Bridge improves answer accuracy above matched sham and matched direct-head retrain, generalizes to direction swaps, and route ablation removes gain | Readout/access bottleneck | Unique semantic binding, natural-language validity |
| Bridge and direct-head retrain improve equally | Optimization/readout training limitation | Representation uniquely encodes directional roles |
| Bridge works only on familiar verb-tag/name/template configurations | Lexical/position memorization | Compositional generalization |
| Event index accurate but all answer heads fail; oracle index also fails | Possible absent directional participant binding, OR insufficient decoder/supervision | Information-theoretic absence of binding |
| Oracle participant answer succeeds but oracle event index does not | Event identity alone insufficient | Which layer lacks participant representation |
| Causal selector perturbation changes event index but not answers, even after rescue | Unused selector route or invalid patch | Functional use of selected representation |
| Causal selector perturbation changes answer with correct donor-target contrast, sham null | Evidence of functional route dependence | Abstract relational mechanism without lexical/syntax controls |

## Locked comparison requirements before any inference
1. Split by **world** and heldout structural family; never treat six questions/world as independent. Include unseen names, crossed role/name order, unseen verb×tag conjunctions, 3/4/5-event structures, passive/active surface alternations.
2. Exact checkpoint ID, parameter counts, optimizer states, seeds, update counts, batches, training examples, trainable parameters and total FLOP proxy per arm. If matched FLOPs are impossible, mark as unequal and do not claim architecture-matched advantage.
3. Primary endpoints: entire-world correctness and joint actor+patient accuracy for a target event. Secondary: event-index accuracy, marginal roles, conditional P(correct answer | correct event index), and invalid-output rate.
4. Arm A: frozen selector + learned label-free answer bridge trained with answer gold only. Arm B: equal-capacity bridge from nonselective or randomly permuted representation. Arm C: equal-capacity answer head retrained from original final state, same answer labels and updates. Arm D: oracle event index (privileged diagnostic; never included in normal inference result). Arm E: syntax-aware and lexical copying heuristics.
5. Causal necessity: compare source and donor worlds with different gold role answers, matching requested role, crossing lexical/position cues; transplant selected-state versus sham and measure *answer* effect and *event-index* effect separately. Explicitly report out-of-distribution whole-state-patch limitations.
6. Negative-control stop: if selector fails heldout family, if equal-capacity sham matches bridge, or if a direction-aware syntax copier matches all gains, report as negative for mechanism claims.

## Reviewer sign-off grid
- Brain-O: owns design implementation; ACK/changes requested on capacity and supervision parity.
- Comput: adversarial shortcut and donor-target crossing review.
- Sentry: leakage, data split, provenance and statistical unit review.
- Armaggedon: priority relative to real-data QA-SRL track and permission for any future run.

No human annotation, external text, paid inference, or merged code required for this worksheet.
