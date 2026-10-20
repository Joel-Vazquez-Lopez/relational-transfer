# Goal-first decomposition: first controlled feasibility tests

**13 October 2026.** This separate research branch tests a narrow instance of the proposed “inverted chain of thought”: generate a goal-derived operation plan before executing it. It does **not** test general task decomposition or an LLM. The local model is a small scratch-trained causal Transformer on an engineered two-hop event language.

**Implementation correction discovered after these runs:** the frozen model code assumed padding token ID `0`, while its sorted vocabulary assigned `0` to a different special token. The attention mask therefore hid that special token and did not hide actual padding. The raw scores below are preserved as **bug-affected diagnostics**, not evidence for or against goal-first training. No old checkpoint is silently repaired or reinterpreted; a fresh corrected experiment is required.

## Design and audit

Each example has four facts, active or passive, and a nested question. Two facts form a two-step actor/patient path; two are distractors. A step may follow a known actor to a patient (`F`) or a known patient to an actor (`R`). Training and in-domain testing use `FF` and `RR`; the compositional challenge uses `FR` and `RF`. Every primitive direction and verb appears in training, but neither mixed two-step dependency does. There are 4,000 train, 400 development, 400 in-domain test, and 400 mixed test examples. Exact sentence/question strings do not repeat across splits. The answer never appears in the question, and first/last fact-position heuristics score about 11–13% on test; see `DATA_AUDIT.json`. These checks do not rule out every statistical shortcut.

A separate surface-only symbolic parser, given just the printed facts and question, recovers the middle entity and answer on **400/400** in-domain and **400/400** mixed items (`symbolic.py`, `SYMBOLIC_AUDIT.json`). The task is therefore solvable from its visible input. This rule baseline exploits the benchmark's restricted grammar; it is an upper bound on this synthetic setting, not a language-model comparison.

Three arms share train examples, optimizer update order, seed, and architecture: **direct answer**, **forward trace** (execute each step and print its value), and **goal-first** (print the second operation as a goal dependency, then the first, then execute). The trace arms both receive operation, intermediate entity, and answer supervision. They differ in output order, markers, and target-token count, so the comparison is a *feasibility screen*, not a matched causal estimate. The direct arm receives less supervision. No natural or sealed project data entered this pilot.

`PROTOCOL.json` and the generated data were fixed before training. The first training attempt stopped after epoch 1 because a progress-print expression used an undefined variable; it saved no checkpoint or evaluation. `PROTOCOL_initial.json` preserves the original. The amended protocol records the variable-name repair before restarting the planned training.

## Frozen small-model pilot

The original model has two layers, width 64, and 378 updates per arm (six epochs). Its frozen positive screen required goal-first to beat forward trace by at least 20/400 mixed answers while preserving in-domain accuracy, and to produce valid mixed plans for at least 300/400. It **failed**.

| Arm | In-domain exact answer | Mixed exact answer | Mixed correct operation pair | Mixed full plan valid |
|---|---:|---:|---:|---:|
| Direct | 61/400 | 77/400 | — | — |
| Forward trace | 56/400 | 48/400 | 0/400 | 0/400 |
| Goal-first | 72/400 | 41/400 | 0/400 | 0/400 |

Both trace models printed the correct operation pair on every in-domain test item, but chose the correct answer on fewer than 20%. The task was not sufficiently learned to interpret a mixed-composition difference.

## Adaptive capacity check

After inspecting the failed pilot, I froze a **separate, explicitly adaptive** capacity check: four layers, width 128, 1,512 updates (24 epochs), same items, starts, objectives, and output protocols. Because it reuses already examined test items, these numbers are exploratory and cannot be presented as an independent confirmation. `CAPACITY_PROTOCOL.json` states a diagnostic gate of at least 280/400 in-domain answers for the trace arms. That gate also **failed**, even though training loss became very low.

| Arm | In-domain exact answer | Mixed exact answer | Mixed correct operation pair | Mixed full plan valid |
|---|---:|---:|---:|---:|
| Direct | 58/400 | 62/400 | — | — |
| Forward trace | 89/400 | 109/400 | 0/400 | 0/400 |
| Goal-first | 129/400 | 72/400 | 64/400 | 3/400 |

In-domain, both trace models again printed the correct operation pair on **400/400** items; goal-first still got only **129/400** final answers. On mixed examples, the forward model printed no fully correct operation pairs, while goal-first printed 64/400 correct pairs and only 3/400 full plans. Thus the problem includes **both operation recombination and entity binding/execution**. Goal-first improved in-domain answers over forward trace in this adaptive setting but did **worse** on the held-out mixed operation order. This is evidence **against this simple output-order training formulation**, not a verdict on the broader idea of learning to derive procedures.

Target-token totals are unequal. At 1,512 updates, direct used **288,000** supervised target tokens, forward **1,152,000**, and goal-first **1,248,000**. Any positive difference would have required token- and output-format-matched controls before a causal training claim.

## Adaptive textual plan intervention

I then froze a no-training follow-up in `INTERVENTION_PROTOCOL.json`. For the capacity-check goal-first model, I prefixed the assistant through `execute` with its own plan, the correct operation plan, or the correct plan with both directions flipped. The supplied plans contain **no intermediate or answer entity**. On in-domain items, prefilling its own plan reproduced the original answer **400/400** times. On mixed items, it reproduced **345/400**; that imperfect replay limits own-plan comparisons. The correct and flipped plans share the same output format and length.

| Condition | In-domain middle correct | In-domain answer correct | Mixed middle correct | Mixed answer correct |
|---|---:|---:|---:|---:|
| Own printed plan | 382/400 | 129/400 | 242/400 | 65/400 |
| Correct operation plan | 382/400 | 129/400 | 385/400 | 86/400 |
| Both operation directions flipped | 380/400 | 51/400 | 384/400 | 78/400 |

Supplying the correct mixed plan strongly improves the **first intermediate entity** but leaves the **second entity/final answer** poor. Gold-plan answers rescue 41 own-plan failures and harm 20 own-plan successes, for a net +21/400; it is an oracle-plan condition, not end-to-end transfer. On the mixed set, the answer changes in 157/400 paired correct-versus-flipped plan interventions. Yet first intermediate correctness barely changes when the plan is flipped, suggesting the model often recovers that first step from the question rather than obeying the printed direction. The second printed entity is correct in only **88/400** gold-plan mixed cases. These are behavioral interventions on visible text, not evidence of a unique internal computation.

## Scientific judgment and next gate

The bug-affected tests have **not** produced a positive result for the stronger novel-contribution claim. They reveal a possible planning/binding failure mode under this implementation, but cannot establish that “print a backward plan before answering” is insufficient in a correctly masked model. Increasing model size and updates once did not clear the predeclared learning gate **in these invalid runs**.

The next experiment should change the learning problem, not rerun more seeds of this failed formulation. A defensible direction is to separate **procedure construction** from **execution** during training and evaluate each with a fresh locked test. In particular, teach operation choices for clauses independently, then test unseen two-step combinations; separately verify whether an executor can use a correct plan to bind both entities. A model that only succeeds with oracle plans or a deterministic executor has learned less than the proposed general decomposition ability. Any larger claim also needs multiple task structures and stronger matched baselines.

`ANALYSIS.json` and the raw `EVAL_*.json`, `CAP_EVAL_*.json`, and `INTERVENTION_*.json` preserve item-level outputs. Training records include update counts, target tokens, losses, elapsed time, and checkpoint hashes. The main project experiments and proposal were not modified.
