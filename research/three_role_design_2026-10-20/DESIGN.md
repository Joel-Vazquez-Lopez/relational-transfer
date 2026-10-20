# Strengthen the experimental task before expanding the model system

20 October 2026. **Design-stage work only.** The executable certificate enumerates stimuli and rival predictions; it does not run an LLM, discover a model mechanism, estimate internal information or produce an empirical significance result. All old failed gates remain unchanged.

## What “not significant enough” could mean

Our present system has several distinct limitations:

1. **Too little independent relational variation.** In the binary task, question plus correct answer identifies the full person-role assignment. This is a logical identification limitation; collecting more examples does not remove it.
2. **An unreliable answer interface.** The extended response-code factorial produced many negative-category defaults. More complex prompts can measure instruction handling and calibration rather than the relationship of interest.
3. **An overly narrow intervention target for the hypothesis.** A period route can be useful while its tested K/V donations do not carry a selectively transferable role assignment. Distributed storage and another functional route remain possible; the measured null stays a null.
4. **Limited independent replication.** Four authored families or one checkpoint per family cannot establish a broad law. Extra layers and repeated queries do not increase the number of independent linguistic samples.

These problems need different remedies. A larger model might improve native task accuracy, but it does not fix an aliased design or a nonselective intervention. A richer design is warranted only where its extra variation gives different predictions to specified accounts.

The earlier Atlas already included Agent/Theme/Goal variation and an unqueried membership target, and the component-reuse pilot already considered multiple questions. This proposal is not the first multi-role idea in this project. Its specific changes are full six-way assignment support for every queried role, candidate-free name responses, explicit cross-form rival predictions, and a distributed-token candidate alongside the previously emphasized source boundary. Native competence and causal selectivity for this combination are untested; no improvement in model behavior has yet been demonstrated.

## Correction to my previous example

“Alice handed the book to Bob” supplies two exchangeable people and a fixed object. Asking about the book is a useful unchanged-answer control. It does **not** stop the answer to either person question from identifying the entire two-person assignment.

Adding a third person also fails if we use only three cyclic assignments. With three roles and three cyclic orders, one role's answer still identifies the complete assignment. The design needs **all six permutations**, so two assignments remain possible after fixing any queried role and answer.

This was checked by exact enumeration, not assumed. The certificate compares four designs:

| Assignment design | Assignments consistent with each person-question/answer | Interpretation |
|---|---:|---|
| Two interchangeable people | 1 | Full assignment is answer-identifiable |
| Two people and fixed object | 1 on either person question; 2 on the fixed-object question | The extra object question is a negative control |
| Three people, cyclic assignments only | 1 | The extra person did not remove aliasing |
| Three people, all six assignments | 2 for every role-question/answer | An unqueried person-role distinction remains |

Under the explicitly uniform enumerated design, `H_design(R | Q, Y, lexical_set, template)` is zero bits for the binary and cyclic designs and one bit for the full six-permutation design. R is the full role assignment, Q the queried role and Y its correct answer. The conditioning context fixes the lexical set and template, not the complete source sentence. This is an exact property of the stimulus distribution; it is **not** an estimated quantity computed by Qwen or Llama.

## Smallest useful stronger task

Use one directional construction initially:

> Agnes sent Basil to Celia.

Three grammatical roles: sender, person sent, destination person. Keep these labels rather than claiming that all three are universally Agent/Patient/Recipient thematic roles.

Use the same three questions across all cells:

- Who sent someone to another person?
- Who was sent to another person?
- Who was someone sent to?

Cross all six name assignments with three surface forms: active, passive, and goal-fronted. Their mention orders are respectively sender–theme–goal, theme–goal–sender, and goal–sender–theme. This is 54 prompts per lexical family. The generator proposes two development and two evaluation name triples: **216 candidate prompts and 72 source sentences**. It provides AI-only exact-template labels; it has not established natural-corpus validity, annotation reliability or native model competence. New lexical families with the same verb are not a new-verb or new-construction generalization claim.

The primary answer format is a copied full name, using the already established candidate-free system instruction. Do not introduce Yes/No, A/B legends or a fitted threshold into the primary test. Preserve unconstrained greedy full responses, invalid outputs and full-name sequence scores separately. Multi-token names require full-name scoring; an initial token is insufficient.

The first scientific question becomes:

**Can a fixed source-side intervention selectively transfer the two unqueried role assignments while preserving the queried answer, and can the same edited source support all three future questions across surface forms?**

Here “queried answer” refers to the matched question used to construct an invariant-answer pair. The source intervention itself is computed before any future question. The three recipient continuations must all reuse that same donor capture and the same intervention map. There must not be a separate donor encoding chosen for each requested answer.

## A discriminating example

Recipient role assignment: Agnes is sender, Basil is sent, Celia is destination.

Donor role assignment: Agnes is sender, Celia is sent, Basil is destination.

Both answer the sender question with Agnes. Their other two answers differ. A coherent source-message transfer predicts the table `[Agnes, Celia, Basil]`: two actual changes and one preserved answer. A fixed code containing only the invariant answer Agnes cannot supply that table. A complete precomputed future-answer table can, and remains a competing implementation.

Crossing donor/recipient surface forms is essential. A donor mention-order list interpreted with the recipient's grammatical role-to-slot mapping gives different predictions from a donor role table. Same-form trials alone cannot separate those two declared accounts.

`RIVAL_PREDICTIONS.csv` contains **486 symbolic answer predictions**, grouped into 162 three-question units. They are not model answers or independent samples. In the enumerated set:

- Role-table and recipient-grammar mention-order accounts disagree on all 324 cross-form answers and agree on all 162 same-form answers.
- A fixed one-answer account disagrees with the role table on 324 of 486 answers.
- A syntax-aware role parser and a complete future-answer table agree with the role table everywhere.

These are restricted algorithmic predictions, not assumptions about what a vector patch physically does. If an intervention transfers donor syntax information along with mention order, a richer positional account may behave like the role table. The experiment must not claim to have defeated that stronger account merely by beating the declared recipient-grammar baseline.

## Claim boundary

The design can test a narrow **functional reuse** claim beyond a fixed single answer and a restricted mention-order table. It cannot identify a uniquely semantic algorithm, distinguish a complete answer table from a role table, or rule out a syntax-aware parser that implements the same function.

The strong unique-mechanism gate therefore remains **NO-GO**. The weaker operational question now has explicit separating predictions, which is progress in the assay rather than a positive model result. General causal-abstraction nonuniqueness is not solved by renaming the representation.

## Proposed instrumentation and execution sequence

This is a proposal requiring a source-specific runtime protocol before any causal execution. No model runner or ready-to-execute causal budget is implied by the generated prompts.

1. **Native competence:** collect the fixed candidate-free baseline panel, with exact complete-name generation and per-source three-question completeness. Record all 216 prompts per checkpoint, including failures. At most 432 baseline completions would cover the two existing checkpoints; actual forward/decoder limits require a tokenizer/runtime preflight and a separately frozen budget.
2. **Numerical controls:** identical batch/total execution shape, original prompt-end readout, wrapped/unwrapped equality, source-prefix invariance across future questions, full-name scoring, repeated native and identity controls. Never interpret shape-dependent prefix drift as future-question processing.
3. **Development-only site comparison:** a small prespecified grid of blocks 7, 14 and 21, considering the source-boundary residual and a joint region consisting of the final subtokens of all three person mentions. The distributed region is a new tested candidate, not a previously validated carrier. Align mentions by surface order, never by gold semantic role; otherwise the experimenter can insert the intended correspondence. Cross-form mappings and subtoken boundaries need explicit compatibility checks. Whole-source replacement, if used as a positive control, belongs to same-form technical validation and cannot count as selective role discovery.
4. **Controls at each candidate:** identity, neighboring-site, same-role cross-form donor and opposite-role donor; preserve donor/recipient condition identity and matching native baselines. A same-role syntax change is a substantive perturbation, not guaranteed inert. Quantify dose/compatibility and all control effects; no claim rests on a dramatic main patch alone.
5. **Freeze before evaluation:** select at most one candidate by a declared development rule, then freeze its maps, margins, control tolerances, question/response scoring and predictions. The rule and dose have not yet been specified sufficiently for model execution. Do not silently search more layers if no candidate qualifies. Evaluation families must remain outcome-unseen until that freeze.
6. **Report coherent transfer:** for each donor/recipient pair, all three future answers must match the prescribed donor assignment, with two real changes and the invariant answer preserved; evaluate both directions. The native prerequisite concerns the six corresponding source/question answers, not perfection across every condition in an entire 54-prompt family. This new unit follows the new claim; it does not retroactively relax any old study's gate. Publish complete admission counts and unfiltered graded results alongside the conditional causal analysis.

Four lexical families provide only an exploratory pilot. They cannot establish a significant population law, and the hundreds of correlated answers/sites must not be treated as independent linguistic observations. A successful pilot would justify a separately frozen replication; it would not warrant announcing a paper-level discovery.

## How this connects to the developmental Observatory

Only after selective functional transfer works should we expand the layer grid or separate attention and MLP ports. The quantities to plot would be the case-level effects on each of three answers and control answers, plus whole-table transfer outcomes across depth and token region. Models can then be compared by which intervention dependencies generalize at which stages. No identical hidden coordinates or cosine threshold is required.

A null at one source region means that region/intervention did not support the tested transfer. It does not prove that unqueried information is absent elsewhere. Failure to decode with a particular probe is likewise not information erasure. The current saved binary traces cannot supply the missing six-permutation variation retrospectively; new model data would be needed for this particular task.

## Prior work and novelty

[RAVEL](https://aclanthology.org/2024.acl-long.470/) already evaluates selective causal changes across related attributes, and [Mixing Mechanisms](https://arxiv.org/html/2510.06182v2) already uses multientity tasks and counterfactuals to separate retrieval accounts, including pointer versus answer. Three-role tasks, intervention tables and selective controls are not our inventions. Dai et al.'s released baseline-subtraction issue is separate; correcting it does not disprove their mechanism or validate ours.

The possible future contribution would have to be an independently replicated developmental dependency that predicts held-out linguistic interventions beyond these competitors. This document improves the logical task design and explicitly exposes its remaining equivalences. It claims neither model findings nor established novelty.

## Artifacts

- `certify_design.py`: stdlib-only enumeration, candidate stimulus generation and all checks.
- `DESIGN_CERTIFICATE.json`: conditioning definitions, complete ambiguity buckets, rival disagreement counts and limitations.
- `RIVAL_PREDICTIONS.csv`: all 486 symbolic predictions, including equivalent strong accounts.
- `CANDIDATE_INPUTS.json`: all 216 proposed prompts and AI-only labels; no model outputs.

Run `python3 research/three_role_design_2026-10-20/certify_design.py`. No model is loaded, no downloaded code is executed, and no API cost or large download is incurred. The present work stops at a concrete reviewable design, following the existing instruction to review substantial new inference first.
