# Frontier investigation: answer contracts and selective relational use

15 October 2026. Exploratory development only. This investigation generated 192 new outputs from untouched Qwen2.5-1.5B-Instruct and scored 48 pairs of gold participant continuations. No training, reserved evaluation, paid inference or remote job. All 24 source events were previously inspected development data. Historical evidence was read without modification. This is a compact research result, not a new confirmation handoff.

## 1. New experiment: a simpler interface explains much of the raw pipeline advantage

The frozen 2×2 manipulation crosses the original versus an explicit participant-span instruction with no assistant prefix versus `Answer:`. The natural sentences, original questions, target predicates, model and generation settings stay fixed. There are two grammatical target roles per event. The explicit instruction says not to turn the question into a statement; neither manipulation supplies the answer, requested role, participant candidates or role fields.

| Instruction / prefix | Subject /24 | Object /24 | Both /24 | Elementary /48 |
|---|---:|---:|---:|---:|
| Original / none | 4 | 15 | 2 | 19 |
| Original / Answer: | 10 | 16 | 5 | 26 |
| Explicit span / none | 10 | 16 | 5 | 26 |
| Explicit span / Answer: | 10 | 16 | 5 | 26 |
| Historical blind extraction + oracle routing (reference) | 15 | 16 | 11 | 31 |

The regenerated baseline reproduces all 48 old raw responses exactly. Each simple cue closes 7/12 of the original elementary-answer difference without a second reasoning pass. It closes only 3/9 of the event-joint difference. The reference still has a 5/48 elementary and 6/24 joint advantage over the clearer direct interface, with its oracle selection and different output/computation intact. Do not call this a matched-compute repair or explain the entire residual as representation quality.

Explicit instruction without prefix improves subject accuracy by 25pp, descriptive document-bootstrap interval [9.09,40.91]. Object change is +4.17pp, interval [-8.70,19.05]. The prespecified instruction×role contrast averaged across prefixes is +10.42pp, interval [-4.55,22.92]: **the role interaction is unresolved**. There are 19 source-document clusters; repeated roles and conditions remain paired. These intervals do not certify a population effect, and multiple exploratory contrasts are not multiplicity-controlled.

Combining the two cues adds no aggregate gain. Their instruction×prefix risk interaction is -14.58pp, descriptive interval [-24,-5]. This additional exploratory contrast is compatible with overlapping/redundant routes to fixing an interface problem. It does not identify two internal mechanisms. Their corrected item sets are not identical: five of the six newly correct subject answers are shared, but only 22/24 subject correctness decisions agree between single-cue conditions. Do not describe equal totals as identical behavior.

Under the original direct prompts, constrained likelihood comparison prefers the gold subject over the gold object in 15/24 cases by either summed or mean token log probability, including 11/20 failed free generations. Object contrast is 17/24 by sum or 18/24 by mean. This uses a gold two-candidate set, one representative string per role and no EOS probability. It is a diagnostic of relative continuation preference, **not autonomous accuracy, a calibrated representation probe, or evidence that the model selected the correct semantic role internally**.

The original direct-subject outputs include five nonexact responses containing a correct accepted subject but not the other accepted participant; three contain both, seven contain only the opposite participant and five contain neither, in addition to four exact answers. Containment is not semantic correctness. Boundary decisions, pronouns/coreference, complete statements and genuine participant errors coexist.

## 2. Historical causal panel: a positional account is already sufficient for most behavior

On the untouched-Qwen v2 panel, generic-actor questions select the last-mentioned name in 128/128 outputs. Verb-patient questions also select it in 128/128. Their active/passive accuracy reversal therefore needs no semantic-binding explanation at the behavioral level.

A simple first/last-mention rule selected by question type and voice predicts 368/384 archived outputs in leave-one-name-pair-out and leave-one-verb-out reanalysis. This model class was selected retrospectively; the result is not prospective confirmation. It shows a simpler rival can fit the old phenomenon extremely well. It does not prove which neural mechanism implements that behavior. Whole-state patch success cannot adjudicate this rival because all selected patches were active and role/position remained confounded.

## 3. Historical natural verifier: emitted fields do not fully mediate final answers

Read-only reanalysis of the 40-item natural verifier development set, with actor/patient labels still provisional:

| Noisy-verifier start | Draft-swap comparisons | Same emitted role tuple | Answer changes despite same tuple | Same *correct* tuple | Answer changes despite same correct tuple |
|---|---:|---:|---:|---:|---:|
| fresh1 | 160 | 124 | 16 | 98 | 13 |
| fresh3 | 160 | 133 | 25 | 104 | 18 |

The same-tuple answer changes occur in 13/40 and 15/40 distinct events. Four prompts/event are repeated measurements, not independent replications. The two starts share data and adapted ancestry. This observation establishes that the printed tuple is not a sufficient description of the final answering procedure in those conditions; original context and hidden state remain available. It does not show a selective internal causal interaction or prove that fields were unused.

The converse is also present. With normal drafts, among actor/patient question pairs where the same tuple is emitted and both fields are correct, noisy-verifier answers change in only 12/55 and 9/55 pairs. They are both correct in 10/55 and 8/55. With swapped drafts, both-correct drops to 2/56 and 2/58. This is a selective-use problem: sensitivity to the requested role is weak even when the emitted tuple is useful. Semantic mapping uncertainty and conditional subset selection remain limitations. The same-tuple comparisons do not themselves require declaring the tuple semantically correct.

A deterministic last-field or last-candidate rule is not sufficient for the noisy verifier: ordinary answers equal the patient field in 80/160 and 64/160 cases, and the last candidate in 81/160 and 75/160. Multiple simpler readout/default strategies remain possible. No global mixed-neural-mechanism conclusion follows.

## 4. Scientific decision

The broad multiple-factor explanation is **not established**. We have separable interface, relative-candidate preference and context-dependent answering observations. The planned instruction×role result is uncertain; the old causal panel admits a strong positional alternative. It is premature to design another training loss around question interference.

Preferred working account: a meaningful part of the newest discrepancy arises in expressing/terminating a requested span; a residual extraction/readout discrepancy remains. In adapted models, original context can affect answering through information not captured by the printed tuple. This is a testable account, not one proven common mechanism across untouched and adapted systems.

Falsification: the interface account weakens if its rescue fails on independent natural questions or is solely annotation-boundary leniency. A contextual readout account weakens if fixed-field reset and continuous readout behave equivalently after appropriate interface and computation controls. A genuine interaction claim needs prediction beyond calibrated main-effect/component models on held-out documents, not merely a nonzero risk-scale coefficient.

## 5. Single next experiment

A **fixed-field context-reset/readout test on untouched Qwen**, with strong direct-QA controls. Reuse generated, nongold role fields; hold their text and the question fixed. Compare reading them within the original extraction history against a fresh context containing the same tuple and question. Include a role-swapped-field negative control and a clearer two-pass direct-QA baseline, recording actual costs. Analyze all eligible development events, not only repairable failures. Known-role routing is an oracle diagnostic alongside natural-question readout, not the primary system.

If natural-question readout fails but known-role routing succeeds, query mapping is implicated. If a fresh readout improves correctness beyond continuous readout and two-pass direct QA with fixed useful fields, contextual competition becomes a stronger lead. If it does not, prefer interface/component accounts over a proposed interference mechanism. Fresh-context invariance for literally identical prompts is guaranteed by construction and **cannot** count as discovery or learned robustness. This extends the earlier fixed-field/tag probe by manipulating retained history in untouched weights, rather than repeating its tag/swap comparison in adapted models.

## Literature constraints checked today

- [Mixing Mechanisms](https://arxiv.org/html/2510.06182v2), especially §3.3–3.4, already tests interacting positional/lexical/reflexive retrieval and separates pointer from answer copying with counterfactuals. Multiple mechanisms or mixture alone is not a new contribution.
- [Semantic Role Circuits](https://aclanthology.org/2026.findings-acl.1964/) already reports causal role circuits. Our old whole-state intervention does not reproduce that level of specificity.
- [Measuring Faithfulness in CoT](https://arxiv.org/abs/2307.13702) already intervenes on intermediate text and finds variable answer reliance. Printed-field/answer discrepancies alone are not novel.
- [Making Reasoning Matter / FRODO](https://aclanthology.org/2024.findings-emnlp.882/) already separates correct intermediate inferences from faithful use and trains both. A generic extractor-plus-faithful-router proposal must face this precedent. Abstract-level verification today, not a reproduction or detailed method comparison.

These sources constrain interpretation; no exhaustive novelty clearance is claimed.

## Reproduce / inspect

`python3 research/frontier_answer_contract_2026-10-15/reanalyze_history.py`

`python3 research/frontier_answer_contract_2026-10-15/analyze.py`

Both use stored outputs and do not run a model. PROTOCOL.json fixed generation code and source/model hashes before inference. COMPLETE.json records output hashes. PREDICTIONS.jsonl logs exact unpadded input/output IDs and finish reasons. ANALYSIS.json contains every cell and paired interval. Historical reanalysis records 14 source hashes and verifies they remain unchanged. New inference used CPU float32, two threads, greedy max48 tokens; 192 calls generated 686 tokens in about140 generation seconds, plus about53 seconds for candidate scoring/overhead after loading. No sealed data were read and no earlier files were rewritten.
