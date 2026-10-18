# Source-access feasibility — results

## Verified rerun: interpretation and decision

**Scoring repair verified; development source-access effects observed; fresh evaluation and cross-model confirmation stopped at competence.**

The rerun used **333/2000 actual forwards**, including every candidate and generation continuation. The earlier failed attempt used 11 separately preserved forwards, so the source-access programme total is 344. There were 107 primary records: 16 baselines, three numerical validations and 88 development intervention/control conditions. No evaluation intervention or Llama forward ran.

An independent offline audit recomputed all name-path probabilities and verified every greedy token and EOS from saved logits for **104 scored records**. Maximum probability-summation discrepancy was 1.07e-10. No additional model calls were used. All frozen files, the exact original stimuli and the original failed attempt remain unchanged.

### Actual complete-answer outcomes

| Development condition | Correct / total | Terminated wrong | Truncated at four tokens |
|---|---:|---:|---:|
| baseline | 8/8 | 0 | 0 |
| cut_0 | 0/8 | 8 | 0 |
| cut_7 | 0/8 | 8 | 0 |
| cut_14 | 0/8 | 8 | 0 |
| cut_21 | 0/8 | 1 | 7 |
| cut_28 | 8/8 | 0 | 0 |
| irrelevant_7 | 8/8 | 0 | 0 |
| irrelevant_14 | 8/8 | 0 | 0 |
| irrelevant_21 | 8/8 | 0 | 0 |
| restore_all | 8/8 | 0 | 0 |
| restore_7_14 | 0/8 | 8 | 0 |
| restore_14_21 | 0/8 | 6 | 2 |

**What this establishes:** in these two exposed active-sentence families, closing all suffix access to the event at the tested nontrivial cuts disrupted the declared short-name response. The matched irrelevant cuts preserved 24/24 responses; cut-28 shams and full restoration preserved 8/8 each. Zero-access suffix states were exactly invariant across name-swapped orientations. This supports dependence on the source under the specified intervention.

**What this does not establish:** an onset of role representation, absence of internally stored information, a reusable semantic mechanism, or a different processing sequence between questions/models. Attention removal changes normalization and can produce unfamiliar states. Both demands lost the short-answer endpoint at every tested nontrivial cut. At block 21, seven responses were truncated, so they cannot be described as seven observed completed wrong names. No longer continuation was generated to resolve them.

### Frozen prediction and fresh failure

The unchanged development rule selected block 14, with Agent-minus-first matched source cost +3.369145 nats and positive differences in both development families. Its raw-development hash was saved before evaluation and independently verified. All eight fresh passive baselines were then evaluated, yielding **7/8**: e1 passed 4/4 and e2 passed 3/4. The required complete-panel gate therefore failed.

> Sentence: Sarah was visited by James.
>
> Question: Who was mentioned first in the sentence?
>
> Expected: Sarah. Generated through EOS: James.

The gold-versus-rival full-name margin was -0.968733 nats. This is a complete wrong answer under the declared task, distinct from the original partial-token scoring defect. Both complete development families and the Helen/Martin passive family passed; we did not select those successful cases to bypass the failed panel. The prospective **intervention prediction remains untested**, rather than having been falsified by an evaluation intervention. Llama was conditional on Qwen's evaluation success and was not run.

### Margin interpretation: exploratory audit, not a revised success rule

Raw margin loss alone cannot identify a question-specific route. If both queries have the same retained fraction r after a cut but different baseline margins b_A and b_F, then their cost difference is (1-r)(b_A-b_F), which can be nonzero with identical source dependence. This is a competing explanation, not an inferred mechanism.

At block 14 the combined probability of either candidate name prefix averaged only **0.0000266** after the cut. Thus a sizeable margin between those two names did not mean the model actually answered with either name. Greedy completions are reported separately above. The exact decomposition below shows how control confidence and residual candidate preference contribute to the selected statistic:

| Cut | Baseline Agent−first margin | Irrelevant-cut Agent−first | Event-cut Agent−first | Primary difference |
|---|---:|---:|---:|---:|
| 7 | 2.320318 | 2.843757 | -0.091151 | 2.934907 |
| 14 | 2.320318 | 3.632820 | 0.263675 | 3.369145 |
| 21 | 2.320318 | 2.375007 | 1.477305 | 0.897702 |

This decomposition was added after outcomes to constrain interpretation. It did not alter the selected cut, prediction, endpoint, thresholds or stop decision. The evidence supports a narrow intervention-dependent access observation; the hoped-for generalizing cross-model developmental account is not demonstrated.

### Numerical checks and closure

Native/wrapped/dense/repeat logit discrepancies and repeat-state discrepancy were all zero. All development cut-28/full-restoration logit errors and zero-access orientation discrepancies were zero. Maximum last-token block reconstruction relative error across captured primary passes was 0.000635454 (limit .002). Maximum original-prompt logit discrepancy on continuation passes was 0.000000000 (limit .05). These are numerical controls, not proof that attention removal preserves a natural computation.

The scorer is verified. **NO-GO for proceeding to evaluation interventions or a cross-model claim under this protocol.** No prompt repair, replacement family, new cut, longer generation or additional model run follows. Scripts, every raw scored record, every forward logit payload, controls and local tensor hashes are preserved. The figure below reports all fixed development conditions; every bar contains four correlated answers from two families.

![Development complete-answer outcomes](ANSWER_OUTCOMES.png)

Closed bounded execution: **333/2000 attempted forwards**. No paid API, training or model download.

## Decisions

| Model | Outcome | Baseline answers | Calls |
|---|---|---:|---:|
| qwen7b | STOP_EVALUATION_COMPETENCE | 15/16 | 333 |
| llama3b | NOT_RUN | 0/0 | 0 |

## What this tests

The intervention closes later access from every question/assistant suffix token to all event-dependent context tokens, while preserving earlier computations. It measures continued-access dependence under that mask. It does not identify when semantic information first exists, establish a reusable role representation, or distinguish a strong syntax-aware copier from semantic composition.

## Complete baseline outcomes

| Model | Family | Correct / total | Entire family correct? |
|---|---|---:|---|
| qwen7b | d1 | 4/4 | True |
| qwen7b | d2 | 4/4 | True |
| qwen7b | e1 | 4/4 | True |
| qwen7b | e2 | 3/4 | False |

## qwen7b: development matched effects

| Cut (zero-based block) | Agent minus first-person source cost, nats | Irrelevant absolute drift, nats |
|---|---:|---:|
| 7 | 2.934907 | 0.785157 |
| 14 | 3.369145 | 1.203126 |
| 21 | 0.897702 | 0.039064 |

### qwen7b: validation

```json
{
  "wrapper_max_logit_error": 0.0,
  "native_dense_max_logit_error": 0.0,
  "repeat_max_logit_error": 0.0,
  "repeat_max_state_error": 0.0
}
```

## Failures and controls

ANSWER_LEDGER.csv includes every scored call, including technical-only validation rows and controls. FAILURES.csv includes every incorrect full-name endpoint, including intentionally source-isolated calls; those are distinct from baseline competence failures. SUMMARY.json retains all decision fields and full control findings. RAW.json includes generated text/tokens, EOS flags, full candidate path scores and top-20 initial outputs; FORWARD_RECORDS.json includes all actual continuation passes; full-vocabulary logits and native captures are local.

## Limitations and prior work

Two constructed lexical families per phase are not a population sample. Questions differ in wording/length; passive evaluation changes construction and names together. Answer identities are matched on active development but differ between demands in passive evaluation. Quantized local models and one runtime limit generalization. Attention removal changes normalization and may cause off-distribution damage. Matched irrelevant cuts match edge count but not absolute position or attention mass. A preserved answer does not exclude later rereading; a failed answer does not prove absence of an internal representation. No clean cross-model replication is claimed after a failed gate.

Attention knockout and staged retrieval have close precedents in [Geva et al.](https://aclanthology.org/2023.emnlp-main.751/), [Mixing Mechanisms, appendix F](https://arxiv.org/html/2510.06182v2), and [Lookbacks, appendix K](https://arxiv.org/html/2505.14685v3). [Dai et al.](https://aclanthology.org/2024.emnlp-main.967/) already demonstrates binding-related activation edits. This feasibility run makes no novelty claim.

## Preservation and closure

LOCK.json freezes source, stimuli, installed native model code and local weight/tokenizer/config hashes. FREEZE_RECEIPT.json identifies the pre-inference commit. BUDGET.json journals attempts before dispatch. Tensor/logit files remain local with checksums in RESULT_MANIFEST.json; uploading text artifacts does not imply uploading every tensor. Earlier experiments and their gates were not modified. This is the user-authorized, separately frozen scoring rerun; the earlier failed attempt remains unchanged. No additional adaptive follow-up follows this report.
