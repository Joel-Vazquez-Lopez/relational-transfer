# Linguistic Development Atlas — repaired pilot results

**18 October 2026 · CLOSED · 69/132 cumulative model calls · 0 held-out calls.**

## Decision

**The implementation repair passed. The prospective scientific decision is NO-GO for held-out evaluation because neither model passed the development competence gate.** The broad linguistic-development hypothesis remains unresolved. Valid descriptive development maps were collected, with no unique semantic-mechanism or novelty claim.

The user explicitly authorized the repair and testing after the original serialization failure. The original failed attempt remains intact at commit `878b715`; the amended continuation was frozen before additional inference at `5425e2516a8f014585517316b522e8391bba99ba`. All held-out model outputs remained unobserved throughout.

## 1. What was fixed and tested

The faulty `dict(**row, **score(...))` supplied `expected` twice. The production persistence function now asserts agreement and uses a dictionary literal. The scorer itself is unchanged. Validation logits are persisted immediately, preventing recurrence of the missing-payload problem.

An inference-free integration rehearsal exercised all **128 actual input metadata rows with explicitly synthetic causal states**, including scoring, dictionary assembly, JSON/tensor/logit round-trips, duplicate/mismatch rejection, prefix controls, the full analysis, and six exported figures. It also replayed the previously saved real Qwen capture through the saving path offline. These are software tests, not 128 empirical model answers.

`SCIENTIFIC_INVARIANCE.json` verifies byte-identical instrumentation, analysis, original protocol and annotation files, plus AST-identical scoring/input functions. `AMENDMENT.json` explicitly supersedes the original administrative no-resumption/per-model-count provisions for this user-authorized continuation. Scientific prompts, sites, scoring, hypotheses and thresholds were retained.

## 2. Numerical controls

| Control | Qwen 7B | Llama 3B | Criterion |
|---|---:|---:|---:|
| Wrapped/unwrapped maximum logit discrepancy | 0 | 0 | ≤ .05 |
| Exact-repeat maximum logit discrepancy | 0 | 0 | ≤ .05 |
| Exact-repeat relative state discrepancy | 0 | 0 | ≤ 1e−6 |
| Before-cue relative interaction | 0 | 0 | ≤ 1e−6 |
| Sentence-prefix discrepancy across questions | 0 | 0 | ≤ 1e−6 |
| Embedding relative interaction | 0 | 0 | ≤ 1e−6 |
| Native block reconstruction relative error | 0.00055844666 | 0.00051603131 | ≤ .002 |

Both numerical gates passed. Identical causal prefixes were exactly invariant in the tested fixed-128-token captures. These results apply to this panel; the historical Llama invariance failure remains a failure in its original study. Attention/MLP reconstruction is numerical accounting, not proof of causal use.

## 3. Answer outcomes, separate from family competence

These are original-prompt-end full-vocabulary next-token predictions, scored with the frozen deduplicated Yes/No alias margin. No generated rollouts were collected. All argmax tokens were recognized Yes/No aliases; there were no unexpected tokens, refusals or formatting failures.

| Demand | Qwen correct | Llama correct |
|---|---:|---:|
| agent | 8/8 | 8/8 |
| theme | 8/8 | 7/8 |
| goal | 8/8 | 2/8 |
| order | 4/8 | 5/8 |
| **Total answers** | **28/32** | **22/32** |

| Lexical family | Qwen correct | Llama correct | Fully competent model/family screens |
|---|---:|---:|---|
| a01: recommend, Ada/Boris/Clara/Neil | 14/16 | 10/16 | Neither |
| a02: refer, Mara/Oscar/Hugo/Rose | 14/16 | 12/16 | Neither |

**Fully competent development families: Qwen 0/2; Llama 0/2.** The threshold remains all 16 answers per family. It was not weakened after viewing outcomes. Each model needed at least one fully competent family before held-out inference. Both gates therefore stopped; a03/a04 received no model calls. The reserved budget-family priority policy was never activated.

### Error patterns

- **Qwen:** Agent, Theme and Goal questions were correct in all **24/24** exposed cases. Mention-order answers were **No in all 8 cases**, yielding 4/8 correct and causing both families to fail. It would be invalid to drop this control post hoc and call the original gate passed.

- **Llama:** Goal questions were **Yes in all 8 cases**, yielding 2/8 correct. It also made one Theme error and three Order errors. These results concern this instruction/question/readout protocol, not general incapacity for Goal assignment or word order.

All 64 answer rows are in `ANSWER_LEDGER.csv`; all 14 incorrect rows are in `FAILURES.csv`. Native RAW.json files include prompts, gold labels, full-vocabulary argmax, class probabilities/margins and top-20 tokens. Full logits remain local.

## 4. Descriptive developmental observations

The original analysis exported every exposed four-cell contrast across embeddings, all 28 residual blocks and all declared sentence/question positions, plus attention/MLP writes and labeled response Grams. Incorrect cases remain present and are explicitly marked. No fitted alignment, selected layer window, PCA, new similarity measure, predictor or additional inference was introduced.

The interaction `I=(across_swap−across_original)−(within_swap−within_original)` was exactly zero before the first distinguishing and/to cue. It became nonzero at eligible positions. The sentence-period curves were exactly invariant across later questions, as causal masking requires.

At prompt end, the Agent questions had the same correct answer in all four cells within each family, but still had a nonzero interaction. A state depending only on a fixed final Yes/No label cannot describe that full geometry. Ordinary nonlinear context processing, syntax/marker-keyed copying, confidence-dependent output preparation and generic depth schedules acting on input-dependent features remain viable. This observation is weak against those stronger rivals.

Qwen showed much larger late prompt-end interaction norms for Theme/Goal than for the fixed-Agent questions. Llama had substantial interactions even on consistently wrong Goal answers. A visible geometric signal therefore cannot itself be read as successful linguistic processing. Raw amplitudes are model-native and cannot be treated as comparable calibrated units across the different widths and training histories.

The table below reports the prespecified all-block mean `||I||/(||Pwithin||+||Pacross||)` at the two anchors. The numerator and both denominator terms, all stages and sites, and undefined values are retained in `DEVELOPMENT_MAP.csv`/`DESCRIPTORS.npz`. A ratio is not a percentage of semantic computation.

| Model | Family | Demand | Period mean ratio | Prompt-end mean ratio | All four answers correct? |
|---|---|---|---:|---:|---|
| qwen7b | a01 | agent | 0.5068 | 0.4690 | True |
| qwen7b | a01 | theme | 0.5068 | 0.6220 | True |
| qwen7b | a01 | goal | 0.5068 | 0.6510 | True |
| qwen7b | a01 | order | 0.5068 | 0.4713 | False |
| qwen7b | a02 | agent | 0.5463 | 0.4363 | True |
| qwen7b | a02 | theme | 0.5463 | 0.4982 | True |
| qwen7b | a02 | goal | 0.5463 | 0.5788 | True |
| qwen7b | a02 | order | 0.5463 | 0.5219 | False |
| llama3b | a01 | agent | 0.6130 | 0.5651 | True |
| llama3b | a01 | theme | 0.6130 | 0.7233 | False |
| llama3b | a01 | goal | 0.6130 | 0.6864 | False |
| llama3b | a01 | order | 0.6130 | 0.5892 | False |
| llama3b | a02 | agent | 0.5685 | 0.5118 | True |
| llama3b | a02 | theme | 0.5685 | 0.6655 | True |
| llama3b | a02 | goal | 0.5685 | 0.6253 | False |
| llama3b | a02 | order | 0.5685 | 0.5323 | False |

### Figures

- [Qwen full atlas](ATLAS_qwen7b.svg) and [Llama full atlas](ATLAS_llama3b.svg).
- [Qwen native-magnitude lines](LINES_raw_qwen7b.svg) and [Llama native-magnitude lines](LINES_raw_llama3b.svg).
- [Qwen normalized lines](LINES_ratio_qwen7b.svg) and [Llama normalized lines](LINES_ratio_llama3b.svg).

PNG copies are local. Representative native Qwen line and Llama atlas figures were visually checked for titles, axes, legends and undefined-value display; all six figures rendered through the integration-tested pipeline. Plots remain descriptive development data, not held-out replication.

## 5. Competing explanations and scientific limits

| Explanation | What this run says |
|---|---|
| Strict additive lexical/frame state | The observed interaction is nonzero; strict additivity does not describe these native states. Nonlinear recoding alone suffices. |
| Fixed answer-label-only hidden state | Same-label Agent cells still differ interactively. Richer output/confidence/context accounts survive. |
| Generic layer schedule | Not excluded as a modulator of content-sensitive features; no validated cross-model operation matching is established. |
| Linguistic composition or constituent-gated retrieval | Compatible with maps, but the competence/control gate failed and no held-out prediction was evaluated. |
| Strong syntax/marker/position copier | Survives; the pre-existing surface-string program solves all draft labels, and this run provides no unique discriminator. |

Two author-constructed development families are not a population sample. Answers, layers and positions are correlated. Coordination frame also changes argument cardinality and local and/to cues; the intervention is not an isolated semantic variable. Annotation is **AI-only by explicit user waiver**, with no independent human validation of scope, pragmatic readings or question naturalness. Only one model per family/scale was tested, so family-level conclusions are unsupported.

The original design discusses Dai et al. (2024), *Representational Analysis of Binding in Language Models* ([primary source](https://aclanthology.org/2024.emnlp-main.967/)), which already studied binding representations and activation edits redirecting retrieval. This observational pilot does not establish a new binding mechanism, causal use, or an advance over those results. Ordinary factorial interactions and layerwise linguistic probes also have extensive precedent. No novelty is claimed.

## 6. Call accounting and protocol history

- Original failed attempt: **2 calls**, including one saved main capture and one unpersisted unwrapped validation. Its failure report remains intact.
- Authorized continuation: **67 additional calls**. The first main capture was reused without recomputation; lost validation was rerun and both exact-repeat checks retained.
- Cumulative: **64 unique main captures + 5 validation calls = 69/132**. Qwen accounts for 35, Llama 34. No held-out calls, paid APIs, downloads or training.
- Both development gates were evaluated before either held-out phase. Each phase was attempted once. The halted held-out commands checked the gates and made no model calls.
- Frozen amended code and scientific input hashes remained unchanged during the continuation. The earlier sandbox/GPU startup recovery is documented in the original attempt, not silently discarded.

`BUDGET.json` is the global attempted-call ledger. The lock and pre-inference amendment identify exact code, installed runtime, weight/tokenizer/config bytes, reused capture provenance and the exposed development case. The original lost validation output remains missing and is counted; no values were imputed.

## 7. Reproducibility and closure

Source scripts, amendments, raw answer/control ledgers, numeric maps, SVG figures and reports are preserved on the existing research branch. Full selected-site arrays, full logits and binary descriptors remain local, with checksums in the result manifest; GitHub text records do not imply that every tensor is uploaded. The original study and failed gates are preserved, and the exploratory observatory branch is untouched.

**The repair is verified; the pilot has completed its allowed development stage and stopped at its scientific gate. No prompt revision, control removal, held-out run, model addition, patching or further adaptive experiment follows this report.**
