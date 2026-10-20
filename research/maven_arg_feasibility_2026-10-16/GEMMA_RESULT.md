# Hosted Gemma on the natural event × role pilot — 16 October 2026

**Follow-up:** A new 17-document panel and an adaptive Control wording check are reported in the [replication report](PANEL_V2_REPORT.md).

## Result

The previously used `google/gemma-4-31B-it` Berget endpoint answered the **same 24 questions on the same six MAVEN-Arg passages** as the initial Qwen pilot. It gave the exact normalized annotated local mention on **17/24** questions. The remaining **7/24** outputs did not satisfy that frozen string rule, but all seven are plausibly the intended participant with an article, title, or longer local phrase. Gemma gave no uniquely identified wrong-event or wrong-role answer and never gave identical Agent and Patient answers for one event (**0/12 pairs**).

This is a meaningful **behavioral competence boundary** relative to the small local Qwen models on these passages. It does not establish Gemma's internal mechanism. The six selected validation documents are exploratory development items, and the hosted API does not expose activations for causal patching.

| Model and interface | Exact local mention | Same-event wrong-role | Identical Agent–Patient answer |
|---|---:|---:|---:|
| Local Qwen2.5-0.5B, initial wording | 7/24 | 7/24 | 12/12 events |
| Local Qwen2.5-1.5B, initial wording | 12/24 | 7/24 | 9/12 events |
| Hosted Gemma-4-31B-it, same semantic messages | **17/24** | **0/24** | **0/12 events** |

The hosted model and local models differ in family, size, chat template, and serving stack. This table is a task-performance comparison, **not** evidence that parameter count alone caused the difference.

## Review of the seven non-exact outputs

This review happened **after** the frozen exact scoring and is reported as a sensitivity check. It does not replace the primary 17/24 score.

| Gold local mention | Gemma output | Review |
|---|---|---|
| `the United States` | `the United States women 's team` | Plausible contextual referent; the model copied a longer team noun phrase. The dataset's country annotation makes this the least exact equivalence. |
| `Netherlands` | `the Netherlands` | Same country, added article. |
| `United Belgian States` | `the independent United Belgian States` | Same named entity, added modifiers. |
| `Church` | `the Church` | Same entity, added article. |
| `Statists` | `the Statists` | Same group, added article. |
| `Géza Lukachich von Somorja` | `Major General Géza Lukachich von Somorja` | Same person, added title. |
| `Novak von Arienti` | `Major General Novak von Arienti` | Same person, added title. |

Thus a permissive contextual-referent review would count **24/24 plausible answers**, with the team/country granularity noted above. Since this rule was formed after inspecting the outputs, it is descriptive and must be specified in advance for a new panel.

## Method and audit trail

The [protocol](GEMMA_NATURAL_PROTOCOL.json) was frozen before paid inference (SHA-256 `cef9e85d1b6c85151ebcc7c61f9493dbd75ef2312a71972d623fbea9854a0d5e`). It locks the curated items, initial pilot protocol, script, questions, generation (`temperature=0`, `max_tokens=24`, no reasoning), and fixed exact/alias scoring. The same semantic system and user messages were sent as separate API calls for each of the 24 questions. Berget's current metadata reported the same model and EUR token prices as in the earlier Gemma study. All 24 calls returned with `finish_reason=stop`. No system fingerprint was supplied by the provider, so the served weight revision cannot be pinned.

The protected key was read only from `.env.local` by the existing client and was not written into research artifacts. A separate ledger imposed a **€0.05 hard ceiling** with per-request reservations and no automatic retries. Reported usage was 3,984 input tokens and 101 output tokens. Estimated token cost was **€0.0010465**, with **€0.002093** conservatively accounted in the local ledger; this is not a provider invoice.

Reproducibility files: [runner and scorer](gemma_natural.py), [raw predictions](GEMMA_NATURAL_PREDICTIONS.jsonl), [analysis](GEMMA_NATURAL_ANALYSIS.json), [completion hash](GEMMA_NATURAL_COMPLETE.json), and the prior [pilot report](PILOT_REPORT.md). The earlier “perfect” Gemma result was on a repeated synthetic panel, documented in the separate [cross-family report](../llama_gemma_cross_family_2026-10-15/RESULTS.md); this is the first natural-passage check in this study.

## Research decision

Gemma passes the **small development-panel competence gate**: it keeps event and role apart on these examples. The next high-value experiment is a larger, document-disjoint natural panel with independently adjudicated event/role labels and a **predefined contextual-referent scoring policy**. Run the frozen panel across Gemma and at least one other strong family, while retaining smaller models as error controls. If the goal is a claim about internal computation, use an accessible open-weight model that passes the same panel; the Berget text API supports behavioral tests but not activation interventions.
