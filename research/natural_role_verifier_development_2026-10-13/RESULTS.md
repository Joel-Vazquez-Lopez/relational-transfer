# Natural-role verifier development pilot — 13 October 2026

## Question and frozen design

Does training a small pretrained language model to *correct* possibly wrong actor/patient drafts improve final role answers? `PROTOCOL.json` was frozen before training. The comparison uses Qwen2.5-0.5B source adapters from two matched starts (`fresh1`, `fresh3`), 160 sentence-disjoint QA-SRL Bank 2.0 training items, 40 separate development items, and 51 previously examined QA-SRL Gold items as an **exploratory** check. Each item has actor/patient questions in both candidate orders (four prompts per item). The original QA question annotations are human data, but our actor/patient mapping is automatic and has not been independently reviewed.

The five arms are answer-only, candidate-first structured output, role-first structured output, clean-draft verifier, and noisy-draft verifier. The two verifier arms have identical gold output targets; the noisy arm sees swapped actor/patient drafts on 303/640 training prompts and must correct them. At evaluation, both verifiers receive the role-first arm's generated draft fields; a separate stress condition swaps those fields. Answer-only has a shorter output target (1,972 tokens per start versus 8,476 for each structured arm); token budgets are in `ANALYSIS.json`. Verifiers also require a second inference pass. This is a development pilot, not a matched-compute or final publication comparison.

The prespecified positive screen required noisy verifier to beat clean verifier by at least 4 of 40 fully correct development items in **each** start, stay within 4 items of role-first, and parse at least 80% of outputs. All four actor/patient × candidate-order answers must be right for an item to count as fully correct.

## Development results (40 items, 160 prompts per start)

| Arm / draft condition | fresh1 exact /160 | fresh1 all-four /40 | fresh3 exact /160 | fresh3 all-four /40 |
|---|---:|---:|---:|---:|
| Answer-only | 98 | 4 | 91 | 0 |
| Candidate-first | 84 | 0 | 82 | 0 |
| Role-first | 87 | 1 | 85 | 1 |
| Verifier, clean training; ordinary draft | 103 | 9 | 97 | 5 |
| Verifier, noisy training; ordinary draft | 87 | 2 | 84 | 2 |
| Verifier, clean training; swapped draft | 76 | 4 | 75 | 2 |
| Verifier, noisy training; swapped draft | 75 | 0 | 76 | 0 |

**The positive screen failed in both starts.** Noisy minus clean all-four was -7 and -3 items, versus the required +4 in each. All outputs parsed, so the failure is not a formatting failure. Clean verification improved ordinary-draft direct answers over role-first in both starts, but the improvement depended on draft reliability and does not establish a robust repair.

### A simple inference-time routing system

The model's generated role fields can instead be used as the answer: a deterministic rule chooses `Actor` when the question requests the actor and `Patient` when it requests the patient. This rule was prespecified as a diagnostic and uses the externally known question type. It adds no model training, but it does rely on structured generation and an explicit role selector.

| Development output | fresh1 all-four /40 | fresh3 all-four /40 | fresh1 exact /160 | fresh3 exact /160 |
|---|---:|---:|---:|---:|
| Answer-only generated answer | 4 | 0 | 98 | 91 |
| Role-first generated answer | 1 | 1 | 87 | 85 |
| **Role-first fields + deterministic routing** | **24** | **26** | **119** | **121** |

On the previously examined Gold items, the same routed system gets 29/51 and 28/51 fully correct, versus 4/51 and 2/51 for answer-only, and 1/51 and 2/51 for role-first direct answers. This is a substantial **pipeline result on these automatically mapped items**, not evidence that the model learned an internal answer-routing mechanism or that it will generalize to an independently reviewed challenge set. The question-role selector is supplied by the evaluation setup; unseen natural question wordings would need their own selector or a tested question-role classifier.

### Where the chain breaks

"Routed" means a deterministic rule selects the requested role from the model's emitted actor/patient fields. It is a diagnostic, not the model's final answer. On the development items:

| Verifier / draft condition | fresh1 routed /160 | fresh1 own answer /160 | fresh3 routed /160 | fresh3 own answer /160 |
|---|---:|---:|---:|---:|
| Clean training; ordinary | 119 | 103 | 121 | 97 |
| Clean training; swapped | 36 | 76 | 32 | 75 |
| Noisy training; ordinary | 118 | 87 | 114 | 84 |
| Noisy training; swapped | 117 | 75 | 122 | 76 |

The noisy arm's role fields were robust to swapped drafts: routed accuracy stayed 118→117 and 114→122. The clean arm's fields collapsed: 119→36 and 121→32. Yet the noisy arm's final answer under swapped drafts was only 75/160 and 76/160. Thus **draft correction in emitted fields did not translate into reliable role-to-answer routing**. The clean arm's own answers were sometimes right despite wrong emitted fields, so final answers cannot be assumed to be mediated by those fields. This is a behavioral dissociation, not proof of a specific internal mechanism.

## Previously examined QA-SRL Gold items (51 items, 204 prompts per start)

These items were used in earlier exploration and cannot serve as independent confirmation. The same pattern appears:

| Arm / draft condition | fresh1 exact /204 | fresh1 all-four /51 | fresh3 exact /204 | fresh3 all-four /51 |
|---|---:|---:|---:|---:|
| Answer-only | 116 | 4 | 112 | 2 |
| Candidate-first | 97 | 1 | 95 | 0 |
| Role-first | 102 | 1 | 101 | 2 |
| Verifier, clean training; ordinary | 119 | 10 | 119 | 6 |
| Verifier, noisy training; ordinary | 108 | 2 | 104 | 3 |
| Verifier, clean training; swapped | 84 | 1 | 90 | 2 |
| Verifier, noisy training; swapped | 101 | 3 | 104 | 3 |

Noisy verifier routed accuracy under swapped drafts was 129/204 and 139/204, compared with only 45/204 for clean verifier in both starts. Its own answers were 101/204 and 104/204. Role-first deterministic routing from emitted fields was 153/204 in both starts, versus 102/204 and 101/204 for its own final answers. These are diagnostic ceilings conditional on generated fields, not intervention gains.

## Interpretation and limits

1. The proposed noisy-draft *learned* verifier is **not a successful answer repair** under the frozen screen. Do not frame it as one. Explicit deterministic routing is a separate, promising system-level repair on these items.
2. Correcting draft role fields and answering from those fields are separable behaviors. That is a sharper target for the next experiment than adding more role labels alone.
3. The 40 development items share a source distribution with training, although sentences are disjoint. Participant identities are not disjoint. Role mappings require independent human review. The 51 Gold items are exploratory. Only one small model family and two starts were tested. No internal activation analysis or causal intervention was run here.
4. The clean and noisy verifier arms have equal target-token counts, but differ in input corruption. Answer-only has fewer target tokens, and verifiers have extra inference computation. The observed clean-verifier gain should be read with these differences in mind.

## Next decision

Do **not** run this verifier on the sealed final set or claim publication confirmation. The immediate high-value next test is to independently review the sealed natural items, freeze labels and the role-selector rule, and compare the existing answer-only and role-first checkpoints against deterministic routing. A separate learned intervention could then try to force the final answer to be selected from *corrected* role fields; it must beat deterministic routing as well as answer-only and role-first at matched data and compute where feasible. Score role fields and final answers separately. The sealed 60-item review queue (`SEALED_TEST_HUMAN_REVIEW.csv`) remains untouched by model inference; it is mostly direct-object material, so a separate passive/construction-balanced set is needed for broader claims.

## Reproduction artifacts

- `PROTOCOL.json`: prespecified comparison and file hashes.
- `SPLITS.json`: development/train items and exclusion list.
- `natural_train.py`: frozen training and inference script.
- `EVAL_*.json`: all raw outputs and per-prompt scores.
- `ANALYSIS.json`: validated summaries, exact checkpoint hashes, and screen outcomes.
- `SEALED_TEST_MANIFEST.json`, `SEALED_TEST_HUMAN_REVIEW.csv`: untouched future review queue; no model scores exist for it.
