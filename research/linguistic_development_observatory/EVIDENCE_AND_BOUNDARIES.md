# Existing evidence: provenance and what it does NOT show

These are **historical reported results**, not replicated by creating this observatory. Primary research files are on Codex's [existing research branch](https://github.com/Joel-Vazquez-Lopez/relational-transfer/tree/research/causal-developmental-relations-2026-10-17/research/causal_developmental_alignment_2026-10-17), with decisions in [Issue #34](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/34). Some newer experiments were pasted into conversation and their full local reports have not been read here.

## 1 — What motivated our visual intuition

Initial local Qwen 7B / Llama 3B role-swaps: baseline greedy answers correct 8/8 each. An internal last-prompt-token residual patch is technically feasible; self-patch sham passed. **Only 2/4 Llama cases had valid frozen whole-name first-token margins**, because “Cl…” / “Ell…” were partial tokens. The first cross-model curves are a feasibility pilot, not comparable algorithm traces. Later full-name scoring fixed this measurement defect in a *different* prospective screen.

Full-dimensional hidden paths captured across participant, verb and final question-token sites. In 2D PCA, changes appeared to propagate differently across sites/models; separate PCAs do not define a shared coordinate frame and hide much variance. The qualitative multi-position up/down trends and intensity/timing differences are observational leads only.

## 2 — Why single-path cosine similarity failed as a linguistic detector

Exposed four-case role contrasts: mean endpoint cosine across cases approximately **0.03 Qwen / 0.07 Llama**. Across-case normalized-step Gram agreement was **0.82 / 0.79** for role changes but **0.92 / 0.90** for unrelated sentence changes. Cross-model turning similarity **0.575** for roles versus **0.586** unrelated. Large final-block movement was generic. At least eight layer steps apart, agreement fell to **0.024 / −0.011** for roles while unrelated remained **0.56 / 0.42**.

Thus the current scalar/matrix captures much generic architecture/depth structure; **it did not establish task-specific shared computation**. A same-endpoint/same-speed/same-within-path-Gram mathematical pair can still execute labeled operations in different orders.

Subject/object step directions under name swaps opposed one another early; raw name exchange can produce this without semantic understanding.

## 3 — Prospective feasibility gates that failed, and remain failed

Frozen actor-swap × active/passive Agent/Patient screen: Qwen passed **2/4 fully competent families**, Llama **0/4**; Qwen 43/48 and Llama 39/48 distinct factorial prompts correct. No intervention on those holdouts. Full details: [FACTORIAL_GATE_REPORT](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/causal-developmental-relations-2026-10-17/research/causal_developmental_alignment_2026-10-17/FACTORIAL_GATE_REPORT.md).

New frozen active↔passive event-site transfer study: shared development-family role-selective patch effect **+0.195 nats Qwen / +0.036 Llama** below the 0.5-nat threshold; Qwen unrelated same-answer donor had larger effects. **No evaluation-family patching**, no selective transfer claim. [TRANSPORT_STUDY_REPORT](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/causal-developmental-relations-2026-10-17/research/causal_developmental_alignment_2026-10-17/TRANSPORT_STUDY_REPORT.md).

## 4 — Recent task-conditioned trajectory lead (not independently replayed here)

Investigator-reported: after removing a direct first-token name readout axis, late active/passive response cosine remained approximately **+0.71 Qwen7B**, **+0.42 Qwen1.5B/Llama3B**, **−0.73 Qwen0.5B**. A simple copying model reproduced key features. In a fresh Qwen7B question-type panel, mean late cosine was **+0.10** for action questions and **−0.91** for first-mentioned-name questions, while geometry signs matched an output-only predictor in **16/16**. Qwen1.5B and Llama failed distinct prospective gates. These observations are useful **constraints**, NOT shared semantic-mechanism evidence. Original raw artifacts and complete protocol need provenance check before further analysis.

## 5 — Competing explanations to keep visible

- Same Transformer-family block schedule, residual architecture and normalization.
- Name/token/absolute-position changes; a shift to the last subtoken of a span.
- Surface syntax differences and tokenizer-dependent lengths.
- Answer preparation, output identity and distributed copying beyond a removed output axis.
- Different scale/model capacity, instruction tuning, chat templates and relative depth.
- PCA/normalization/trajectory alignment choices and out-of-distribution patching.
- Real linguistic or functional reorganization (hypothesis; not assumed explanation).

## Ground rules

Observed curve similarity is a **reason to ask why**, not a mechanism claim. Negative results are durable. Avoid substituting future held-out sets into development after viewing outcomes. Do not aggregate 16 correlated prompts as 16 independent event-family replications. Verify local data paths/hashes before computing from them.

## Provenance action at next session

Locate and read the locally saved NPZ and full reports, verify exact hashes, record model/weight/tokenizer/chat revisions, block indexing, token span mappings, units and plotting code. Existing GitHub files prove that reports exist; they do not prove that every locally mentioned NPZ was uploaded here.
