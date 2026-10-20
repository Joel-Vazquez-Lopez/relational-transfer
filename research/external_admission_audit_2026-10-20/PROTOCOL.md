# External admission-sensitivity audit: availability gate

20 October 2026. Locked before inspecting any external per-example outcome arrays. This is an artifact reanalysis, with zero model inference, training or paid API calls. Prior local outcomes are exposed and unchanged.

## Candidate policy

Inspect these independently authored studies in this fixed order, based on relevance and published artifact links, not effect size:

1. Dai et al. 2024, binding: `cl-tohoku/OI-Subspace`, including official ACL software supplement.
2. Zhang & Nanda 2024, patching best practices: paper-linked code/artifacts. The paper identifies `redwoodresearch/Easy-Transformer` as an experiment codebase; attribution of any results must be verified.
3. Fierro et al. 2025, multilingual factual recall: `constanzafierro/multilingual_factual_memorization`.
4. Nikankin et al. 2025, arithmetic heuristics: `technion-cs-nlp/llm-arithmetic-heuristics`.
5. Jenner et al. 2024, learned chess look-ahead: `HumanCompatibleAI/leela-interp`, including linked Figshare metadata.
6. Meng et al. 2022, ROME causal tracing: `kmeng01/rome`, including official project-linked tracing artifacts if present. This last candidate is included for an established factual-recall intervention task; it is not a language-construction replication.

Select the first eligible study under the criteria below. Inspect repository metadata, schemas and collection code before numerical outcomes. Pin every inspected repository to a commit. Avoid executable deserialization (pickle/torch.load) or executing downloaded research code. Fetch only bounded source and data artifacts: individual files ≤20 MiB, total ≤200 MiB. Do not download weights. Notebook source and output schema can be inspected; recorded outcome values will count as exposure and must be disclosed.

## Required data

Per-example stable IDs; unperturbed native choice or sufficient scores to recover it; gold/target identity; intervention results at specified components; original metric definition; and prescreening/collection provenance. At least two natural strata relevant to the original task must be definable without inspecting intervention magnitudes. Linguistic construction strata are preferred; nonlinguistic replication must be labeled accordingly.

The source must include both originally correct and originally incorrect cases with the same intervention measurement, or another legitimately justified prespecified admission contrast. A correct-only intervention archive cannot identify its own excluded population. Datasets alone, plotting aggregates, code alone or corruptions without paired outcomes do not pass. Missing or unverified provenance is not interpreted as an empirical null or a flaw in the original paper.

## Analysis gate and freeze

If a source passes availability, write and timestamp/commit a source-specific analysis plan before computing filtering sensitivities. Specify population, original causal endpoint, native correctness definition, stable support, strata, missingness, equal-weight comparison, minimum denominator guard if needed, and the decision about the original claim being evaluated. Keep original conditional and unfiltered estimates distinct. No calibration on evaluation outputs, no best-layer or best-stratum search, and no new significance test from this availability stage.

If none passes, STOP the external numerical analysis and publish the exact availability ledger. Do not replace the missing external data with local synthetic prompts or claim a replication. A scoped failure to locate usable artifacts is not proof that no such public data exists.
