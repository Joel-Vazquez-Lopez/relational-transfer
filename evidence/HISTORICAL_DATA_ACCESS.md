# Historical raw research access and import status — 2026-10-15

The prior research programme is much larger than the evidence registry. A vetted **external research copy** has been produced and ZIP-integrity tested:

- File: `RELATIONAL_RESEARCH_HISTORY_REVIEWED_2026-10-15.zip`
- SHA-256: `2fbc2d376b43a2841ec5645a000acb2058a0f63df0f958869896f3a8df26658c`
- Source: `COMPASS_RESEARCH_FOUNDATION_2026-10-15.zip`
- Source SHA-256: `4d005bffa43131a0f24d4e52a33c076617578489181b8b4db4b19276362793ae`
- Source archive: 4,448 file entries
- Screened handoff: **3,505 unmodified original source entries**, plus readme and detailed `SHARING_MANIFEST.json`.
- Excluded: **943 entries**, with path-level reasons recorded in the ZIP manifest.
- Coverage: historical controlled Transformer studies (2,547 items); recent LLM development (696); older LLM studies (256); root research documentation (6).
- Exclusions include logs/binary results, restricted or reserved evaluations, review sheets, large datasets requiring separate manual review, and a few archive metadata files.

**Crucial:** This package is currently **not checked into this GitHub repository**. It must be transferred securely by the research lead, with SHA-256 validation before use. Access to this repository is not equivalent to access to the original evidence. This foundation PR therefore does **not** yet make all historical raw predictions executable from GitHub alone.

## Original historical reports actually in this branch

- [Research frontier](../historical_reports/RESEARCH_FRONTIER.md)
- [Reproduction guide](../historical_reports/REPRODUCTION_GUIDE.md)
- [Role diagnosis, October 10](../historical_reports/LLM_ROLE_DIAGNOSIS_2026-10-10.md)
- [Independent research audit](../historical_reports/INDEPENDENT_RESEARCH_AUDIT_2026-10-14.md)
- [Independent ACL handoff review](../historical_reports/INDEPENDENT_ACL_HANDOFF_REVIEW_2026-10-15.md)
- [Research kickoff plan](../historical_reports/ACL_RESEARCH_KICKOFF_2026-10-23.md)

Three additional source reports were not pushed because the file-write tool rejected their full contents. Do not treat those omissions as scientific exclusions, and do not attempt to bypass repository safety controls.

## Workflow for investigators

1. Read the registry, limitations, and original reports committed here.
2. Obtain authorized historical evidence ZIP from the research lead.
3. Check ZIP SHA-256 and inspect `SHARING_MANIFEST.json`; run independent calculations on raw inputs/outputs and report source paths.
4. Keep sealed confirmation data and restricted corpora outside research worktrees.
5. Make evidence-backed PRs with commands, configuration, code, machine-readable outputs, limitations, and negative findings.
6. Do not equate archivally reported outcomes with independently replicated results.

**Status:** Curated scientific summaries and selected original reports are in GitHub; raw experimental history is packaged externally but not yet uploaded to GitHub.
