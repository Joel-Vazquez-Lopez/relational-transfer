# Independent workstreams and evidence flow

**Research lead:** user — approves scientific priorities, expensive compute, releases, and final decisions.

**Scientific coordinator:** GPT-6 chat — reviews evidence, priority, novelty and integration; does not independently validate every numerical claim.

**Astra/Codex:** `research/astra-adaptation` — learning and adaptation investigations.

**Compass:** `research/compass-reuse` — computational organization, relational reuse and question-conditioned reconstruction.

**Research lead + Nova:** `research/nova-implementation` — implementation, reusable experimental tooling and original research.

**Mira:** `research/mira-replication` (proposed; verify existence) — independent replication.

**Comput:** `research/comput-falsification` — competing simple mechanisms and falsification.

**Brain-O:** idea development and scientific critique — no mandatory Git branch.

**Brain-X:** `research/brain-x-theory` — independent scientific invention, mathematical/computational theories, adversarial hypothesis testing and principled training/architecture proposals. Distinct from Brain-O's causal routing investigations and from the GPT-6 coordinator. Own research notebook, controls, negative results and reproduction code: [`research/brain-x/`](brain-x/README.md). **Status:** exploratory; changes pending review through [draft PR #21](https://github.com/Joel-Vazquez-Lopez/relational-transfer/pull/21).

## Integration rules

- This repository's branch names are organizational, **not access or security boundaries**.
- Workstreams may disagree. Do not interpret agreement among AI investigators as independently generated empirical evidence.
- `main` accepts reviewed, reproducible results via approved PRs. Do not auto-merge exploratory findings.
- Separate originals, archive-derived analysis, new development samples, and sealed confirmation data.
- Keep secrets, paid API keys, restricted corpora and model weights outside Git.
- Include raw prediction identifiers, protocol/version, dataset provenance, negative controls, uncertainties, and limitations in experiment reports.
- New claims require citation to exact file/checkpoint/model versions. Do not promote ARCHIVAL-REPORTED results to independently replicated without fresh verification.
- An investigator must verify repository and branch access. A chat session does not automatically share local worktrees or another chat's uploaded files.
