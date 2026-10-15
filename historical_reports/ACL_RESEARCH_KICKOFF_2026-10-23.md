# Explaining Language Through Science — research supervision and GitHub kickoff

**Prepared:** 14 October 2026  
**Target kickoff:** 23 October 2026  
**Status:** planning document, not a preregistration or completed experiment.

## Research north star

Investigate whether query-dependent event-role errors in pretrained language models reflect failures of event-role information, failures to use it for answering, or interactions with construction, participant order, question context and adaptation. Make the strongest claim that survives frozen evaluation and independent replication, rather than force a particular theory.

**Possible working title:** *When Questions Change Who Did What: Diagnosing Context-Dependent Relational Computation in Language Models*. This is NOT a novelty or publication claim.

## Evidence carried into the project

- Candidate-free adapted-Qwen dev panel: both actor and patient correct rose 11/100→31/100 (start 1) and 13/100→21/100 (start 3) with typed cues, **but the prespecified success gate failed**.
- On the same panels, printed two-question role-tuple stability fell 71/100→56/100 and 76/100→45/100; answer-to-field routing improved. Exploratory data, with cue, wording and requested role confounds.
- Construction-exposure interventions showed passive gains with active losses across two starts and two lexical banks (related panels; not four independent replications).
- Earlier same-sentence hidden-state patching in untouched Qwen2.5-1.5B shows question-state steering in selected active, actor-first examples; still confounded with first/last-person selection.
- Higher-weight graph-penalty attenuation failed the average ability-matched test, so **do not** claim ability-independent structural robustness.
- Independently verified semantic labels for the natural active/passive challenge were not yet available at the October 14 audit. No established natural-language repair or ACL-ready claim.

## Responsibilities and collaboration

**Astra:** Complete its currently frozen untouched-model diagnostic as designed. Preserve complete configs, seeds, checkpoints/ancestry, raw outputs, and all gate outcomes. Do not retrofit its design to the interaction hypothesis.

**ChatGPT (research supervisor/red team):** Critically review preregistration, label validity, leakage, experimental controls, metrics, uncertainty, code, and claims. Propose adversarial alternative explanations. Compare papers and novelty at submission time. Review each milestone before expensive experiments. Can author code, reports, issues, branches and PRs during active sessions where the appropriate tools and permissions are available. Does not autonomously work in the background without a scheduled task/workflow.

**Researcher:** Decide scientific scope, supply model compute and human annotation, approve releases and GitHub mutations, adjudicate disputed semantics, make final scientific calls.

## October 23 — proposed first-session agenda

1. **Freeze the artifact baseline.** Import a clean code-and-results snapshot; record data and checkpoint hashes, exact environment, provenance, licensing, and which artifacts are excluded. Keep datasets and weights out of the Git history if access or size requires it.
2. **Review Astra's completed diagnostic**, if available, against its *original* prespecified criteria. If still unfinished, log it as pending; do not imply results.
3. **Pick one primary scientific claim and falsifier.** Candidate: changing the role question alters reported event bindings in ways that are construction- and position-dependent. Explicit null: variation is explainable by prompt format, answer choice or role extraction conventions without a binding effect.
4. **Build the validated evaluation set before intervention.** Independent semantic annotation of actor, patient, target event and span; balanced voice/order/reversibility; held-out event and document families. Maintain a sealed test set.
5. **Commit a minimal baseline pipeline.** Frozen untouched models; deterministic parsers; direct QA, neutral extraction, actor-conditioned extraction, patient-conditioned extraction, question-after-extraction; candidate-free and candidate-supported settings clearly separated.
6. **Predeclare analysis.** Joint exact correctness at the event level as a primary outcome; tuple stability, answer/field agreement, invalid outputs and worst-construction correctness as diagnostics. Paired contrasts with event-cluster uncertainty; model/seed and annotation uncertainty. Register interaction terms and a separate confirmation sample.
7. **Stop/go review.** Advance to activation patching only with verified labels and a robust behavioral effect; advance to repair only after mechanism alternatives have been tested.

## Experiments — sequence with evidence gates

### Gate 0 — Validity and clean baselines
- No adapted checkpoint mislabeled as pretrained.
- Human-confirmed semantic roles; participant order and constructions controlled.
- Comparable prompts and compute budgets, parsing failure rates reported.
- Raw predictions and failures saved; event IDs de-identified if necessary.

### Gate 1 — Reproducible behavioral phenomenon
- Identical sentences/events across actor/patient/no-question contexts.
- Test whether question-conditioned tuple differences occur beyond formatting/parsing artifacts and surface positional rules.
- Analyze interactions (voice × requested role × order × candidate support); include appropriate clustered/mixed-effects models and out-of-sample replication.
- If no clean phenomenon: pivot to adaptation-induced regressions or the stronger controlled construction trade-off.

### Gate 2 — Causal mechanism
- Probe and patch the same event across answer contexts, with active/passive and reversed-role counterfactuals.
- Separate semantic actor/patient from first-mentioned/last-mentioned and name-token position.
- Include sham patch locations, random donors, and output-format controls; test necessity/sufficiency conservatively. Probe decodability alone ≠ causal use.

### Gate 3 — Mechanism-motivated method (only if justified)
- Compare against direct QA, SRL+router, separated extraction, matched inference-time deliberation, generic consistency, supervised augmentation, and compute-matched shams.
- Test untouched or fresh model families, held-out constructions, independent natural documents, noncandidate answers, and worst-condition regressions.
- Preserve negative outcomes. No novelty or ACL claim before literature/empirical verification.

## Proposed repository layout

```text
explaining-language-through-science/
├── README.md
├── LICENSE                          # choose after checking third-party data/code terms
├── CITATION.cff
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── research_questions.md
│   ├── evidence_register.md
│   ├── experiment_registry.md
│   ├── decisions.md
│   └── limitations.md
├── protocols/
│   ├── clean_pretrained_baseline.md
│   └── natural_role_adjudication.md
├── src/els/                         # data validation, evaluation, stats, patching
├── experiments/
│   ├── 00_frozen_baseline/
│   ├── 01_question_construction/
│   └── 02_causal_patching/
├── tests/
├── configs/
├── results/                         # compact machine-readable metrics and manifests
├── artifacts/MANIFEST.md            # externally stored weights/large inputs
└── paper/
    ├── claim_evidence_matrix.md
    └── outline.md
```

## GitHub operating rules

- Start **private** until licenses, privacy and author permissions are resolved; researcher decides when to publish.
- Initial commit includes reproducible scripts and documentation, not 3 GB of raw weights or private/unlicensed corpora.
- Use feature branches + reviewable pull requests, not direct work on `main` for experiments.
- Version experiment configs and analysis scripts *before* inspecting sealed test results; record hashes and metadata for outputs.
- CI: lint, schema validation, unit tests for metrics/parsers, leak tests, and smoke test on tiny public or synthetic fixtures. Large model runs are not CI tasks.
- Never commit secrets, access tokens, private participant data or restricted dataset text.

## Paper decision tree

**Strong clean pretrained interaction + causal evidence + external replication** → primary ACL scientific/mechanistic submission; optionally add method if genuinely general.

**Clear behavioral effect but inconclusive mechanism** → rigorous evaluation/diagnosis paper; do not claim that internal bindings were observed to change.

**Only fine-tuning creates the issue** → adaptation-regression paper with strong controls; compare fixes against forgetting and distribution-shift baselines.

**No stable question effect, but construction-transfer trade-off replicates** → focus on construction-conditioned training generalization as the main contribution.

**No trustworthy cross-model, held-out effect** → publish careful negative results or reframe as an openly reproducible benchmark; do not claim a successful repair.

## Day-one acceptance criteria

- GitHub repo exists only after user authorizes repository and access.
- Source scripts run and tests pass on a clean environment.
- Every displayed number maps to experiment ID, hash, data split and code commit.
- Astra's sealed protocol untouched.
- One-page claim/evidence/falsifier table agreed.
- No claim of ACL acceptance, novelty, or general mechanism without supporting evidence.

## Information required at kickoff

- GitHub repository owner/name (or authorization to create one).
- Desired license and release privacy (private default recommended).
- Astra's finished diagnostic output, including raw predictions and prespecified criteria.
- Available compute and annotation resources.
- Target ACL venue/submission cycle and coauthor approval process.