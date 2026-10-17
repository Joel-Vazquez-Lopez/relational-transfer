# Family-dependent computation: first feasibility gate

**17 October 2026. Frozen behavioral screen; gate failed.** This is a new, bounded study following the completed and stopped transport experiment. It makes no claim about shared or different internal mechanisms. No hidden states were captured, no interventions were run, and no paid service or model download was used.

## Question and frozen design

Could the available local panel support a prospective test of computational similarity *within* a model family and *across* model families on the same role-retrieval task? The primary panel was Qwen 1.5B, Qwen 7B and Llama 3B; Qwen 0.5B was diagnostic. Eight fresh constructed name/verb families, four development and four evaluation, each had two active and two passive Agent questions with both role assignments. The exact [lock](GATE_LOCK.json) and [runner](run_gate.py) were saved before model output; the output files record their SHA-256 hashes and the reused [full-name scorer](../causal_developmental_alignment_2026-10-17/factorial_gate.py). The lock SHA-256 is `375e1927af634342c0d86f1e6f072859b96680426682469c005ec78dbdfec0d7`; runner SHA-256 is `d0ec378804857bfbe2f4ede8ae92a731e0b04df389d69f96173e4e225ec0dbe2`.

A row passed only if greedy generation exactly matched the expected name after normalization **and** the teacher-forced full-name log-probability margin favored that name. A family passed only if all four rows passed. The predeclared threshold for proceeding to a separately locked causal study was **at least two development and two evaluation families competent in all three primary models**. Families were not replaced. The printed raw logs and [machine summary](GATE_SUMMARY.json) preserve every answer, prompt, candidate score, and failure.

## All outcomes

| Model | Joint-correct answers | Fully competent families | Development IDs | Evaluation IDs |
|---|---:|---:|---|---|
| Qwen 0.5B (diagnostic) | 20/32 | 0/8 | none | none |
| Qwen 1.5B | 22/32 | 2/8 | f03 | f07 |
| Qwen 7B | 28/32 | 5/8 | f01, f03, f04 | f05, f07 |
| Llama 3B | 26/32 | 3/8 | f01 | f05, f07 |

| Family, verb | Split | Qwen 0.5B | Qwen 1.5B | Qwen 7B | Llama 3B | All primary |
|---|---|---|---|---|---|---|
| f01, hugged | development | fail | fail | pass | pass | fail |
| f02, called | development | fail | fail | fail | fail | fail |
| f03, pushed | development | fail | pass | pass | fail | fail |
| f04, followed | development | fail | fail | pass | fail | fail |
| f05, hugged | evaluation | fail | fail | pass | pass | fail |
| f06, called | evaluation | fail | fail | fail | fail | fail |
| f07, pushed | evaluation | fail | pass | pass | pass | pass |
| f08, followed | evaluation | fail | fail | fail | fail | fail |

The common primary-model set contains **0/4 development families and 1/4 evaluation families**. The frozen gate therefore fails. No causal fingerprint or path comparison is permitted as a result of this gate.

The answer-level profile also matters. Qwen 0.5B answered 1/8 B-Agent active questions correctly but 7/8 B-Agent passive questions correctly, so an apparent passive advantage in these particular items is possible; this is not a population finding. Qwen 7B answered all 16 active questions correctly but only 12/16 passive. Llama answered 15/16 active and 11/16 passive. Qwen 1.5B had five rows where its candidate margin favored the correct name but greedy generation returned more than the name; the frozen rule counted these as failures. The other three models had no greedy/scorer disagreements. As an explicitly **post hoc diagnostic**, ignoring greedy formatting and requiring only correct candidate margins still leaves exactly one common family, f07; it would not rescue the development requirement. Complete row details are in [Qwen 0.5B](GATE_qwen05.json), [Qwen 1.5B](GATE_qwen15.json), [Qwen 7B](GATE_qwen7b.json) and [Llama 3B](GATE_llama3b.json).

## Decision and limits

**NO-GO for the proposed family-dependent causal comparison on this task and local panel.** The primary models do not share enough fully competent families to estimate a within-lineage versus across-lineage causal pattern. Comparing only f07 would select one evaluation item and provide no development fit. Relaxing the exact-answer rule, changing the prompt, adding a new verb or choosing a different model now would be a new study, not a repair of this one.

This result does **not** refute family-dependent computational equivalence. The gate is small, synthetic and limited to one Agent question form, four verbs and eight name pairs. Family identity bundles names and verbs; it does not isolate a verb effect. It does show that model competence and construction sensitivity are material obstacles to the proposed comparison, even on short, unambiguous sentences. No activation or circuit claim follows from these behavioral outputs.

The broader research bet remains conditional: a credible test needs a task that at least two models in one lineage and a second lineage all perform on enough independent held-out families, plus predeclared native causal effects that beat answer, position and generic-depth controls. [Bhardia et al. (2026)](https://aclanthology.org/2026.acl-srw.84.pdf) already show model-family-dependent numerical comparison circuits, and [Aljaafari et al. (2026)](https://aclanthology.org/2026.findings-acl.1964.pdf) already compare semantic-role circuits across families. Thus even a later positive result would need to add a *prospectively predictive boundary* to this prior work.

## Reproduction and provenance

Run each model in a separate process with the existing MLX environment, e.g. `research/stronger_model_development_2026-10-15/.venv/bin/python research/family_dependent_computation_2026-10-17/run_gate.py qwen15`, then `python3 research/family_dependent_computation_2026-10-17/analyze_gate.py`. The runner refuses to overwrite raw output. The analysis script verifies all prompts, conditions, expected names, lock and runner hashes before writing the summary. Model configuration hashes and raw file hashes are included in the JSON. Model weights and runtime are existing local assets outside this study directory.
