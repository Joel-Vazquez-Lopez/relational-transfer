# Comput — independent falsification and statistical audit

**Status: active, exploratory.** This workstream now includes *independent retrospective counting/reparsing of some published archival model outputs* in addition to deterministic shortcut, identifiability and synthetic benchmark-design tests. **No new pretrained-model inference or independently reproduced neural training/activation-patching run** is established by the reports alone. See `evidence/HISTORICAL_DATA_ACCESS.md` on `main` for access limits.

## Reproduce (where original unsealed inputs are accessible)

```bash
python3 -m unittest discover -s research/comput -p 'test_*.py' -v
python3 research/comput/identifiability.py research/comput/controlled_design.jsonl
python3 research/comput/comput015_v2_shortcut.py /path/to/open_binding_pairs_v2.jsonl
```

The first command is a **requested independent branch-level test run**, not evidence the whole suite passed in CI. Report-specific invocations, archived source provenance, and denominator definitions remain authoritative in each report. Keep sealed data, model weights and credentials outside Git.

## Reports / interpretation gates

- **R003–R005:** cluster sensitivity and mathematical bounds over aggregate cells. These bounds do not recover missing event-level pairs or justify unclustered significance.
- **R006 → R007 (IMPORTANT ORACLE-LEAKAGE CORRECTION):** Early cross-family position-rule fits included policies derived from **reference actor/patient labels and construction metadata**. These retrospective maxima were **NOT verified label-free deployable heuristics**. R007 narrows to restricted position-policy proxies; their agreement with model outputs is neither gold accuracy nor an internal algorithm. Cite **R007 whenever reporting R006**.
- **R008–R010:** counterbalanced synthetic benchmarks can remain construction-position solvable; source-surface equivalence and raw-event recovery gates. Perfect programmatic benchmark scores do not prove natural relational computation.
- **R011:** independent descriptive rescore of the unsealed historical `llm_role_internal_v2/RESULT.json`: 384 archived answer records, 240/384 correct and 368/384 agreeing with a voice×question policy. Selected archived patch summaries yield 41/48 versus 0/48 under the two donor conditions. These are source-row checks, **NOT independent replay of a neural activation intervention** and not a claim that the model actually implements the heuristic.
- **R012–R013:** verb-specific residuals and margin predictors. Gold/reference-role-indexed diagnostics are **not equivalent to autonomous raw-text predictors**; pair-level fit may fail leave-one-verb-out. Report both.
- **R014:** independent comparison of archived Astra direct role responses versus extraction, with distinct format/semantic outcomes and clustered dependence. Valid formatting does not explain all direct-role collapses; successful extraction does **not** establish stored semantic bindings.
- **R015:** generator-aware, label-safe shortcut on protocol-auditor v2. Correct **768/1,024** answers, including all 512 base and 256 donor; **0/256** on donor cases whose gold role answer changes. This invalidates aggregate score as a relational-mechanism metric but **does not defeat the paired gold-changing intervention**.

### R015 independent exact-source verification (2026-10-16)

A separate supervisor check used the **byte-identical committed source scripts**, confirmed by Git blob SHAs:

- Upstream `build_pairs_v2.py`: `b4fa28bc0782c861ed54cedf33ce2d03123ff417`.
- Comput `comput015_v2_shortcut.py`: `b009a225c4dea96599a00ca2f0831635d12e8333`.

The supervisor executed the upstream generator with 16 families, then ran the actual committed Comput scorer over all 512 produced rows (1,024 base/donor evaluations). This reproduced **768/1,024**: base 512/512; identity donors 128/128; distractor-role donors 128/128; target-role donors 0/128; predicate-swap donors 0/128. Original generator full-text parser 1,024/1,024; sorted-name rule 256/1,024. Upstream JSONL SHA-256: `a33773e9a7c43dc28060282fab203cc94276fcc3538aad71df7b023cf4027167`. Complete log and checksums are cited in [supervisor issue #7 comment](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/7#issuecomment-6203465177). This is **a deterministic source-locked reproduction**, not independent LLM inference.

## Open scientific and review dependencies

1. **H1:** Independent intake of *authorized original unsealed* Mira V4 generator/data; Comput has explicitly reported `BLOCKED: original ZIP`.
2. **H3:** Independent intake of Nova's sanitized CAL-01 48-response package; `BLOCKED: authorized transfer`. Do not mistake a prepared package for a recipient-verified handoff.
3. **H2:** Independent cold replay of Brain-O E037–038 checkpoint bundle; source+weight archive has not been received here. Its correction of passive participant-token indexing supersedes earlier values. A competent lexical-copying model remains a live rival.
4. **PR #13:** Review the full expanded report scope, run and archive the exact committed unit tests, verify raw-prediction provenance and group dependencies before considering merge. This README revision alone does **not** satisfy these gates.
5. **Authentic-language requirement:** Predeclare label-safe first/last, syntactic/voice, local-verb, event-scope and output-format baselines for the human-adjudicated pilot in [issue #23](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/23). Do not treat synthetic family IDs or prompts as independent documents.

**Scientific rule:** Keep descriptive agreement, benchmark-identifiability, real-pretrained behavior and internal causal mechanisms as separate claims. Publish both negative controls and oracle-leakage corrections; never hide failed results or count correlated rows as independent samples.
