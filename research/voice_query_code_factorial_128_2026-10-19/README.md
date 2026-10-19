# Current status: technical stop; zero completed fresh outputs

See REPORT.md and EXECUTION_STATUS.json. The interrupted STARTED.json remains intact; no automatic resume or retry.

# Qwen sentence–question–code factorial

The current scientific panel is the128-token correction frozen at31661d9. The original192-token preflight atc819994 stopped after two exposed-fixture calls because max-logit drift .0859375 exceeded .05. It remains failed and preserved in the neighboring `voice_query_code_factorial_2026-10-19` directory. No fresh passage was evaluated before the correction. Scientific prompt/token rows and hypothesis functions are byte-identical. New panel1117 calls + original exposed2 = total1119 cap.

## Design

Six AI-authored asymmetric directed-predicate/name families; j01–j03 development and j04–j06 evaluation. Role, question direction, sentence voice, question voice, answer lexicon, code mapping and neutral prefix are crossed. All768 new-task prompts use one explicit entailment instruction and a response legend after the question. Forty-eight old-prompt references separate task reframing from code/lexicon effects. No old threshold or new calibration is used. Both meaning and requested output symbol are recorded separately.

The three operational patterns, their tolerances, held rule and independent fixed cut14 routing diagnostic are in PROTOCOL.md. They can all fail. The two-of-three family rule is a small-panel predictive criterion, not a p value or evidence for unique semantic stages. All individual answers, four-answer units and full-family completeness remain distinct. No competence filtering changes collection.

## Reproduction

Use the exact existing Qwen weights/native-source hashes and environment listed in LOCK.json; no download/API is performed. `prepare.py --freeze` refuses an existing lock and rehearses the full controller on synthetic scores without loading an LLM. `run.py` refuses a second STARTED.json. Independent recollection must use a new output directory and separately preserved freeze; never overwrite these results.

1. `run.py`: logs every attempt before dispatch and saves full vectors before scoring;1117 fixed forwards if numerical controls pass.
2. `analyze.py`: frozen score/shape/hash replay and all factorial/routing outcomes; prediction replay checks that held outcomes followed the saved development prediction.
3. `storage.py`: verified lossless consolidation into tar bundles and an indexed archive reader. Only expanded staging NPZ copies are removed, after their exact member bytes are verified. Every original NPZ byte remains locally and remotely recoverable.
4. `packaged_audit.py`: independent post-bundling score/member verification.
5. `report.py` and `plot.py`: descriptive report/figures, no extra hypothesis tests or model calls.

`storage.load(root, raw_record)` transparently reads a staging NPZ or its tar member. Extract bundle members to restore ordinary per-call paths if desired. After packaging, rerunning `analyze.py` retains the same scientific results; its extra staging-file SHA check is replaced by the archive loader's member SHA assertion, so bookkeeping check counts may differ. The originally collected frozen audit is preserved; the separate packaged audit verifies every member explicitly.

## Scope

Native traces preserve embeddings plus28 residual block outputs at sentence period, question end and original assistant end. Intervention traces preserve five depth anchors with full-trace/prefix hashes; omitted full-prefix arrays are not recoverable. Attention/MLP decomposition is deferred. Prefix invariance constrains what can depend on later input; it does not prove an earlier semantic computation is complete, because later blocks can reread earlier states.

Positions at the three measured endpoints are exactly matched for prefixed-active versus unprefixed-passive cells. Individual word positions/order and effects of the prefix itself remain different. Causal masks are collected only in canonical A/B-normal/empty-prefix conditions, across both question voices; causal format-invariance across all code conditions is not tested.

Answer symbols/legends can induce mapping or formatting failures. Scores use declared semantic targets; reversing a legend can reverse target-aligned scores if the model ignores the legend. Such outcomes do not establish lost relational information. Neither score comparability across lexicons nor a successful operational pattern identifies a unique linguistic mechanism.

`PACKAGING_DEDUPLICATION.json` documents release of earlier duplicate archives after all1885 original old payloads and published archive hashes were verified. Those original per-call data remain local; the removed packaging remains recoverable from the specified GitHub commits. No earlier scientific values were deleted or failed gate overwritten.
