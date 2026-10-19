# Competence-selection confirmation (19 October 2026)

A single prospective 24-lexical-family Qwen/Llama study following an explicitly post hoc construction-selection finding. `PROTOCOL.md`, `LOCK.json` and `FREEZE_RECEIPT.json` identify the source/inputs frozen on GitHub before real inference. Historical failed gates are unchanged. AI-only authored/annotated synthetic sentences; no human review or corpus inference.

## Reproduction

Use the existing Python/runtime and exact local weights listed in LOCK.json. No downloads or API calls are performed by the inference scripts. `prepare.py` refuses to overwrite the frozen lock. `run.py` refuses automatic resume or retry. To recollect independently, copy the study into a new directory, preserve the protocol, prepare a separate freeze receipt and use a new output directory; never overwrite these observations.

- `prepare.py`: static tokenizer, freshness, model-source hash, masks and complete synthetic controller rehearsal; zero real model forwards.
- `run.py qwen7b`, then `run.py llama3b`: exactly781 attempted forwards each unless a technical gate fails. Answers never determine collection eligibility.
- `analyze.py`: reconstruct every score and label from NPZ vectors, check journals/hashes, then compute the frozen family-level primary test and every secondary profile.
- `core.py`, `engine.py`, `instrument.py`: inherited native scorer/mask/wrapper code. The inherited core FAMILIES, jobs, select_prediction and summarize are unused; authoritative fresh panel is INPUTS.json.
- `SOURCE_MANIFEST.json`: all24 AI annotations, uncertainty, earlier INPUTS hashes and freshness check scope.
- `LITERATURE_SCOPE.md`: relevant direct precedent, especially Jenner et al.'s reported filtering-related structural shift; no novelty claim.

Native is the vocabulary-wide first-token argmax at the original assistant prompt end, not full generated-answer accuracy. External fixed-threshold labels are separate classifiers. Full29-depth period/answer residual traces are stored for native baselines; interventions store depths0,7,14,21,28 plus hashes of full traces and full-prefix numerical checks. Every call stores full output logits losslessly. No attention/MLP writes or KV donation are captured by this experiment.

Final REPORT.md, RESULTS.json, ANSWER_LEDGER.csv, numerical audits and payload archive manifest/shards are added after collection. Archive members preserve the original per-call NPZ bytes; their SHA256 values are recorded in raw journals and the manifest. Concatenate ordered shards and extract the resulting gzip-compressed tar. No model weights or protected keys are published.
