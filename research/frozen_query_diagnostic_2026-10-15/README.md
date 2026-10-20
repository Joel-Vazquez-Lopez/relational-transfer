# Frozen-model query diagnostic — 15 October 2026

This is a development diagnostic, not independent confirmation. It tests untouched Qwen2.5 0.5B and 1.5B instruction weights on 24 original QA-SRL event/question pairs selected from an already examined development source. No adapters are loaded. No training occurs.

## Validity gate before inference

The earlier 100-item selection includes prepositional questions, e.g. `Who did someone finish behind?`, incorrectly treated as patient questions. The diagnostic selects the first 24 of 56 events passing original QA-SRL active subject/direct-object question-slot rules: no preposition, no second object, no negation or passive. Original consensus answers and alternative consensus spans are retained. Grammatical subject/object are not asserted to be semantic agent/patient. All selected events are active and subject-first; no passive or positional robustness claim is possible.

## Comparisons

- Direct answers to the original question.
- Subject/object extraction and final answer with the original question visible.
- Subject/object extraction without a question, with deterministic selection by the known original-question slot.

The last condition has an oracle question-role selector. It tests a pipeline diagnostic, not end-to-end question interpretation. Reusing one extraction guarantees stable printed fields by construction. Correctness, not guaranteed consistency, is the meaningful outcome.

## Commands

Use `/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python` for inference.

```
python run.py run --model 0.5b
python run.py run --model 1.5b
python analyze.py
```

Run from this directory, or use the full script path from the project root. Per-response JSONL writes allow identical configurations to resume. Completed IDs are skipped; model and protocol hashes are checked. Historical experiments remain unchanged. See SMOKE_NOTES.md for a schema failure and analysis-hash recording timing.

The 1.5B weights were already downloaded. Temporary-cache configuration links were broken. Public config/tokenizer-config/generation-config files were restored from revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` into `local_qwen15`. Tokenizer vocabulary/merges/tokenizer JSON are byte-identical shared files referenced from the intact 0.5B cache (matching the original 1.5B blob identifiers). Weights are linked from the existing local cache. Exact file hashes are in PROTOCOL.json.

Completed results and interpretation: SCIENTIFIC_REPORT.md. The follow-up `field_only_ablation.py` controls the extra Answer-field instruction; `analyze_field_only.py` analyzes it. `secondary_routing.py` applies the pre-existing question-classifier recipe to cached outputs as a post hoc diagnostic. All historical predictions are retained.
