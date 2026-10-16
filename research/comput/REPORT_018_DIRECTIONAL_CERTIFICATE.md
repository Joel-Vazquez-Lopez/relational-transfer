# COMPUT-018 — Directionality-identifiability certificate (16 Oct 2026)

**Status:** Executed deterministic **open synthetic development** benchmark. A proposed **acceptance control for Mira**, not a replacement V6, human-reviewed authentic dataset, LLM inference result, or new neural mechanism.

## Non-duplication gate
Before coding, reviewed active research branches, open PRs #13–#26, operations issues #7/#16/#22/#23 and current Brain-O E043–044, Compass I16, Mira authentic-language intake PR #26 and synthetic V5 PR #25. None of the inspected workstreams runs this exact **same query + same bag + same ordered undirected clauses + same voice markers + opposite gold** certificate. Division: Comput = deterministic directionality falsification; Mira = authentic annotation and eventual dataset redesign; Astra = Qwen output-field order; Brain-O = trained compound-key mechanism/copier; Compass = learned composition; Nova = inference runner. GitHub status can change after snapshot.

## Pair specification

Five distinct names `A,B,C,D,E`; two verbs `v,u`; four clauses. World A: `A v B; B v C; D u A; E u C`. World B: `B v A; C v B; D u A; E u C`.

For question `Who u the person who v B?`: World A gold = **D**, B = **E**.
For question `Who u the person whom B v?`: World A gold = **E**, B = **D**.

Generator exhausts **16 nonoverlapping five-name pools × 24 clause permutations × 16 per-clause active/passive voice masks × 2 question roles = 12,288 paired contrasts**, each with two opposite worlds = **24,576 model prompts**. Eight verb pairs cycle across families. These rows share one topology/grammar, so **do not count as 12,288 independent structural worlds or human language examples**.

## Executed checks

| Invariant / baseline | Outcome |
|---|---:|
| Pair query, token-bag equal | **12,288/12,288** |
| Pair predicate-labelled undirected graph equal | **12,288/12,288** |
| Pair ORDERED unordered event clauses + active/passive markers equal | **12,288/12,288** |
| Pair reference answer flips | **12,288/12,288** |
| Grammar-aware directional active/passive parser | **24,576/24,576 (100%)** |
| Undirected reader unique-answer cases | **0/24,576; abstains on all** |
| Alphabetical tie-break (undirected candidates) | **12,288/24,576 (50%)** |
| First-mentioned-candidate tie-break | **12,288/24,576 (50%)** |
| Last-mentioned-candidate tie-break | **12,288/24,576 (50%)** |
| Per nominal family alphabetical score | **768/1,536 (50%)** |
| Independent cold label parser, separate script | **24,576/24,576 agreement with private gold** |
| New executed Python tests | **9/9 passed** |

**Proof / scope:** A deterministic predictor restricted to the **fixed query, ordered *unordered* predicate–name edges, and per-clause voice markers** observes identical features in the two paired worlds and therefore must output the same name. Since the two correct answers are opposite, its paired accuracy is at most 50%. This **does not bound** grammar-aware, actor/patient-position-sensitive or other syntax-based shortcuts: the simple directional template parser scores 100%. This experiment is a necessary directionality gate, *not* demonstration of abstract relational computation or a neural cache.

## Reproducibility / access

Original standalone user-delivered ZIP `COMPUT_018_DIRECTIONAL_IDENTIFIABILITY.zip`, SHA256 **`ce24a641b0c2ef5363e87ffd0324d9ea38c2480a1142ad2b94638427b96bc78e`**, 390,976 bytes; source, nine tests, separate cold parser, prompt-only generated JSONL, isolated private-gold JSONL, paired contrasts, audit JSON, internal SHA manifest, test log and full detailed scientific report. ZIP CRC PASS.

`python3 -m unittest -v test_comput018_directional` and `python3 comput018_directional.py --out generated`. Independent `python3 independent_replay.py generated/model_prompts_open.jsonl generated/reference_gold_private.jsonl` returned 24,576/24,576.

Only the literal `prompt` value is model-visible. Private IDs, family/voice metadata, original pairs and scorer gold must remain outside model calls. No inference performed; no neural checkpoints or sealed data involved.

## Limits and next gate

A restricted active/passive **surface parser** perfectly handles these synthetic questions, so passing COMP-018 need not imply genuine abstract event semantics, OOD language generalization or a discovered internal algorithm. Repeated verb families and shared structural template prevent population-level claims from 24,576 repeated transformations. Require independent adjudication, held-out structural families and a competent syntax-only readout before interpreting any LLM outcomes.

**Recommendation:** Mira can use the paired invariant as a **minimum V6 direction-sensitive stimulus test**, while keeping ownership of actual V6 and real human-annotated source data. Comput will critique, not take over its generator. No paid inference, PR merge, or new dataset release is authorized by this protocol alone.

**Signal: PROTOCOL FALSIFICATION / PROGRESS, not EUREKA.**