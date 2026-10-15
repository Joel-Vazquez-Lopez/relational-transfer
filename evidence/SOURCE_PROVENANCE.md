# Verified archive provenance

The source ZIP was inspected locally; file existence and sizes were confirmed via its central directory. Archive SHA-256: `4d005bffa43131a0f24d4e52a33c076617578489181b8b4db4b19276362793ae`.

All paths below are *inside* `COMPASS_RESEARCH_FOUNDATION_2026-10-15.zip`, underneath the top-level directory of the same name.

| Original archive path | Bytes | SHA-256 |
|---|---:|---|
| `EXPERIMENT_INDEX.md` | 3558 | `88ece0fddc29887c77e7853fece3e0eb77847aea1af58c1f9b30c9ae8622fd8e` |
| `REPRODUCTION_GUIDE.md` | 3046 | `c46fc0fdd5e87e59af272124ceb3d1d2bb71dc45da9a84a8acc919ba5c9b1c9b` |
| `source/RESEARCH_CHECKPOINT_2026-10-14.md` | 13028 | `12fe53a3124efe55ce9781fe1b3440b0e6b568b239dd059214f5b7eee1eb0183` |
| `source/RESEARCH_FRONTIER.md` | 9411 | `45194e330458af3d3502cd9151f0c1a0a6caf8b90d5fa577f738334c872c9324` |
| `source/llm_role_repair_confirmation/RESULTS.md` | 6648 | `804f0d968c6fc8f4623b453cd32f93cdbd1686cdd43de3ffb435425983e99db1` |
| `source/llm_role_unordered_control/RESULTS.md` | 4711 | `69c1cf25a7084a5190d0bdabdbacdab51ffa684232897b2c9546963105964622` |
| `source/research/stronger_model_development_2026-10-15/RESULTS.md` | 7673 | `523558482bb4a119ca6dfae72284f39ef0e888b14acbddcd0d4929d3350e4162` |
| `source/research/phi3_cross_family_development_2026-10-15/RESULTS.md` | 11840 | `56e49abcec152972270aef7c430d7b495642466c2f5db5b0bfeaa97a2876a204` |
| `source/research/llama_gemma_cross_family_2026-10-15/RESULTS.md` | 2530 | `2094e20f6c5c19c98824168b3907761f7705f5e70b31705d5131bd43da335e06` |

The archive also contains `MANIFEST.json`, `VERIFY_PACKAGE.py`, raw JSON/JSONL predictions, the original experiment protocols, dependency specifications, and negative findings. These are **not present in this compact GitHub snapshot**. Do not cite GitHub as though it hosted the raw experiments.

Archive exclusions include model/adaptor weights, virtual environments, restricted QA-SRL payloads, sealed/future validation material, and some redundant binary output. Exact recovery of historical fine-tuned checkpoints is not possible from the ZIP alone. 
