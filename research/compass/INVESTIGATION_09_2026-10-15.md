# Compass Investigation 09 — Discrete cache optimization landscape

**Date:** 2026-10-15. **Status:** synthetic mechanistic calibration, not a pretrained LLM discovery.

## Question
Why did the shared-backbone causal Transformer with one-bit pre-question cache and question-time raw reader (Investigation 08) stop at cost 20.0 at c=0.4, despite the exact optimum of 18.4?

## Exhaustive code landscape
All 16 independent binary worlds, four queries each (64 questions), all 65,536 possible world-to-one-bit encoders. For each encoder, choose optimal per-code/query cached majority answer or a perfect raw reconstruction at c=0.4. All losses are exact integer counts scaled by 5.

- Global best loss: **18.4**, reached by **32** encoders.
- **1,126** encoders have no strictly improving **single-world assignment flip**.
- Local minima by loss: 18.4 (32), 19.2 (8), 20.0 (128), 20.8 (432), 25.6 (526).
- Learned Investigation 08 encoding integers (world 0 LSB): seed 19 = 59520; 29 = 59528; 41 = 60072. Their optimal policy cost is 20.0. **Only seed 19** is a strict-descent single-flip local minimum; thus single-flip traps cannot explain all learned failures.

## Code-neighborhood search
Deterministic best-neighbor strict descent over 1 or 2 world-to-bit changes:
- seed 19: 20.0 → 19.4 → **18.4** (2 steps)
- seed 29: 20.0 → **18.4** (1 step)
- seed 41: 20.0 → **18.4** (1 step)

Of 100 fixed pseudorandom encoding starts, one-flip search reached global optimum in 75; two-flip search in 100. **These are random encoding starts, not neural training seeds or independent test examples.**

## Learned neural repair
Reloaded each original trained causal Transformer checkpoint and ran 600 supervised continuation steps with improved code labels selected from **training-world gold labels and reconstruction cost**, plus optimal cached-answer and gate targets. All three seeds exactly matched the selected 16/16 code and achieved **12 errors, 16 reconstructions, total cost 18.4**.

**Preserve-code negative control:** same architecture, same 600-step continuation, but keep original cache assignment targets. All three stayed at **20.0**. Reproducing the original training from frozen full source and then repeating the searched repair also yielded 20.0 → 18.4 in every seed.

## What this does and does not show
Shows a specific capacity/optimization distinction: the trained architecture can implement the better bit encoding and routing policy, but the original end-to-end objective did not find it. The external search uses full training labels and subsequent explicit target supervision; **it is not spontaneous neural discovery of reuse, an OOD test, or a new Transformer mechanism**. No abstract linguistic structure is tested: worlds are 4-bit configurations. The discrete code landscape is *not* a map of the continuous parameter-space loss landscape. A one-world code change can fail to generalize outside these 16 known worlds.

## Files/reproduction
The fully runnable artifact `COMPASS_INVESTIGATION_09_ENCODING_OPTIMIZATION_2026-10-15.zip` has baseline checkpoint copies, complete code (including frozen `router_reference.py`), results, search/control predictions, full scientific report, and SHA256 manifest. Independently verified ZIP and manifest. Commands:
```sh
python test_exact.py
python landscape.py --exp08 baseline_results.json --out landscape_results.json
python search_landscape.py --exp08 baseline_results.json --out search_results.json
python repair_cache.py --strategy search --exp08-results baseline_results.json --checkpoint-dir baseline_checkpoints --out repair_runs --steps 600
python repair_cache.py --strategy preserve --exp08-results baseline_results.json --checkpoint-dir baseline_checkpoints --out preserve_runs --steps 600
```
No external compute, sealed test data, Nova/Astra branch changes, or merge to main. Do not cite this work as evidence of natural pretrained-model relational reuse.

## Research decision
The original claim that a failure to allocate compute demonstrates a structural absence is **not** supported. The key unknown remains whether pretrained models spontaneously compute and reuse role-resolved relationships through native paths. Next test should examine causal route dependence using known mechanism controls and independent held-out event families.
