# Compass Investigation 18 — Mixed-layout counterfactual falsification

The paired training and evaluator run against two orderings of the same world:
`[A/B table, start, three instructions, ask]` or `[start, three instructions, A/B table, ask]`.

**Predeclared** `runs/frozen_protocol.json` identifies the generated train/test split and evaluation constraints. Everything is open synthetic development, not confirmation or a pretrained model test. **Reversing order changes both recency and causal visibility**: this is not an isolated position manipulation.

Quick replay from bundled saved checkpoints (Python + PyTorch):

```bash
python -m unittest -v test_exp18
python replay_checkpoints.py --root runs
python analyze_pairs.py --root runs
python shortcut_controls.py
```

Training from scratch, if desired:

```bash
python exp18.py --out rerun --steps 1600 --seeds 19 29 41
```

The script saves the frozen protocol before training. `replay_checkpoints.py` re-executes evaluation from saved weights and checks every saved matched prediction; `analyze_pairs.py` recomputes pairwise contrasts and conditional world bootstrap. Refer to `RESEARCH_REPORT.md` for the negative controls and limitations.
GitHub also contains `standalone_falsification.py`, a checkpoint-free executable negative-control certificate. All neural checkpoint sources and outputs are in the verified Compass conversation archive, not stored in GitHub.
