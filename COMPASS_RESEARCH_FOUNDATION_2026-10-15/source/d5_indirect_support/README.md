# D5 indirect-support experiment

This is a new, runnable follow-up to D5. The original D5 files and results are not modified.

**Current status:** the first six-pair feasibility study and a fresh six-pair exact-pairing replication are complete. See [feasibility result](FEASIBILITY_RESULT.md) and [matched replication result](MATCHED_RESULT.md). The originally proposed positive direction failed; the reverse-direction pattern replicated with fresh seeds. The second run's source and decision rule were locked in [matched protocol](MATCHED_PROTOCOL.md) before its training started.

## Question

Does the number of shared *training* neighbours of the same withheld relation affect its later functional response, after matching direct exposure, endpoint degree, graph size, and graph-wide structure?

The two conditions are relabelings of one eight-vertex, degree-three graph. In both, relation `(0, 1)` is absent, each endpoint has degree three and participates in one triangle, and the entire graph has four triangles. In `near`, `(0, 1)` has two common neighbours; in `far`, it has none (shortest path three). The graph is two four-node cliques with one edge removed from each and two cross-clique edges added.

This retains D5's oracle message, four-choice scene task, Transformer size, scene physics, and R-family reversal/event foils, adapted to eight identities. It trains the R family on **every observed edge**. Original D5 mixed S and R examples drawn from triangles; copying that sampler here would leave the cross-clique edges untrained. This is a controlled follow-up, not an exact D5 rerun; the exact source copy is in `../d5_original/`.

The proposed primary outcome is the correct-minus-reverse score margin on held-out `(0, 1)` queries, paired near-minus-far across seeds. Held-out four-way accuracy and endpoint query separation are secondary. This is a functional result, not a claim that a particular internal circuit causes the effect.

## Run

Use a Python environment with `torch` and `numpy`. On this machine, the available environment is `/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python`.

```sh
PY=/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python
$PY d5_indirect_support/run.py audit
$PY d5_indirect_support/run.py smoke
$PY d5_indirect_support/run.py train --condition near --seed 7201
$PY d5_indirect_support/run.py train --condition far --seed 7201
$PY d5_indirect_support/run.py summarize
```

The 7201–7206 run is complete. The full run used 6,000 updates of batch size 128 per model and fresh rendered scenes throughout training. `summarize` checks that both members of a pair are present and reports each seed-level contrast. Output is under `d5_indirect_support/results/` and is ignored by Git. Each run saves its model, metrics, and checkpoint hash; existing run directories are never overwritten.

The original six-pair launcher, for use only from an empty `results/` folder, is:

```sh
PYTHON_BIN=/opt/homebrew/Caskroom/miniforge/base/envs/ai-fingerprint/bin/python bash d5_indirect_support/run_pairs.sh
```

The exact-pairing replication used `matched.py` and `run_matched_pairs.sh`, with independent seeds 8201–8206. Verify either finished study with `analyze.py` or `analyze.py --matched`. Do not rerun these scripts against the populated result directories.

## Interpretation gate

The experiment is interpretable only if both conditions attain at least 95% familiar four-way accuracy on independently rendered familiar examples at the fixed final checkpoint. Do not select another checkpoint after seeing the held-out scores. A positive primary contrast means *indirect comparison structure* changed a held-out decision in this model. A null contrast with reliable familiar learning limits that hypothesis. Familiar failure leaves the question unresolved in this apparatus.

The first run used the internal lock in `PROTOCOL.md`. The second used `MATCHED_PROTOCOL.md` after the first result was known and before its fresh seeds were trained. Neither is a public preregistration or evidence for a general law across graph families.
