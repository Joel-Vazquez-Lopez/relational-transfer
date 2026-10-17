# Independent exact audit of reported Qwen7B wrong-event provenance (2026-10-17)

## Source and provenance
Armaggedon #33 reports corrected frozen owner-run Qwen7B counts: 32/32 isolated, 64/64 plain paired, 57/64 with named bystanders, with seven errors **all targeting event 2** and borrowing event 1's participant. The experiment's predeclared eight-error gate **FAILED**. Raw prompts and outputs were not independently received. This is a reanalysis of **reported aggregate counts**, not a raw-data replication.

## Executed exact aggregate calculation
Local stdlib Python `/mnt/data/relational_competition/error_provenance.py`, assuming the 64 loaded questions contain exactly 32 first-target and 32 second-target queries (this balance MUST be verified from the frozen manifest). Under the null that seven errors are uniformly distributed across these 64 positions, conditional probability all seven are second-target:

```python
from math import comb
print(comb(32,7)/comb(64,7))
# 0.005418171714...
```

This is the one-sided conditional Fisher/hypergeometric tail for observed 0/32 first-target vs 7/32 second-target errors **under that balance assumption**. It is not a confirmatory p-value: the aggregate was inspected and the specific directional hypothesis may be post hoc, observations share templates/events, and the preregistered threshold was missed. Dependence between questions could greatly change uncertainty. Jeffreys binomial 95% marginal intervals (treating questions as independent, for illustration only): loaded 7/64 error [0.0503, 0.2028], plain 0/64 [0, 0.0383], second 7/32 [0.1036, 0.3822], first 0/32 [0, 0.0749]. These are NOT world-clustered intervals.

## Mechanistic identifiability
Define rival M_event: if confused, borrow requested-role participant from the **other event**. Define M_first: if overloaded, fall back to the **first event**. On a second-target query, both produce the first-event participant; on a first-target query, M_event predicts a second-event error while M_first predicts no error. Because **zero first-target errors** were reported, there are **zero observed error-source cases separating these hypotheses**. This is stronger than saying the seven errors look coherent: their direction is exactly confounded with sentence position. A joint syntax+position+lexical retrieval rival remains mandatory.

## Recommended next step (NO new inference)
Obtain the original 64-item frozen manifest/raw outputs and verify 32/32 target balance, distinct event/world clusters, preregistration and exact answer strings; compute paired within-world error tables and cluster bootstrap. Test whether error-source=other-event remains after crossing target-event order and semantic event identity, and whether any first-target errors point to event 2. If no divergent heldout prediction versus a competent joint retrieval copier, STOP mechanism inference under #31. No model calls, paid inference, source downloads or sealed data touched.

## Reproduction
```python
from math import comb
assert abs(comb(32,7)/comb(64,7) - 0.005418171714) < 1e-12
```
