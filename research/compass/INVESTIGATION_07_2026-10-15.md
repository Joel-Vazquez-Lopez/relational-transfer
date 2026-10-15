# Compass Investigation 07 — Optimal cache versus fallback (15 October 2026)

## Exact finite-world result
Four independent equally likely binary relations, 16 worlds, 4 equally likely questions, one question-independent binary cache bit, optional perfect reconstruction from source. A wrong answer costs 1; a reconstruction call costs c. The route decision is a function only of the cache message and question (not hidden ground truth). Exhaustive search across all **65,536 deterministic cache encoders**, with optimal decisions for each cache-message/question cell, yields the exact **total cost over 64 equally weighted world/question cases**:

```
L*(c) = min(48 c, 12 + 16 c, 20),  for 0 <= c <= 1.
```

Proof certificate: exact integer exhaustive minima at c=0,3/8,1/2,1; each encoder's optimal cost is concave in c, so these endpoint lower bounds extend across each corresponding linear interval. Three fixed witness encoders (IDs 255, 831, 279) attain the three lines across their intervals; source provided.

## Three regimes
- **0 < c < 3/8:** 0 errors, 48/64 reconstruction calls; cache encodes a perfectly reliable single relation.
- **3/8 < c < 1/2:** 12 errors, 16/64 reconstructions; compresses/guesses for most questions.
- **c > 1/2:** 20 errors, no reconstruction; uses accuracy-optimized cache alone.
- At equality thresholds multiple policies tie; do not claim unique mechanisms.
- c=0 permits free fallback on every question, which is optimal (and many encoders tie).

## Scientific interpretation
Even in this tiny *exact* case, optimizing answer error alone and optimizing compute-aware relational reuse lead to different caches and gating strategies; an intermediate mixed policy can dominate both extremes for intermediate fallback cost. This is a classic finite selective prediction/source coding tradeoff, **not a new general theorem** and **not observed in a pretrained Transformer**. The fallback has perfect accuracy by assumption; measuring realistic fallback errors/latency is necessary.

## What should come next
Test a **single shared-backbone trained model** with both routes and a predeclared cost/error trade-off, then compare against finite oracle and simple heuristic routing. Use withheld worlds/structures and route-specific causal cuts. Require learned-policy comparisons to use genuine compute costs and abstention calibration. Substantial training or external API spend needs coordinator approval.

## Reproduction
`research/compass/exp07/one_bit_optimal_router_certificate.py` (NumPy) prints PASS and checks the exact formula via exhaustive encoders and endpoint/concavity argument. Full local scratch files are `/mnt/data/compass_exp06/oracle_routing_frontier.py`, `ORACLE_ROUTE_FRONTIER.json` and `ORACLE_ROUTE_CERTIFICATE.json`. No sealed data, no model training, no other branches touched.
