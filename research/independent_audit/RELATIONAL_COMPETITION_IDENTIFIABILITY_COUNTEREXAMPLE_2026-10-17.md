# Relational Competition Dynamics — executed identifiability counterexample (17 Oct 2026)

## Scientific question
Does correlated Agent/Patient wrong-event borrowing identify a shared internal event-address mediator?

## Executed experiment
Local Python/NumPy, fixed RNG seed 20261017, N=100,000 paired samples. A correct two-record role table is never altered. An event-score margin determines whether the answer uses the correct or competing event. Compared:
1. Shared event-selector feeding two role lookups.
2. Two independently noisy role-specific selectors with matched marginal error rate.
3. Two **separate deterministic** role readers, each recomputing the identical score from the same input, with no shared latent event-address node.

At distractor-load 1, wrong-event selection = **18,272/100,000 (18.272%)**. Shared-selector Agent/Patient error disagreement = **0%**; independent-noise readers = **29.75%** (theory 2p(1-p) = 29.8667%); separate deterministic readers = **0%**. Hence perfect role-error correlation cannot distinguish shared latent address from deterministic independent readers with common input.

For the illustrative monotone load model, errors rise as load increases: 0→7.796%, 0.5→12.240%, 1→18.272%, 1.5→25.916%, 2→34.844%. Paired 0→0.5 yields 4,444 new errors, zero repairs; 0.5→1 yields 6,032 new errors, zero repairs; 1→1.5 yields 7,644 new errors, zero repairs. This monotonicity is **an assumption of the constructed score**, not an empirical property of LLMs. Real retrieval systems can be nonmonotonic.

## Formal identifiability argument
Shared graph: input X→Z(X)→(Agent,Patient). Independent-reader graph: X→Z_A(X)→Agent and X→Z_P(X)→Patient, with Z_A(X)=Z_P(X) as deterministic functions. Both induce the same observable joint answers for every X, despite distinct internal graphs. Any observational test based only on role-answer agreement, even across distractor loads, is insufficient to identify the shared-mediator graph. A conditional causal/path intervention with valid isolation is required; a whole-state or common-prefix patch is also not enough if it modifies both readers' inputs.

## Reproduction
Local script: `/mnt/data/relational_competition/experiment.py` (container-local, not accessible from GitHub). Exact core:
```python
import numpy as np
rng=np.random.default_rng(20261017); N=100000
base=rng.normal(0,1,N); noise=rng.normal(0,1,N)
margin=1.5-base-0.35*noise
p=np.mean(margin<.55)
shared=(margin<.55)
ind_a=rng.random(N)<p; ind_p=rng.random(N)<p
print(p,np.mean(ind_a!=ind_p),2*p*(1-p))
assert np.array_equal(shared,(margin<.55))
```

## Interpretation and next action
This is an **executed synthetic non-identifiability witness**, not a pretrained-model experiment or a novel mechanism claim. It independently reinforces Armaggedon's FRONTIER-003 warning that shared-prefix interventions cannot distinguish shared event selection from independently reconstructed event identity. It does not demonstrate that either architecture is used by Qwen. Scientific decision: **do not use coordinated Agent/Patient wrong-event answers as positive evidence of a shared latent bottleneck**. Before new inference, audit existing frozen raw outputs and define a path-isolated test with a genuine divergent prediction versus independent readers and a joint query-aware copier. Do not duplicate #34's failed transport patches or #35's trajectory study. No new model calls, source access, paid inference or test exposure.
