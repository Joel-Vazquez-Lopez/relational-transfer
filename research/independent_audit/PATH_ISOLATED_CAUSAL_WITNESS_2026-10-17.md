# Path-isolated causal witness — executed synthetic graph test (2026-10-17)

## Hypothesis
Two architectures can be observationally equivalent while differing under a *surgically isolated* mediator intervention: a shared event address Z versus role-specific Z_A and Z_P. Does patching a common input suffice? No.

## Execution
Ran `python /mnt/data/relational_competition/path_intervention.py` locally with NumPy, seed 20261017, 100,000 independent normal draws each for input and donor. Binary event selection `Z=1[x>0]`. Both systems answer Agent and Patient using the same correct role-indexed event table; there is no role corruption.

- Baseline shared and separate architectures: observational agreement 100%.
- Shared architecture, replace Z with donor Z: both answers change in **49,987/100,000 = 49.987%**.
- Separate architecture, replace only Z_A with donor Z_A while preserving Z_P: both answers change in **0/100,000**; Agent-only changes in **49,987/100,000**.
- Separate architecture, replace *common upstream input* X with donor X: both answers change in **49,987/100,000**.

These percentages are a property of independent symmetric normal threshold draws, not fitted Transformer effect sizes.

## Full reproducible script
```python
import numpy as np
rng=np.random.default_rng(20261017)
n=100000
x=rng.normal(size=n)
donor=rng.normal(size=n)
base=x>0
shared_agent=donor>0
shared_patient=donor>0
separate_agent=donor>0
separate_patient=base
upstream_agent=donor>0
upstream_patient=donor>0
print('observational agreement',np.mean(base==base))
print('shared mediator both changed',np.mean((shared_agent!=base)&(shared_patient!=base)))
print('separate agent-mediator both changed',np.mean((separate_agent!=base)&(separate_patient!=base)))
print('separate agent-mediator agent-only changed',np.mean((separate_agent!=base)&(separate_patient==base)))
print('separate common-input both changed',np.mean((upstream_agent!=base)&(upstream_patient!=base)))
assert np.all(separate_patient==base)
```

## Interpretation and limits
This is a **graph-specific causal identifiability witness** with privileged access to actual mediators. It does not discover a mediator in a Transformer. It does not rule out a joint query-aware copier or readers sharing another upstream latent. A whole-residual-stream, common-prefix, or source-token patch generally alters multiple paths and cannot be interpreted as a surgically isolated Z intervention. A model-specific test would require identifying a candidate Z independently, checking path completeness, within-role selectivity, bypass routes, donor/recipient alignment, sham and off-manifold controls, and a preregistered rival prediction. No pretrained inference was performed. Consistent with Armaggedon FRONTIER-003 and #31 no-go gate.

## Research decision
The shared-vs-separate-reader distinction is theoretically identifiable *under privileged isolated intervention*, but current available evidence does not establish such isolation. Do not launch a new model intervention on this basis alone.
