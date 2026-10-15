# Compass Investigation 11 — Probe-induced computation false positive
Date: 2026-10-15. Status: controlled synthetic calibration, **not** novel pretrained LLM mechanism.

## Central adversarial finding
Fitted high-capacity supervised probes read four final-room labels from three different representations of each of 36 ordered two-swap contexts: (1) pre-question contextual 48d states from trained cache Transformers; (2) raw 192d embeddings from matched trained reconstruction Transformers, *before* any pre-question contextual computation; (3) basic 16d one-hot input tokens. Frozen models from Exp04 were already trained on all 36 contexts, across seeds 19,29,41. External probe is a LayerNorm→128→64→16 MLP; 600 AdamW steps.

When all contexts included in probe training, **all three feature types** achieve 144/144 answer decoding in **each** model seed—even unprocessed one-hot inputs and reconstruction models where composition has not been computed before seeing the question. Such a probe can itself implement the relationship.

Across three different nine-context holdouts for *probe training only* (each total counts over three seed models, 9 contexts × 4 questions × 3 base seeds = 108):

| Heldout probe split | Cache-state readout | Recon raw-token readout | One-hot source readout |
|--|--:|--:|--:|
| 555 | 104/108 | 26/108 | 23/108 |
| 777 | 103/108 | 22/108 | 21/108 |
| 999 | 103/108 | 31/108 | 32/108 |

Cache-state features show more readily transferable decoding under *this* probe protocol, but different architectures, dimensionality and representation training confound the comparison. No native algorithm is thereby located.

**Adversarial mapping control:** Fit arbitrary shuffled world→four-answer mappings. Every externally trained probe, including raw one-hot and recon embeddings, memorizes 144/144 shuffled labels per seed; those arbitrary labels agree with gold in only 42/144 cases. Demonstrates enormous probe flexibility. No extrapolation to heldout shuffled maps is claimed.

## Interpretation
Reject **pre-question probe decodability by itself** as evidence that a native Transformer has already composed relational structure. A probe can learn to compute the algebra from raw tokens. Even high transfer on heldout *probe* items is not heldout for the frozen base model. This is a negative identification control, not a new scientific mechanism.

## Reproduce
Executable code plus frozen Exp04 source/checkpoints, split JSON logs, shuffled-control script, scientific report and SHA256 manifest: `COMPASS_INVESTIGATION_11_PROBE_IDENTIFIABILITY_2026-10-15.zip` delivered in Compass conversation. `blind_probe_calibration.py` also committed to this branch; it imports `learned_mechanism_controls.py` from the artifact. Do not represent the GitHub script alone as standalone without that dependency. CPU PyTorch required.

No sealed data, no other investigator branches, no merge. Model versions and task are synthetic with no pretrained LLM inference.
## Decision
A native-algorithm assay must distinguish an *actually executed causal path* from raw representations amenable to external inference; include reconstructed controls with blinded schedule labels. Until then do not attribute an abstract algorithm to pre-question probe performance.
