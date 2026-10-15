# Compass Investigation 05 — Latent transport is not an algorithm

**Date:** 2026-10-15 · **Status:** exploratory controlled-model experiment; not a pretrained LLM discovery.

## Question
Does a learned Transformer state support consistent, transferable relational index transformations, and does this establish natural reuse of relational computation?

## Design
Reuse three previously trained 48-dimensional causal-attention cache models and three question-time reconstruction models from Investigation 04. The base model weights were trained on **all 36 ordered two-swap contexts**. New *affine transformation maps* were fitted on 27 context states and tested on the other nine. This is held-out map fitting **not unseen relational-task generalization**. Normalize unordered pairs after entity relabeling; earlier noncanonical data were discarded.

## Results
- At raw ridge alpha=10, pre-question cached states recover **257/324** transformed room answers (79.3%) across three seeds and three relabelings; random prefix features with a post-hoc fitted, perfectly memorizing answer decoder recover **56/324** (17.3%).
- Raw ridge hyperparameter sweep was exploratory and reveals regularization sensitivity. More comparable alpha normalized to source state variance at lambda=0.01 gives **262/324 (80.9%)** for trained cached states and **103/324 (31.8%)** for random prefix features.
- Composed latent operations are imperfect: (01) squared gets **102/108** answers; (01)→(12) gets **85/108**; commuting paths agree on **75/108** answers; braid-equivalent three-step paths agree on **74/108**. These counts re-evaluate the **same nine configurations per seed**; do not treat as independent samples.
- Known question-time reconstruction hidden states, when an edit is fitted only for the room-0 question, transfer to **118/324** edited answers across rooms; same-question map fitting improves to **198/324**. This is **not causal evidence of scheduling differences**, because architectures, training, and representations differ.
- The random-feature control's decoder memorizes **432/432** original answers across seeds; answer competence on finite contexts is not enough to ensure comparable hidden-state geometry.

## Interpretation
Trained whole-world cached states admit externally fitted affine semantic relabeling better than the tested random representations. Some edits compose approximately. This is a **representational assay**, **not a discovered naturally executed operation or a proof of pre-question reuse**. The task and caching readout were explicitly architected; all base models had already seen all worlds. Regularization, random geometry and decoder confounds remain.

## Falsification / next decision
A future shared-architecture model with both routes available must be competence matched on genuinely unseen compositions. Preregister two independent cuts: cached-state access with raw input preserved, and raw-input access with cache preserved, and verify the cuts on known-mechanism controls. If external edits show equivariance even in known reconstructor controls, reject equivariance as a reuse identifier.

## Reproducibility
Full source, six original frozen checkpoint copies, all evaluation JSON, regularization sweeps, SHA256 manifest, and detailed report are in user conversation artifact `COMPASS_INVESTIGATION_05_LATENT_ALGEBRA_2026-10-15.zip` (not copied into GitHub). Scripts: `latent_group_test.py`, `group_laws.py`, `reconstructor_control.py`, `random_feature_control.py`, `variance_matched_control.py`. No sealed data used, no other workstream changed, and no merge.
