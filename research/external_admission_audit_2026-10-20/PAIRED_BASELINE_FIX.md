# A concrete correction for review

This concerns `resid_act_patch_pca_ld` in zero-based cell 23 of the official Dai et al. ACL software notebook, SHA256 recorded in `BASELINE_CERTIFICATE.json`. It is a local proposed correction, not an upstream change and not a tested Transformer rerun.

Before the outer intervention-step loop, initialize a baseline table:

```python
baseline_logits_by_case = {}
```

The existing code resets `datai = 0` at each intervention step and increments it once per example. Replace its single cached-vector subtraction with:

```python
if pi == 0:
    baseline_logits_by_case[datai] = logit6.copy()
logit_diff = logit6 - baseline_logits_by_case[datai]
```

This uses the same example order already present in the notebook. A more robust implementation should use explicit stable example IDs and assert that the intervention steps contain the same IDs and candidate-token ordering. Store every example's unmodified and intervened candidate logits, native vocabulary-wide choice, target, and candidate-token IDs. Preserve actual native choice separately from argmax restricted to the seven options.

`baseline_certificate.py` checks the original source structure without executing it. Its authored arithmetic fixtures establish the expected paired calculation, the mean-bias identity and the affected/unaffected summaries. They do not validate model hooks, establish the size of the error in the published runs, or show that the released notebook produced the final paper figures. Those require the authors' per-example outputs or a separately authorized faithful rerun. No external issue, email or other message has been sent to the authors.
