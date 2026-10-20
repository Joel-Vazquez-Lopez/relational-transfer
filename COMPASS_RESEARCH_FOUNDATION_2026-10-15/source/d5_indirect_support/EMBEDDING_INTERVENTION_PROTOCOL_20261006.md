# Exploratory focal-embedding intervention

This was designed after the 30-model screen and 12-model independent query
confirmation had completed. It is an exploratory causal probe, not a test of
the original prospective hypothesis.

For every saved model in both datasets, keep all weights fixed except the
query-channel identity embeddings for entities 0 and 1. Let their original
midpoint be c and difference be d. Replace them by c +/- alpha*d/2 for
alpha = 0, 0.5, 1, 1.5, 2. Alpha 1 is the unchanged model, 0 collapses the
two identities, and 2 doubles their separation without changing their mean.
No gradient updates or held-scene labels are used for the intervention.

Evaluate all five alpha values on the same new 1024 held scenes per model,
with geometry generated from a seed offset not used in prior validation.
Evaluate all 1280 familiar meanings for alpha 1 and alpha 2. The primary
descriptive contrast is alpha 2 minus alpha 1 in held correct-minus-reverse
margin for the models with original held accuracy below 0.8. Report every
model, the count improving, average change, and whether full familiar
accuracy remains >= 0.95. Report high-transfer models as a comparison.

An increase in held performance with familiar accuracy retained would make
embedding separation an intervention candidate, but would not prove that
the original difference in training-graph conditions was mediated by this
single quantity. A failure or familiar degradation would count against a
simple separation-rescue account.
