# Targeted implementation and component diagnostics

Designed after the crossed evaluation audit, using world new2, seeds 10402
and 10403, both conditions. These are selected failure/control examples,
not independent replication of a population effect.

Train two variants for all four settings, at the original 6000 updates:

1. `padding`: give the constant fourth message token a dedicated embedding.
   Initialize every existing parameter exactly as in the original run. Add
   one new embedding row drawn from the original initialization distribution
   with an independent random generator. Preserve original training episodes.
   This tests sensitivity to the original entity-0/constant-token alias.
2. `symbolic`: retain the original message encoder and input coding, but
   replace the learned scene encoder with fixed vectors encoding each
   candidate's source, target, and event in separate slots. Candidate contents
   are provided uniformly; no answer indicator is supplied. The candidate
   geometry and learning problem change, so success narrows the investigation
   but does not prove that the visual encoder alone caused the original error.

Report all four settings per variant, complete familiar evaluation, and the
new crossed evaluation. Save intermediate and final checkpoints. A familiar
accuracy below .95 makes a transfer comparison inconclusive. If dedicated
padding changes the failure substantially, investigate that dependency before
expanding graph or language experiments. Do not assume a failed embedding
expansion established a query-only mechanism.
