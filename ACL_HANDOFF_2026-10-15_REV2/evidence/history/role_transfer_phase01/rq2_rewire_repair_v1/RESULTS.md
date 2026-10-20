# Fresh-graph rewiring repair: completed results

All 96 planned models finished: eight fresh base graph classes, three paired
starts per family, matched sham/high rewiring, and parameter-matched unordered
identity versus ordered-role auxiliary training. Graph selection and analysis
rules were frozen before training in `GRAPH_MANIFEST.json` and `PROTOCOL.json`.
The saved `SUMMARY.json` contains every family and start; checkpoint hashes
passed verification during summarization.

## Prespecified tests

The unordered control retained a high-minus-sham focal actor/patient accuracy
penalty of **−24.45 percentage points** across the eight equal-weight families
(exact two-sided family sign-flip **p = 0.0078125**, graph-level t(7) 95% interval
**[−35.99, −12.92] points**). This passes the frozen first gate.

Ordered-role training reduced that penalty by **+22.59 percentage points**
relative to the unordered control (exact two-sided family sign-flip
**p = 0.0078125**, graph-level t(7) 95% interval **[+11.18, +34.00] points**).
The ordered high-minus-sham penalty was therefore approximately **−1.86
points** on average. All eight family attenuation effects were positive:
**+19.66, +28.06, +40.43, +20.38, +35.61, +6.32, +29.23, and +1.04 points**.
The second fixed scene rendering showed a similar mean attenuation of
**+22.14 points**. The prespecified repair success rule passes.

The smallest familiar-pair four-choice accuracy among ordered-role models was
**99.82%** (frozen floor: 95%). Mean ordered-minus-unordered familiar accuracy
was approximately **−0.001 points**, and mean unseen-event accuracy was
**+0.74 points**, both within the frozen guardrails.

## Interpretation and limits

On fresh controlled graph families, explicitly supervising actor and patient
roles protects transfer to a withheld pair against the matched local rewiring
package. The comparison holds architecture, training examples, number of
updates, auxiliary heads, and auxiliary weight fixed; the control predicts the
same two identities without their actor/patient assignment.

The two weakest repair effects came from families where the unordered control
had relatively little high-versus-sham penalty (−7.23 and −1.43 points). The
mean effect is substantial, but it is not equally large in every family.

The matched rewiring changes other local paths and training co-occurrences
alongside shared-partner overlap. These experiments establish an effect of the
rewiring package and a role-specific repair in this coded-message/generated-
scene Transformer task. They do not identify overlap as the unique cause or
demonstrate an analogous effect in natural-language pretrained LLMs.
