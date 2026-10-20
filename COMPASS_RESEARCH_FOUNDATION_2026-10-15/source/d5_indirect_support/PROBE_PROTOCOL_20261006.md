# Diagnostic familiar-trained role decoders

Added after the crossed audit and the first symbolic-candidate results.
For all original 42 models, extract query and scene representations from the
1280 original familiar training configurations, with new scene geometry.
Fit separate linear ridge decoders of source identity, target identity,
and event, using only familiar labels. Deduplicate query meanings (80 inputs)
so repeated configurations do not count as independent training queries.

Use centered features and targets; regularization is mean squared error
plus lambda times squared weights. Lambda .001 is primary; .0001 and .01
are sensitivity checks, not a model selection grid. No held labels choose
lambda. Score exact source/target recovery and event recovery on the crossed
controlled held scenes. Report training fit as an operational check; it is
not an independent familiar-validation result.

This is supervised representational diagnosis, not evidence that the native
model uses a decoded feature causally. If both familiar-trained decoders
recover held roles but the native match fails, an alignment account becomes
plausible. If a decoder fails, this limits linear decoding under this training
distribution and does not prove the representation contains no role information.
