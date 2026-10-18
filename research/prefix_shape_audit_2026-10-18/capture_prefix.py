"""Fixed-shape prefix-only measurement helper; never used for answer scoring.

Trailing dummy tokens are causally masked from selected prefix sites. This is
a measurement convention, not a claim of identical full-model execution across
shapes, devices or runtime versions. No model load or download is performed.
"""
import hashlib
import numpy as np


def fixed_shape_prefix_capture(model, prefix_ids, sites, capture_fn, *, total_length=128, padding_id):
    """Return declared-site vectors and input provenance using one fixed shape.

    capture_fn(model, ids, positions) must capture only the requested prefix
    positions. Callers must predeclare total_length and check all inputs fit.
    Never trim a sentence or adapt the shape based on a desired result.
    """
    assert 0 < len(prefix_ids) <= total_length
    positions=list(sites.values())
    assert positions and all(0 <= p < len(prefix_ids) for p in positions)
    ids=list(prefix_ids)+[padding_id]*(total_length-len(prefix_ids))
    states=capture_fn(model,ids,positions)
    return states,{'prefix_length':len(prefix_ids),'total_length':total_length,
                   'sites':sites,'padding_id':padding_id,
                   'input_token_ids_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
                   'warning':'For prefix measurement only; do not score answers on dummy-padded continuations.'}
