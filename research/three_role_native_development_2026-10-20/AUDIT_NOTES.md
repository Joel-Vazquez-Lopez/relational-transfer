# Independent replay arithmetic note

The first postcollection audit stopped at its own 1e-10 log-probability comparison tolerance. The production scorer uses NumPy's sequential logaddexp reduction across152,064 logits; the independent implementation uses a max-centered exponential sum. Across all324 candidate name probabilities, their maximum discrepancy was1.0870849359578827e-10 nats, with two values above1e-10. All108 classification signs agreed; the smallest absolute recorded margin was0.09375 nats.

The independent arithmetic comparison uses1e-8, retaining exact native token/EOS decisions and all original technical/scientific gates. The frozen collection, score definitions, original results and eligibility decisions were not changed, and no model was rerun. The original audit stop and correction are disclosed here. The independent replay script itself is postcollection verification, not frozen outcome analysis.
