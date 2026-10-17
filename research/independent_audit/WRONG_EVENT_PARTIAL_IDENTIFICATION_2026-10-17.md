# Wrong-event provenance: independent offline sensitivity audit

This is an executed offline analysis of *reported aggregates*, not independent raw-output verification. Reported Qwen7B loaded condition: 57/64 correct, seven errors all target the second event and copy the first event's participant. Original preregistered eight-error gate FAILED.

**Identifiability:** On every observed error, 'copy the other event' and 'fall back to the first event' are observationally identical. Zero observed first-target errors distinguish them. A semantic binding claim is not identified.

**Exact sensitivity:** Assuming 32 first-target and 32 second-target questions (NOT independently verified), C(32,7)/C(64,7) = 0.005418171714365101. With 24, 28, 36, 40 or 48 second-target questions, the same conditional probability is respectively 0.000557, 0.001906, 0.013438, 0.030011, 0.118524. These are descriptive conditional calculations, not confirmatory p-values: balance, cluster dependence and post-hoc hypothesis choice are unresolved. If 7 distinct matched worlds each show only a second-target error, a one-sided sign randomization yields 2^-7 = 0.0078125 under strong exchangeability; the actual world count is unknown.

**Local execution:** Python standard-library exact calculations were run. Separately, an offline provenance-scoring prototype was implemented and seven synthetic fixture unit tests passed; it is not a validated scorer for the original Qwen outputs, and its source strings are not in GitHub yet.

**Next:** Obtain the frozen original raw outputs, question/event order, per-world IDs, alias policy, original preregistration and permission for analysis. Do not run new LLM inference. Preserve the original failed gate. Strong query-aware joint syntax/lexical/position/salience retrieval is a mandatory rival; if no heldout prediction differs, stop.