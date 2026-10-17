# Scope before readout diagnostic

17 October 2026. Exploratory reanalysis of existing f07 states, selected after behavioral results because it was the only family competent in the three primary models. The active/passive cosine pattern was already observed before writing this file: late answer-site response cosines average approximately -0.736 Qwen 0.5B, +0.447 Qwen 1.5B, +0.712 Qwen 7B and +0.428 Llama 3B. The event-end responses remain opposed. These observations are not preregistered confirmation.

For each layer/site define d_active = h(A-Agent, active) - h(B-Agent, active), and d_passive with the same Agent orientation. Compute their cosine. Norm-only lines cannot determine this angle. Positive cosine can be caused by a common answer identity direction as well as a more general relation representation.

The next saved-state diagnostic applies each model's final RMSNorm to the captured states, extracts the native first-token Miriam-minus-Adrian output-weight axis g, projects both response vectors off the *same* g, and recomputes cosine. Verify the selected weights against the native output head. Compare with 256 fixed-seed random rank-one removals, and report removed squared norm. No new sequence inference, causal patching or held-out evaluation is part of this audit.

If removal reduces alignment substantially, this supports a simple direct-readout contribution. If alignment survives, it rules out only that single axis as the full explanation; distributed answer coding, lexical/syntax interactions, and generic processing remain. No result of this diagnostic establishes semantic-role abstraction, causal use, family effects, or novelty. The earlier failed gates stay failed.
