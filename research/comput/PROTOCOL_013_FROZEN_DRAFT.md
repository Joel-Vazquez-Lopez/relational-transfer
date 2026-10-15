# COMPUT-013 — Counterfactual discrimination protocol (DRAFT, NOT EXECUTED)

Objective: falsify the original Qwen position-and-question explanation on genuinely new human-reviewed language. Do not select models or heuristics using confirmation responses.

**Historical rival frozen:** Last-mentioned name for generic actor and verb-patient questions, and for passive performed-actor; first-mentioned name for active performed-actor. Name order must be parsed from sentence text, not reference actor/patient metadata. The rival has no rule for unseen constructions and must abstain when unsupported.

**Additional competitors:** Always-first and always-last; independent syntax-based subject/actor-patient parser (pre-specified grammar and failure/abstention policy); format-only and answer-contract controls; optional regularized interaction baseline trained exclusively on development event clusters.

**Design:** Unseen paired names and verbs; full actor reversal and active/passive balance; independently verified meaning-preserving discourse prefix shifts of global mention order; meaning-changing actor/patient reversals while first mention is fixed; separate natural-sentence panel human-reviewed blind to model output. Protect held-out predicate, name-pair and transformation families. Pre-register all exact questions and scoring including non-name responses.

**Primary measures:** paired invariance under meaning-preserving edits, paired equivariance under meaning-changing edits, agreement with frozen shortcut, exact role correctness and output-format compliance. Bootstrap and uncertainty at source-event and group level, never treating transformed prompts as independent events.

**Interpretation:** Strong residual role tracking beyond frozen shortcuts would establish a behavioral capability, NOT a unique internal relational computation. A mechanistic claim would require known-mechanism controls and appropriately targeted causal interventions.

**Execution gates:** (1) exposure and sealed-set screen; (2) independent dual review of semantic labels; (3) freeze designs/analysis; (4) explicit user authorization before substantial inference or paid compute. This file alone does not certify a confirmatory dataset, launch any run, or claim a result.