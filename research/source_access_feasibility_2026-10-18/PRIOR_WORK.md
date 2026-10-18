# Prior work checked before execution

This is a local feasibility diagnostic, not invention of attention knockout or staged retrieval. Full primary method passages were inspected before target outcomes:

1. Geva et al. (2023), Dissecting Recall of Factual Associations in Auto-Regressive Language Models, section 5 and limitations. It blocks attention from the final position to source groups in layer windows, examines factual recall, and explicitly discusses information leakage. Our cut closes event-context access to all suffix positions from a boundary onward, preserving earlier transfer. This operational difference alone establishes no novelty. https://aclanthology.org/2023.emnlp-main.751.pdf
2. Gur-Arieh, Geva and Geiger, Mixing Mechanisms, appendix F / Figure 23. This is particularly close: blocking access to a target entity distinguishes a retrieval pointer from an already retrieved answer in an interchange experiment. The proposed linguistic demand contrast does not supersede that mechanism. https://arxiv.org/html/2510.06182v2#A6
3. Prakash et al., Language Models Use Lookbacks to Track Beliefs, appendix K. It knocks out all heads/layers from a sentence to earlier sentences, then reopens individual layers to study information flow. Restoration and repeated retrieval are already studied. https://arxiv.org/html/2505.14685v3
4. Dai et al. (2024), Representational Analysis of Binding in Language Models, establishes low-rank binding representations and retrieval-changing activation edits. This study neither supplies such a subspace nor claims a new binding mechanism. https://aclanthology.org/2024.emnlp-main.967/

The possible useful observation is a narrowly prospective query-specific dependence profile that predicts fresh linguistic cases and a second local model. It may fail or duplicate a known pattern. An eventual publication claim would require a result-specific, broader literature comparison and independent replication. No novelty is asserted now.

Local overlap: earlier role-path final-state patches and period-token syntax transport did not run this all-suffix source-access cut. Component-reuse patches were stopped by gates; the latest Atlas was descriptive. The new study preserves all those failures and does not reactivate their unused budgets.
