# EWT error diagnosis while the new blind test awaits review

**Status:** descriptive, post-result analysis of the already evaluated 75 EWT items. It does not use or change the new active/passive challenge. Adapter counts are reproducible in [POSTHOC_ERROR_DIAGNOSTIC.json](POSTHOC_ERROR_DIAGNOSTIC.json) from [the script](../llm_role_ewt_error_analysis.py); the untouched baseline is in its [item-level output](qwen05/EVALUATION.json).

The EWT sample is structurally easy in one crucial respect: the selected subject phrase comes before the object phrase in **all 74 sentences where both source spans can be located literally**. The remaining item has an actor span tokenization mismatch (`K K Paul` versus `KK Paul`), not demonstrated reverse order. Thus a first-participant strategy would get the actor position right on effectively this entire selected sample. The new active/passive test addresses this specific confound by changing the surface position of the agent while retaining the event.

| Checkpoint | Candidate contrast correct /75 | Exact actor /75 | Exact object /75 | Both exact /75 | Same answer to both questions /75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched 0.5B | 57 | 16 | 2 | 0 | 21 |
| Fresh1 source | 56 | 16 | 2 | 0 | 23 |
| Fresh1 natural role training | 75 | 37 | 30 | 15 | 17 |
| Fresh3 source | 56 | 16 | 2 | 0 | 22 |
| Fresh3 natural role training | 74 | 41 | 32 | 19 | 17 |

The direct-answer gain is real for this fixed syntactic task: **15** and **19** items change from not-both-correct to both-correct, with no reverse transitions. Object exact-match gains are **28** and **30** items, also with no losses. Actor exact-match has **3** and **4** losses as well as **24** and **29** gains. The untouched 0.5B baseline is close to the two source checkpoints on the main measures; it was added after the adapted outcomes were known. The two starts share the same items and training examples, so these are paired replications, not independent datasets.

Candidate contrast and answer generation give different pictures. The adapted checkpoints reach 75/75 and 74/75 candidate discrimination, while both-role exact generation remains 15/75 and 19/75. Each adapted model still gives the **same response to both role questions on 17/75 items**, and its object answer repeats the gold actor phrase on **10/75** or **11/75**. The copy-control checkpoint produced 0/75 both-exact answers under its separately run diagnostic. Some strict errors involve an answer that contains the gold span plus extra words, but the repeated-actor errors are not explained by such span-boundary issues.

These results support a narrow conclusion: natural role supervision improves behavior on a new corpus of mostly active subject/object examples, beyond the available copy control, yet it has not established semantic agent/patient transfer or voice robustness. The EWT sample has no reviewed semantic labels and little actor-position variation. The new reviewed active/passive evaluation should use **exact both-role answers** as its decisive measure and report candidate contrast separately.
