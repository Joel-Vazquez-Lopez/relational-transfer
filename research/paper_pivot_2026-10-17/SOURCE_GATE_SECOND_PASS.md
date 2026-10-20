# Source-only follow-up: remaining candidate documents

17 October 2026. One investigator read the shortest candidate pair in each of the **12 eligible documents not selected** in the seeded 12-document sample. These are targeted feasibility judgments, not independent adjudication and not a model result. All passages and gold entries are in [SOURCE_CANDIDATE_CENSUS.json](SOURCE_CANDIDATE_CENSUS.json); the eligibility rule and first sample are in [source_gate_audit.py](source_gate_audit.py).

| Document | ID prefix | Judgment | Reason |
| --- | --- | --- | --- |
| Battle of Chongju (1950) | `6f5d7723` | Reject | `Yalu River` is the destination of an offensive, not the attacked Patient. |
| Invasion of Poland | `e485c920` | Conditional | Two real invasions, but `Saar` is only indirectly named as the first target. |
| Invasion of Yugoslavia | `e85b4a73` | Conditional | Two invasion events, but an additional unmarked invasion of Greece and ellipsis around `German-led` complicate the second Agent. |
| War of the Three Henrys | `70d6a5a1` | Reject | First annotated Agent/Patient reverse the sentence's victor and loser; second invasion is described as prevented. |
| Fall of Constantinople | `ded6063e` | Reject | Capture and defeat describe the same historical event at different granularity; participant referents overlap. |
| Narva Offensive (1–4 March 1944) | `c85995f3` | Reject | `Taking Estonia` is a desired goal, not an asserted completed conquest. |
| Taboo Tuesday (2005) | `a263e914` | Reject | First defeat has two named opponents but only one annotated Patient. |
| Action of 10 November 1808 | `8bd3b56b` | Reject | Specific naval engagement and broader war/blockade do not have comparable Agent/Patient records. |
| Action of 18 June 1793 | `79f23b41` | Reject | The two triggers describe the same encounter at different levels, not distinct events. |
| Battle of Krosno | `c2061768` | Reject | First Agent is annotated as Swedish forces despite the text saying Commonwealth forces defeated them. |
| Battle of P'ohang-dong | `e7ddbf1d` | Reject | First and second triggers refer to the same battle, with participant aliases across the pair. |
| Korean War | `9ada9ec0` | Reject | Korean border clashes and the Cold War differ in event scope and roles. |

Across the **24 shortest-per-document pairs** in the eligible old census, this review has **0 unambiguous accepts, 5 conditionals and 19 rejects**. This describes those selected pairs only. Eleven additional candidate pairs occur in already-reviewed documents and could be better than the shortest pair; the census itself is not exhausted by this review. A targeted look at an alternate `Use_firearm` pair in *Air engagements of the Gulf War* is **conditional**: one Iraqi pilot fires at a US plane and US planes later shoot down Iraqi planes, but `fired` versus `shot down` and the meaning of Patient need independent adjudication. It is not promoted to gold here.

The immediate issue is **measurement validity**, not model competence. No held-out causal run should use these labels as automatic four-cell truth. The next source task is an independently reviewed, manually selected set across MAVEN-Arg and WikiEvents or a new annotation layer. A small convenience set from these conditionals is suitable only for transparent development, not the planned paper test.
