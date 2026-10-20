# Second disjoint source review

17 October 2026. Read the [36 source cards](FACTUALITY_SOURCE_CARDS_ROUND2.json) selected from the remaining CT− and PS+ validation candidates, excluding all first-round documents. One nonblinded reviewer, source-only; no model inference. This is not inter-annotator agreement or a random population estimate. The provisional criterion is the same as in [round one](FACTUALITY_SOURCE_REVIEW_2026-10-17.md): locally intelligible event, participant and status scope for *described participant* versus *established occurrence* questions. Temporal bounds must be explicit in any later prompt.

| Card | Label | Decision | Reason |
| ---: | :---: | :---: | --- |
| 00 | CT− | Accept | Turkish reinforcement attempt was destroyed. |
| 01 | CT− | Accept | Tottenham's otherwise-available entry was displaced; counterfactual condition explicit. |
| 02 | CT− | Reject | Landings at other sites occurred; “not exploited” scopes to a different event. |
| 03 | CT− | Reject | Withdrawal threat was later denied; the document does not directly settle whether withdrawal occurred. |
| 04 | CT− | Reject | “No injuries” is clear, but aircraft as Agent of injuries is unsuitable. |
| 05 | CT− | Reject | Bombardment is asserted as having happened despite CT− label. |
| 06 | CT− | Accept | Lukin refused to sign the agreement. |
| 07 | CT− | Reject | “Miami did not report” is a place-as-reporter Agent and mixes reporting with snowfall. |
| 08 | CT− | Reject | Necho's absence is modal (“possible”) and Agent is pronominal. |
| 09 | CT− | Reject | Source says Butler was powerless to restrain the Seneca; the joined Agent points to Brant. |
| 10 | CT− | Reject | “Ceased to exist” is an event termination, not a simple negative occurrence. |
| 11 | CT− | Accept | Referee failed to keep control of the game. |
| 12 | CT− | Accept | Fish chose negotiation rather than war. |
| 13 | CT− | Reject | Singapore venue was changed, but ACC Agent is outside the local sentence. |
| 14 | CT− | Accept | French forces were prevented from crossing at the specified point. |
| 15 | CT− | Accept | Murray's siege failed to retake the castle. |
| 16 | CT− | Accept | Bulldogs did not appear in another Grand Final until 2016; temporal scope must be in prompt. |
| 17 | CT− | Accept | Iraq refused withdrawal by the specified deadline. |
| 18 | PS+ | Reject | Tour “to promote” album is purpose, while the tour itself occurred; promotion outcome not separated. |
| 19 | PS+ | Accept | Manchester United were trying to win a third consecutive title. |
| 20 | PS+ | Accept | British attempt to capture Charleston explicit. |
| 21 | PS+ | Accept | Alleged acquisition of weapons by ARBiH explicit. |
| 22 | PS+ | Reject | Agent is “it” and needs a preceding sentence for an unambiguous antecedent. |
| 23 | PS+ | Accept | Barbarossa tried to resolve the issue diplomatically. |
| 24 | PS+ | Accept | Named party was slated to receive nominated seats; later award not established locally. |
| 25 | PS+ | Accept | British under Lake intended to finish war. |
| 26 | PS+ | Accept | Hindenburg attempted to dock and was destroyed. |
| 27 | PS+ | Accept | Armed forces launched an operation *to restore* control; success not asserted locally. |
| 28 | PS+ | Accept | Savvy scheduled to open but was bumped. |
| 29 | PS+ | Accept | Chilean army's possible entry was conditional on another battle. |
| 30 | PS+ | Accept | Charles intended to attack; Poyntz's later attack is a different Agent. |
| 31 | PS+ | Reject | Liberal Democrats' talks began later in the excerpt; not an unsettled future event. |
| 32 | PS+ | Reject | Trigger “launched” itself is asserted; planned recapture is a different event. |
| 33 | PS+ | Accept | Austria prepared to suppress revolt; suppression is not established in the excerpt. |
| 34 | PS+ | Reject | Intended army positioning and subsequent operations are not cleanly separated. |
| 35 | PS+ | Accept | Continuing invasion explicitly depends on battle outcome. |

**Round two: 22/36 provisional accepts (9 CT−, 13 PS+). Both rounds: 44/72**, across 72 different source documents. These are individual event cards, not 44 valid two-event passages. Some accepted cards are conditional on preserving source perspective or time, so a later pair-construction review may reject them. The original within-document natural-passage gate remains failed. No Qwen result is implied.

The repeated high rejection count is a methodological finding for this project: factuality labels and role annotations refer to different units and can conflict when joined into a new question. Any paper dataset derived from them must expose the source sentence, local referent, factuality scope, and adjudication rationale. It cannot silently treat a join as QA gold.
