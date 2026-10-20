# Factuality–participant source gate, first review

17 October 2026. **No model inference.** This is a single-reader review of 36 deterministic, stratified candidate snippets from public MAVEN-FACT validation joined to MAVEN-Arg validation. The [generator](build_factuality_source_cards.py) saves the [full cards, IDs, offsets, and exact snippets](FACTUALITY_SOURCE_CARDS.json). Selection: one mention, CT− or PS+, one annotated Agent, nearest Agent mention within 80 characters, one selected card per document, 18 per status. Seed `20261017`. This is an enriched feasibility sample, not a random estimate of corpus quality. The reviewer saw the source label and candidate role; an independent blinded review is still required.

**Operational provisional accept:** the excerpt itself makes the event trigger, Agent referent, and factuality scope clear enough for a *described-event participant* question and an *established-occurrence* question. “Established occurrence” is deliberately different from “definitely did not happen”: a PS+ claim may leave occurrence unresolved. Provisional accept does **not** certify a matched distractor or transformation.

| Card | Label | Decision | Reason |
| ---: | :---: | :---: | --- |
| 00 | CT− | Accept | German forces' lack of response is explicit. |
| 01 | CT− | Accept | Australia had not lost a Test in the stated period. Time scope must remain in the question. |
| 02 | CT− | Accept | US had not entered the war as of August 1941. Time scope must remain. |
| 03 | CT− | Accept | Byzantines were unable to extract more assistance. |
| 04 | CT− | Accept | Kiplagat was not present to defend title. |
| 05 | CT− | Reject | ETA's Agent link to the four bombs with no injuries is not explicit in the excerpt. |
| 06 | CT− | Accept | NATO's authorization attempt was opposed; later action without authorization confirms failure. |
| 07 | CT− | Reject | Exception to transferred powers does not make the royal government's Agent role clean. |
| 08 | CT− | Accept | Gates “could have tried” to understand Crowley's view; counterfactual scope clear. |
| 09 | CT− | Reject | “No changes were made” has no local Agent; IOC merely stated the program would be the same. |
| 10 | CT− | Reject | “Did not sail until” describes delayed occurrence; a binary nonoccurrence question would be wrong. |
| 11 | CT− | Reject | Failed rescue is clear, but the Agent link to Israel is nonlocal in this excerpt. |
| 12 | CT− | Accept | Johnson decided not to take retaliatory action. |
| 13 | CT− | Reject | “He then travelled to Germany” asserts occurrence despite the source CT− label. |
| 14 | CT− | Reject | Cahill was passed fit to start; source CT− label does not support a clean negative reading. |
| 15 | CT− | Accept | Finland refused to cede territory. |
| 16 | CT− | Accept | Nadal failed to recover from injury before withdrawal. |
| 17 | CT− | Reject | He eventually received the medal; the negative applies only to “on the day” and the Agent is pronominal. |
| 18 | PS+ | Reject | Raglan intended to send cavalry *to prevent* removal; Agent of the prevention itself is ambiguous. |
| 19 | PS+ | Accept | Levine considered cancelling; the later decision to proceed resolves occurrence. |
| 20 | PS+ | Reject | Probable preference is an uncertain mental state, poorly matched to participant retrieval. |
| 21 | PS+ | Accept | Whitlam's attempted deadlock-breaking is locally anchored; use named Agent in prompt, not “he.” |
| 22 | PS+ | Accept | Hope that James would address grievances is explicit; named antecedent must be preserved. |
| 23 | PS+ | Reject | Estimated cost is not a clear Agent-action event for this retrieval question. |
| 24 | PS+ | Reject | “Forced Vicious to leave” may assert actual departure despite PS+ label. |
| 25 | PS+ | Accept | Department's claimed preparation is explicit; occurrence is attributed, not independently established. |
| 26 | PS+ | Accept | Obtaining victory is embedded in a conditional prediction; Germans are locally named. |
| 27 | PS+ | Reject | “Forcing him into hiding” is described as a consequence; status source and Agent role are not clean. |
| 28 | PS+ | Accept | Linois attempted to drive off pursuer without success. |
| 29 | PS+ | Accept | French army weapons test is explicitly alleged. |
| 30 | PS+ | Accept | Digicel argued it should get rights; this is a claim, not established acquisition. |
| 31 | PS+ | Accept | Durham Light Infantry attempted to recapture bridge; later securing is a different event. |
| 32 | PS+ | Accept | US Army tried to dislodge Modoc; Agent and attempt are explicit. |
| 33 | PS+ | Reject | “Scheduled to take place” could be followed by actual occurrence; snippet does not establish the final status. |
| 34 | PS+ | Accept | US 12th Army Group could capture Germans; anticipated possibility is explicit. |
| 35 | PS+ | Accept | Petliura tried to strengthen the republic; Agent and attempt are explicit. |

## Result and consequence

**22/36 provisional accepts: 10/18 CT− and 12/18 PS+.** The 14 rejections include two apparent label/text conflicts (13, 14), temporal-scope traps (10, 17), Agent linkage failures (05, 07, 09, 11, 18, 27), and event-status ambiguity (20, 23, 24, 33). These categories overlap and are review judgments, not independently verified dataset errors. The acceptance count does **not** meet the planned 30-source-valid, multi-event document gate: these are individual event cards, and many lack any suitable distractor.

The first responsible next step is a second independent review of a new source sample and a targeted distractor census, or creation of a small annotated corpus of paired authentic sentences. **Do not run a causal patch on these 22 as if they were held-out gold.** The immediate surviving question could support a behavioral pilot with explicit status and role questions, but an internal-factorization paper still needs a predeclared causal witness against a syntax-aware copying rival.

## Targeted same-document distractor census

Among the 22 provisional accepts, a source-only scan found only **two cards** whose document also contains a CT+ event of the same MAVEN type with a different annotated Agent named within 120 characters of its trigger: card 01 (`1948 Ashes series`, Australia “had not lost” versus England “lose”) and card 31 (`Operation Fustian`, Durham Light Infantry “attempted to recapture” versus British paratroops “captured”). This is a structural count, not two certified contrasts. The Ashes events have a time-scope and long-distance confound; the Fustian mentions concern the same bridge at different stages and may be related episodes. Thus the joined validation split does **not** presently support the intended natural *same-document, same-type distractor* design at useful scale.

The honest redesign is to assemble **paired authentic source sentences** from different documents, preserve each sentence's wording, and explicitly label the combination as experimenter-constructed. Cross event order, status, participant names, and question type; reserve fully authentic documents as an external-validity test. This would be a controlled experiment grounded in real language, not a claim that each paired passage is an untouched source paragraph. It still needs independent semantic review and a frozen causal prediction certificate before any intervention.

A final source-only census checked whether each of the 22 provisional cards has any **other validation document** with a CT+ event of the same MAVEN type, a single annotated Agent, and a nonpronominal Agent mention within 80 characters. All 22 do; counts range from **one** document for `Testing` to **59** for `Conquering`. This establishes a raw pairing pool, not valid pairs. MAVEN types such as `Action` and `Getting` are broad, and a source sentence may require antecedents outside its excerpt. Match trigger semantics and adjudicate both sentences before pairing.
