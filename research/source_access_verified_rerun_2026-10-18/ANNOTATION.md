# AI-only stimulus review

Authorized by the user's persistent instruction: **use ai only annotation**. Deterministic template annotations, independently checked against their rendered sentences by the assistant. No human or external annotator reviewed these items; no paid annotation call was made.

Active pattern: A VERB B. Agent=A, Patient=B, first-mentioned=A. Passive pattern: B was VERB by A. Agent=A, Patient=B, first-mentioned=B. Both orientations are generated mechanically; every question asks either the Agent or first mention in the marked sentence, not the neutral Background.

The verbs followed and visited allow ordinary transitive person-to-person readings. Both participants are named people, so no person/team conflict. Passive voice explicitly identifies the Agent through by. The background contains no people or event participants. Names are capitalized and distinct. The word someone in the Agent query denotes the named Patient, while first-mentioned refers to the marked sentence. The explicit system instruction scopes answers to that sentence.

Uncertainty: synthetic single-clause sentences do not represent natural discourse. Repeated simplistic questions and an irrelevant preamble may change instruction following. Active and passive differ in function words, token count and mention order. Agent and first-mentioned questions differ in wording and length. The passive evaluation changes both names and construction, so a failure cannot isolate which changed factor caused it. Full-name prefix probabilities exclude EOS; actual greedy completion is separately observed with a four-token cap, so truncations remain a limitation. Report all such limitations rather than declaring an isolated semantic manipulation.
