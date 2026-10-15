# Data validity audit before this diagnostic

The existing candidate-free development set has 100 events. Its `patient_question` sometimes asks for a prepositional participant: `Who did someone finish behind?` paired with `Kyle Busch finished second ... behind Gordon ...`. Gordon is the answer to that original QA-SRL question, but the question does not establish Gordon as the semantic patient of `finished`.

The new rule keeps original active, non-negated question pairs with no preposition and no second-object slot. The subject question must have an unknown subject and `someone` as object; the direct-object question must have `someone` as subject and an unknown object. Exactly 56/100 events pass this rule; 44/100 fail one or more requirements. This is not a claim that all 44 are annotation errors, nor that the remaining 56 have verified semantic roles. It identifies incompatibility with this narrow grammatical diagnostic.

The selected 24 events are the first eligible entries in the prior hash-ordered list, with 24 unique sentences from approximately 19 source documents. No outcome-based selection occurs. All are active and subject-first. Keep any paired-event bootstrap intervals descriptive; sentences can share documents, and these are development data.

The prior candidate-free outcome differences remain real under their recorded scoring definitions. Their interpretation as clean semantic actor/patient understanding was too broad. Historical files and protocols are retained unchanged.

Future semantic evaluation must distinguish grammatical subject/object from agent, patient, experiencer, theme, recipient and other roles. Original QA-SRL question answering can use its annotated spans directly. Newly assigned semantic categories and sentence transformations require separate verification.
