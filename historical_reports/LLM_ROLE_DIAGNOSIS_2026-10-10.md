# What the first internal LLM diagnosis shows

**Model:** local, unmodified Qwen2.5-1.5B-Instruct. All experiments used fixed constructed items and saved protocols before their respective runs. Results are from one model, not evidence about LLMs in general.

1. **The answer is strongly question-dependent.** On one new 64-event panel, the model answered the actor on only 3/64 active sentences with “Who did the action?” but 38/64 with “Who performed the action?” It answered 64/64 passive sentences under both phrasings. [Protocol and results](llm_role_internal_v1/RESULTS.md).
2. **A middle-layer state can steer the answer.** On 35 cases that the first wording missed and the second got right, swapping in the clearer prompt's state after block 20 gave the actor in 21/35 cases. A role-reversed donor gave 16/35. The actor-minus-patient score contrast was positive in all eight name-pair groups. This first control also changed the sentence, so it did not isolate roles.
3. **The tighter same-sentence control has a large effect, but remains position-confounded.** On a second fresh panel, the actor donor from the same sentence produced the actor in 41/48 selected cases, whereas a patient-question donor produced the actor in 0/48. All selected cases were active; actor and patient were always the first and last names. [Protocol and results](llm_role_internal_v2/RESULTS.md).
4. **The model is not a universal last-name chooser.** On a third fixed panel, generic actor-question accuracy was 4/64 for active, 64/64 for passive, 56/64 for object clefts and 63/64 for subject clefts. The subject-cleft actor was the *first* name. Changing to “Who performed the action?” raised active accuracy to 63/64 but lowered object-cleft accuracy to 41/64. [Protocol and results](llm_role_structure_v1/RESULTS.md).

## Scientific reading

The safe conclusion is that **question wording and linguistic construction jointly determine whether this model uses the right role in its answer**. The patch demonstrates a causal effect of a prompt-conditioned internal state on the answer. It does not yet prove that an abstract actor/patient representation is present under the failing prompt and merely ignored, nor that the controlled Transformer's ordered-role repair would work in this pretrained model. Simple position selection explains some results but not the full cross-form pattern.

## Next decisive test

Fit a role readout on one set of constructions and names, then test it on **different constructions and held-out names**, balancing whether actor is the first or last name. Compare the readout with the model's answer on the same items. Only if a transferable signal is found should a narrow intervention and matched ordinary-versus-role-consistency fine-tuning study follow. An authentic-sentence set and a second model family would test how far any effect extends. The three panels here are diagnostics that determine that design, not a publication-ready LLM mechanism claim.