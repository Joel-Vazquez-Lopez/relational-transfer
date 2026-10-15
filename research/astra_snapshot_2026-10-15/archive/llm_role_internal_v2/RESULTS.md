# Same-sentence donor control on fresh Qwen items (10 October 2026)

After the first diagnostic, this separate [protocol](PROTOCOL.json) fixed new name pairs and verbs before running the model. It used the same unmodified Qwen2.5-1.5B-Instruct model and compared three questions on each sentence: generic actor, performed actor, and a verb-specific patient question. The [raw responses and patches](RESULT.json) and [summary](SUMMARY.json) are retained.

| Question target | Active correct | Passive correct |
|---|---:|---:|
| Generic actor | 0/64 | 64/64 |
| Performed actor | 48/64 | 64/64 |
| Verb-specific patient | 64/64 | 0/64 |

All answers were valid names. The fixed selection rule retained 48 cases where the generic actor answer was wrong while both the performed actor and patient questions were answered correctly. **All 48 were active sentences.** Their donor states came from the *identical sentence*, changing only the question. Transplanting the performed-actor block-20 state into the generic prompt produced the correct actor answer in 41/48 cases. Transplanting the patient-question state produced 0/48 correct actor answers. The actor-donor minus patient-donor score-margin difference averaged +10.75 when each of eight name-pair groups received equal weight; all eight group contrasts were positive (exact two-sided sign-flip p = 0.0078125).

This establishes that changing the question creates a causally consequential block-20 state that can steer which name the model gives. It does **not** isolate an abstract actor/patient representation: every selected example was active, so actor was always the first name and patient the last. The large difference could include a first-name/last-name selection rule. The patient prompt failed on every passive sentence, which prevented a balanced intervention across both voices. A later syntax test was therefore needed to investigate that shortcut.
