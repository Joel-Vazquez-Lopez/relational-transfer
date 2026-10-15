# Literature position after the token-matched result

Checked 13 October 2026. This is a targeted search, not a comprehensive novelty review.

- Wang et al., [*An MRC Framework for Semantic Role Labeling*](https://aclanthology.org/2022.coling-1.191/) (COLING 2022) already conditions argument extraction on semantic-role queries. Supplying a role query is therefore **not** a new general method.
- Zhang et al., [*Transfer Learning from Semantic Role Labeling to Event Argument Extraction with Template-based Slot Querying*](https://aclanthology.org/2022.emnlp-main.169/) (EMNLP 2022) compares several role-query strategies and finds template slot querying effective for transfer. It is a relevant structured-prediction baseline, not evidence that our repair is unprecedented.
- [*Promptly Predicting Structures: The Return of Inference*](https://aclanthology.org/2024.naacl-long.7/) (NAACL 2024) uses role questions with generative language-model span prediction and constrained structural inference. Compare to this before any broad claim about a role-routing innovation.
- Raghav and Jana, [*Are LLMs Good for Semantic Role Labeling via Question Answering?*](https://aclanthology.org/2025.ijcnlp-srw.21/) (2025) evaluates multiple LLMs on QA-SRL with prompting. It establishes a nearby evaluation setting; our current experiment is smaller but includes a matched fine-tuning intervention and answer-routing diagnosis.

**Defensible present contribution:** within one small pretrained model family and one candidate-supported QA-SRL slice, we isolate a failure to route often-correct generated role fields into final answers, and show that an explicit requested-role input cue repairs that behavioral failure against a token-matched neutral cue on a new sentence bank. The goal-first output ordering failed. This is a controlled empirical finding, not yet a new general decomposition algorithm or a conference-ready cross-domain result.

**Next gate for a stronger paper:** a human-checked, fresh natural-language test with unfamiliar question formulations, unseen constructions and no candidate list; role-query and deterministic-routing baselines from the prior-work families above; replication in another model family. The exact claim can then be scoped to whichever effects survive.
