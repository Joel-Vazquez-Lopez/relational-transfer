# Annotation review before model outputs

The proposed design requires a human annotation pass. No human pass has yet been recorded. This document contains all 16 sentence variants. Please check the ordinary English reading, not only the intended bracketed parse.

For each sentence, the doer is the Agent, the recommended/referred people are Themes, and the recipients/targets introduced by to are Goals. No means the queried assignment is absent from this sentence; it does not assert real-world negation.

| ID | Sentence | Agent | Theme members | Goal members |
|---|---|---|---|---|
| a01_within_0 | Ada recommended Boris and Clara to Neil. | Ada | Boris, Clara | Neil |
| a01_within_1 | Ada recommended Clara and Boris to Neil. | Ada | Clara, Boris | Neil |
| a01_across_0 | Ada recommended Boris to Clara and Neil. | Ada | Boris | Clara, Neil |
| a01_across_1 | Ada recommended Clara to Boris and Neil. | Ada | Clara | Boris, Neil |
| a02_within_0 | Mara referred Oscar and Hugo to Rose. | Mara | Oscar, Hugo | Rose |
| a02_within_1 | Mara referred Hugo and Oscar to Rose. | Mara | Hugo, Oscar | Rose |
| a02_across_0 | Mara referred Oscar to Hugo and Rose. | Mara | Oscar | Hugo, Rose |
| a02_across_1 | Mara referred Hugo to Oscar and Rose. | Mara | Hugo | Oscar, Rose |
| a03_within_0 | Leah recommended Felix and Simon to Vera. | Leah | Felix, Simon | Vera |
| a03_within_1 | Leah recommended Simon and Felix to Vera. | Leah | Simon, Felix | Vera |
| a03_across_0 | Leah recommended Felix to Simon and Vera. | Leah | Felix | Simon, Vera |
| a03_across_1 | Leah recommended Simon to Felix and Vera. | Leah | Simon | Felix, Vera |
| a04_within_0 | Eva referred Colin and Ivan to Ruby. | Eva | Colin, Ivan | Ruby |
| a04_within_1 | Eva referred Ivan and Colin to Ruby. | Eva | Ivan, Colin | Ruby |
| a04_across_0 | Eva referred Colin to Ivan and Ruby. | Eva | Colin | Ivan, Ruby |
| a04_across_1 | Eva referred Ivan to Colin and Ruby. | Eva | Ivan | Colin, Ruby |

Question wording used for every family (with its names/verb substituted):

- Did Ada recommend someone? (Two families instead ask about the final person; gold No.)
- Was Boris recommended to someone?
- Was someone recommended to Boris?
- In the sentence, was Boris mentioned before Clara?

Potential concerns: coordination scope, singular someone in plural-recipient sentences, referral/recommendation pragmatics, and binary verification versus open participant retrieval. The model system prompt specifies sentence-role extraction and mention order.

A human response can either confirm the readings, flag an ambiguity, or explicitly waive the human-pass prerequisite and accept an AI-only annotation limitation. No confirmation is inferred from elapsed time.
