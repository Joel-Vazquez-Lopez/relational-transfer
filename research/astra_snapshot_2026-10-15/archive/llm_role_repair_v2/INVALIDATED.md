# Version 2 training results are invalid

The version 2 training batches were left-padded but did not pass an attention mask to the model. Evaluation did pass an attention mask. This train/evaluation mismatch changed the model's predictions even on a training example: the saved ordinary adapter answered `Stephen` for `Stephen chased Alan.` under the training-style padded input, but `Alan` when the same input used a padding mask. The ordinary result in this folder must not be interpreted as a training effect.

Generic-consistency and ordered-role version 2 runs were interrupted after this bug was identified. The unmodified pretrained baseline did not use the faulty training path and remains a valid reference for the identical fixed test rows in [version 3](../llm_role_repair_v3/PROTOCOL.json). Version 3 passes the attention mask during training and is the basis for all trained-arm comparisons.
