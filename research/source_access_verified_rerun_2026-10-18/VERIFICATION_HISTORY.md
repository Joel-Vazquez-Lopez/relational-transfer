# Verification history

Before the amended lock and target inference, adversarial probability tests caught a small numerical issue: subtracting a Python log-normalizer from a NumPy float32 scalar could round the result before conversion to float64. The implementation now converts the selected logit to a Python float before subtraction. The independent sum fixture then passed.

The full synthetic controller rehearsal passed before an input-identity assertion failed because in-memory tokenizer offsets were tuples while their saved JSON counterparts were lists. Comparing normalized JSON representations fixed the assertion; prompt text, token IDs and offsets did not change. The complete rehearsal and all identity checks were then rerun successfully before freezing.

The synthetic rehearsal executed 966 simulated dispatches across both model stages, all phases and controls. These were temporary fake logits, not pretrained model outputs. The old first-token-logit fixture combined real saved first-token logits with explicitly hypothetical continuations and was never credited as an empirical completion.

Commit d1cc521daced34b02e50350c8674829130e3eebf freezes the amended inputs, endpoint, full controller, verification results and unchanged scientific gates before any new Qwen output. The actual run made 333 new forwards, including every scored candidate and greedy continuation. Numerical native/wrapper/dense/repeat checks and continuation causal-prefix checks passed. The independent saved-output audit then checked 104 scored records without additional inference.

The original failed source-access attempt remains unchanged. The rerun stopped on a real complete-answer evaluation error and did not substitute a successful family or change the question. Evaluation source cuts and Llama were not run.
