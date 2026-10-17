# Factorial competence gate: cross-model causal study stopped

**17 October 2026. Frozen bounded local screen; no hidden-state capture or causal intervention.** The complete [pre-output lock](FACTORIAL_GATE_LOCK.json), [runner](factorial_gate.py), [summary code](summarize_factorial_gate.py), [machine summary](FACTORIAL_GATE_SUMMARY.json), and per-prompt [Qwen](FACTORIAL_GATE_qwen7b.json) / [Llama](FACTORIAL_GATE_llama3b.json) raw records are preserved. Qwen 7B and Llama 3B were loaded only from existing local MLX model files. No downloads, paid APIs, training, or adaptive replacement items.

## Frozen design and technical amendment

Four new lexical families were split before inference: Iris/Noah *thanked* and Maya/Owen *invited* for development; Nora/Peter *thanked* and Ruby/Victor *invited* for evaluation. Every family had four actor-swap × voice cells, two questions (Agent/Patient), and two frames: canonical passive (Patient first) and agent-first passive cleft (“It was by [Agent] that [Patient] was [verb]”). The active sentence was identical in both frames, so each model received **64 recorded rows but only 48 distinct factorial prompts**. All 64 rows count toward the frozen family screen; duplicates are declared, not treated as independent evidence.

The first Qwen invocation stopped before any new-family output: the installed MLX lacked `mx.log_softmax`. The scorer was changed to the mathematically identical `row[token] − logsumexp(row)` and both lock and code were committed on the isolated research branch before the next invocation. Prompts, models, names, thresholds and hypotheses were unchanged. This is recorded in the lock. The full-name scorer used the log probability of *every token* in each name, including Llama names that begin with partial tokens, and greedy outputs were checked separately. It does not include end-of-turn probability; the scorer is a pairwise candidate readout, not the probability of a complete free-form response.

Both models passed the exposed eight-prompt pre-factorial scorer validation: correct greedy full name and positive full-name margin **8/8 Qwen, 8/8 Llama**. In the new factorial, greedy and full-name score correctness agreed on all 64 rows for both models. Thus the old first-token Llama readout problem was resolved for this gate, but the model's task competence was not.

## All family outcomes

| Model | Development Iris/Noah thanked | Development Maya/Owen invited | Evaluation Nora/Peter thanked | Evaluation Ruby/Victor invited | Fully competent families |
|---|---:|---:|---:|---:|---:|
| Qwen 7B | 14/16 | 16/16 | 13/16 | 16/16 | **2/4** |
| Llama 3B | 13/16 | 12/16 | 13/16 | 13/16 | **0/4** |

Qwen got **59/64 recorded rows** and **43/48 distinct prompts** correct. Llama got **51/64 recorded rows** and **39/48 distinct prompts** correct. Qwen's Agent/Patient counts were 29/32 and 30/32; Llama's were **31/32 Agent but 20/32 Patient**. These are descriptive counts from a tiny, non-random family set, not population accuracy estimates. No family was replaced.

Every failed answer, expected answer and full-name margin is in the [machine summary](FACTORIAL_GATE_SUMMARY.json) and [raw outputs](FACTORIAL_GATE_llama3b.json). The pattern is informative:

- Qwen's five failures were all *thanked* passive cases. Two were in canonical Iris/Noah, and three in Nora/Peter, including one cleft. Both *invited* families passed all 16 rows.
- Llama's 13 failed recorded rows correspond to **nine distinct prompts** because active prompts appear in both frames. Twelve of the 13 failed rows asked for the Patient. All four families failed an active Patient question in at least one role assignment, before a voice transformation; the remaining failures include cleft passive Patient questions. One Llama failure was an Agent question in the canonical Maya/Owen `RV` cell.
- Both models had no failures on the exposed scorer-validation prompts. Passing those old active Agent cases did not guarantee competence on the new Patient/voice factorial.

The alternate cleft frame is a valid design contrast at the text level: switching active to cleft passive preserves Agent-first name order, whereas switching to canonical passive makes Patient first. However, lexical and construction differences remain. This screen does **not** show how either model represents that contrast; it only tests whether reliable answer labels exist for intervention analysis.

## Decision under the frozen stop rule

The lock required at least one development and one evaluation family to pass **both frames and both roles in each model** before a cross-model geometry or causal-patch interpretation. Qwen met that limited family count (Maya/Owen and Ruby/Victor). Llama did not pass a single family. Therefore **the cross-model gate failed**. I stopped before capturing new hidden states or running patches. Selecting only Qwen's successful *invited* families, changing the Patient prompt, or replacing Llama cases now would be adaptive and would invalidate this prospective test.

This is a useful negative feasibility result. It does **not** falsify the possibility of task-conditioned developmental relations in general; it shows that this exact two-model, two-role factorial does not provide a clean test. The next design, if independently authorized and frozen later, should first establish reliable Agent/Patient competence under voice alternation with a source-valid task and then repeat the surface-order control. No new atlas method, shared computational pathway, or paper-ready effect is demonstrated here.

## Provenance and limitations

Raw results contain all prompts, token counts, generated strings, candidate token IDs, teacher-forced full-name scores, expected names and failures. The summary validates that both raw files reference the final lock SHA-256 and records raw-file hashes. Distinct prompts are the proper answer-level unit; lexical family is the proper gate unit. The four families and synthetic sentences are too few for generalization, passive clefts are somewhat marked, and candidate full-name ranking excludes termination probabilities. This was a competence gate, not a causal experiment.
