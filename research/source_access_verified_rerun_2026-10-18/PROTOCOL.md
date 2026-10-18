# Bounded source-access feasibility protocol

Frozen before target-model outcomes. User authorization: **can you verify it and run it again?** The scoring and call-accounting amendment in AMENDMENT.md supersedes the original endpoint and no-rerun closure; prior results are preserved. Existing local Qwen 7B first; existing local Llama 3B only after Qwen supports its fresh evaluation prediction. No training, downloads or paid calls. Earlier stopped studies remain closed.

## Scope

Test whether a question-specific difference in dependence on continued context access can be measured, then predict its direction on new passive sentences. This is a diagnostic of causal access under an explicit attention intervention, not a semantic-versus-copying mechanism test or reconstruction of a full computation. Strong syntax-aware copying remains compatible with success.

## Stimuli and endpoint

Two development families: Nina/Luke *followed*, Emma/Jack *visited*, active voice. Two evaluation families: Helen/Martin *followed*, Sarah/James *visited*, passive voice. Each has both Agent assignments and two questions: Who VERB someone? / Who was mentioned first in the sentence? Exactly eight prompts per phase per model. The active prompts make both queries select the same answer within an orientation. Passive prompts separate the selected identities; orientation averaging counterbalances names but does not eliminate distributed answer preparation. Names/verbs/construction are not independently sampled factors.

An invariant neutral background precedes the event to supply an irrelevant-source control. All templates, labels, prompts, token offsets, aliases, source intervals and model-specific token IDs are frozen in INPUTS.json. AI-only annotation is explicitly authorized by the user's earlier instruction; no independent human validation is claimed.

At the original assistant prompt-end, record the full-vocabulary first-token argmax for inspection, but do not classify a multi-token name by that token alone. Score the sum of token log probabilities along each complete natural name sequence, with distinct leading-space and no-leading-space paths combined by logsumexp. No length normalization and no EOS score. Independently generate greedily until an observed EOS or four new tokens. Correctness requires the exact stripped name, observed EOS and positive gold-versus-rival full-name margin. Persist every candidate path, token log probability, generation token, EOS/truncation flag and all child forward IDs. Partial names and unobserved continuations cannot be credited. Truncation or format failure is distinct from established linguistic failure. See AMENDMENT.md for verification and scope.

## Intervention

All calls use batch one, total length 128 with dummy tokens strictly after the original prompt. Each wrapped block receives an explicit dense boolean causal mask, including baseline blocks. Preserve native attention, MLP, normalization and residual arithmetic.

For every suffix query position (from the first question token through the original assistant end, and later dummy positions), remove attention to every source position from the first event token up to the suffix. Include subsequent context-dependent punctuation and Question label tokens. No native cache. Invariant pre-event background/instructions remain accessible and cannot have received future event information. All earlier suffix positions receive the intervention, preventing relays through other question tokens.

Block indices are zero-based; a cut k affects blocks k through 27. Fixed cuts: 0, 7, 14, 21, 28. Cut 28 is a null. For cuts 7, 14, 21, a matched control removes the same number of source columns immediately before the event, in the irrelevant background, at the same suffix queries and blocks. These controls match edge count and intervention depth but not absolute source position or attention mass.

Two fixed restoration conditions start with cut zero and reopen source access only in blocks [7,14) or [14,21). Fully restored [0,28) is an additional identity control. No other cut, restored window or patch is permitted.

## Baseline and technical gates

Evaluate all eight development baselines before any substantive attention cuts. Require 8/8 correct, including all four answers in each family. All failures stay in the denominator; do not substitute successful families. Evaluation requires the analogous 8/8 baseline gate before evaluation interventions.

On the first development prompt, compare an unwrapped native-mask pass, wrapped native-mask pass, dense baseline, and dense repeat. Maximum full-logit discrepancies must be <= .05 nat; repeat last-token captured states must agree within 1e-6 maximum absolute difference.

Across all wrapped calls require last-token residual/attention/MLP reconstruction relative error <= .002. Prefix states must be exactly equal across later question texts at baseline. Under cuts, maximum prefix difference norm divided by the maximum baseline prefix-state norm must be <= 1e-6; last-token captures in blocks before the cut must agree to 1e-6 maximum absolute difference.

After all fixed development interventions, require:

- Cut-28 and fully-restored full logits match baseline within .05 nat on every prompt.
- Cut-zero outgoing suffix states across name-swapped orientations agree within 1e-6 maximum absolute difference, and logits within .05 nat, separately for each family and query. Prompts have exactly matching surviving suffix tokens and source intervals. This tests an intended no-information path; its answers are not evidence of linguistic knowledge.
- At least 23/24 matched irrelevant-cut answers correct. Zero reconstruction, prefix or before-cut failures.

Stop at any gate failure. Restoration-window outcomes are reported descriptively and cannot rescue a failed gate. Native-versus-dense disagreement is a technical failure, not permission to change numerical criteria.

## Prediction selection and prospective evaluation

For each k in {7,14,21}, define source cost on a prompt as margin(irrelevant cut k) minus margin(event cut k). Within each family, average orientations and form D = Agent cost minus first-mentioned cost. Average the two family differences to obtain the development D.

Select the largest absolute D meeting all these criteria, breaking ties toward the shallower cut:

1. Absolute mean D >= .5 nat.
2. Both family differences have the same strict sign as the mean.
3. Absolute D is at least twice the mean absolute irrelevant-cut-versus-baseline margin drift across all eight prompts.

If no cut qualifies, stop: no evaluation-model outputs, no Llama. Save the selected cut/sign and hash of raw development outcomes before any evaluation baseline call.

Prediction: at this unchanged cut, evaluation D has the same sign in both new passive families, absolute mean >= .5 nat, and exceeds the same twice-irrelevant-drift criterion, with all competence and intervention controls passing. Record every other cut regardless of outcome. No fitted depth alignment or selected secondary peak.

If Qwen supports that prediction, run Llama once using the identical stimuli and Qwen-selected cut/sign. Llama must pass its own baseline/numerical/intervention gates and satisfy the same signed difference on its development panel before evaluation. No independent favorable cut selection for Llama. A failure is a failed correspondence, not absence of all related computation. Full developmental profiles remain descriptive.

## Rival predictions

- An early-transfer known program becomes insensitive to later source cuts; a late-read program remains sensitive over a longer range, despite identical terminal answers.
- A query-insensitive dependence profile predicts D near zero after matched controls; no selected cut means the proposed query distinction is unsupported in this pilot.
- A developmental pattern tied to active first-person retrieval may fail on passive evaluation. That failure challenges generalization rather than justifying new constructions.
- Generic mask disruption may harm both query demands or irrelevant controls. Failed controls prevent a selective-access interpretation.
- Copying/answer preparation can yield nonzero D and successful generalization. The study does not distinguish these from an abstract role mechanism.

## Call and adaptation bounds

Maximum **2,000 attempted model forwards total; 1,000 per model**. The original 195 primary records per model remain: 16 baselines, three validations and 16 x 11 conditions. Full candidate scoring and observed greedy completion require additional, explicitly journaled forwards; at most 963 total per model given the frozen tokenizations and four-token generation limit. The previous failed attempt used 11 forwards, preserved separately. All attempts, including exceptions and validation, are journaled before dispatch. One attempt per model; no automatic resume, rerun, replacement, prompt repair, site search or adaptive experiment. A technical exception closes the attempt for review. Stop after reporting.

## Reporting

Keep family-level competence separate from answer-level accuracy. Preserve all prompts, labels, aliases, full logits, compact native states, cut-zero full suffix states, baseline prefix captures, controls, attempted-call ledger, failures and code hashes. Publish source text/ledgers/report on the existing research branch, keeping local tensor hashes and explicit upload limits. Report small constructed-sample, quantization, wording, answer-identity and off-distribution limitations. No population, family-law, universal sequence or novelty claim follows from this feasibility study.
