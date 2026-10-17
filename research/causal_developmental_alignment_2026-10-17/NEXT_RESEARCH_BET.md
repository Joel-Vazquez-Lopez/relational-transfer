# Next bet: counterfactual transport across syntax

**17 October 2026 — design proposal, not a frozen protocol or new inference.** This follows the failed [actor-swap × voice competence gate](FACTORIAL_GATE_REPORT.md). It does not reinterpret its 2/4 Qwen or 0/4 Llama fully competent families as a positive atlas result.

## The conceptual change

The object worth measuring may be **transportability of a controlled change**, rather than similarity of the hidden paths that the change produces. If a model encodes a role reversal in an active clause, can an intervention drawn from that contrast produce the appropriate counterfactual answer when inserted into a *different syntactic realization* of the same event? Does the relation among these transfer effects evolve across layers, and does a second model show a corresponding pattern in its own coordinates?

This is a restricted causal question. A positive result would not prove identical circuits or a universal semantic-role vector. It would show that a role-changing intervention generalizes across specified contexts better than controls. The existing final-token transplant is especially vulnerable to answer copying, so the next design must target an event-bearing site and include answer-matched unrelated donors.

## Why this bet follows the data

The single-path Gram was more similar for unrelated contrasts than role reversals. The saved role-swap traces have a subject/object opposition that lexical name exchange alone can cause. The factorial competence screen showed a sharp task asymmetry: Llama was correct on 31/32 recorded Agent rows but 20/32 Patient rows, with 0/4 fully competent families. Qwen passed 2/4 families, both using *invited*. The next study cannot select those successful families as confirmation data or silently relax the previous stop rule. It can use these results to define a **new, narrower prospective question** and then test entirely fresh lexical families.

## Minimal prospective experiment

1. **Fresh source set.** Freeze new names, verbs and two frames before any model output. Use natural active and canonical passive pairs plus an agent-first passive construction, balancing whether the Agent is the first mentioned person. Hold out entire lexical families. Existing four families are development evidence only.
2. **Narrow competence gate.** Ask only the Agent question in the first study, because the previous Llama Patient prompt failed often. Record Patient responses as a separate descriptive failure mode, not a prerequisite or a mechanism endpoint. Require correct full-name greedy answers and positive teacher-forced full-name margins in *all* Agent cells of a family. Report every rejected family; no replacement. This tests Agent selection, not full two-role binding.
3. **Native counterfactual transfer.** Within each model, take a clean/corrupt role contrast from an active donor pair. At predeclared semantically aligned event-token sites and depth bins, transplant the appropriate donor state into a passive recipient. Measure the complete-name log-probability margin for the intended counterfactual Agent. Repeat in reverse (passive donor into active recipient), in the canonical and Agent-first passive frames. Do not align Qwen and Llama hidden coordinates or fit a flexible depth warp.
4. **Controls that can defeat the idea.** Include self-patch, same-answer but unrelated-event donor, **same target name in the wrong semantic role**, different-answer unrelated donor, same-meaning voice donor, adjacent/wrong token, and a donor matched for surface first-name order. Report perturbation norm and residual norm at every site and depth. If unrelated same-answer donors induce the same response, the effect is generic answer transfer. If the same name works equally well when it is a Patient in the donor, the effect is lexical identity rather than role-sensitive. If the result follows first-name position when canonical and Agent-first passive disagree, it is positional.
5. **Prediction, not label recovery.** On development families, choose one fixed site/depth summary and a simple cross-model stage mapping. Predict the *sign and relative size* of selective transfer on untouched families and the second voice frame. Compare to normalized depth, layerwise step norm, raw output-margin trajectory, single-path Gram, a position/name-order heuristic, and shuffled transformation labels. A relation descriptor adds value only if it improves prediction after answer-matched donor correction.

For model `m`, layer/site `z`, source contrast `p`, recipient context `c`, and target answer pair `(a,b)`, define the native effect

`T_m(z,p→c) = [log P_m(a|patch(z,p,c)) − log P_m(b|patch(z,p,c))] − [log P_m(a|c) − log P_m(b|c)]`.

Define **selective transfer** by subtracting the corresponding same-answer unrelated-donor and self-patch effects, with all components reported separately. The resulting matrix over donor transformation, recipient syntax, site and depth is the candidate *causal response atlas*. This is a response matrix, not a new invariant kernel. Across models compare held-out effect signs, ranks and profiles in this shared experimental label space. A small `T` may reflect inaccessible information, bad site choice or a genuine mechanistic difference; its interpretation requires the controls and competence gate.

## Decision gates

- **Stop at competence** if a fresh common set of Agent cells is not solved by both models. Do not swap prompts or models after inspecting outcomes.
- **Stop at selectivity** if event-site patches cannot beat same-answer unrelated donors and wrong-site controls on development items. A final-token-only effect is insufficient.
- **Stop at prediction** if the atlas does not beat the simple depth, output and position baselines on untouched lexical families. Do not add transformations or tune a warping function on the holdout.
- **Escalate only after replication** on independently checked natural sentences and another model family. A synthetic matched-pair success alone is a method feasibility result, not a paper-level linguistic law.

## Novelty boundary

Activation patching and its metric sensitivities are established; see [Zhang and Nanda](https://arxiv.org/abs/2309.16042) and [Heimersheim and Nanda](https://arxiv.org/abs/2404.15255). [Aljaafari et al.](https://aclanthology.org/2026.findings-acl.1964/) already study causal semantic-role circuits and developmental tracking. [CKA](https://proceedings.mlr.press/v97/kornblith19a.html), [DSA](https://papers.neurips.cc/paper_files/paper/2023/hash/6ac807c9b296964409b277369e55621a-Abstract-Conference.html), [InputDSA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a437d77270d5588a84d63a851a3fdeb-Abstract-Conference.html), [Transformer trajectory geometry](https://arxiv.org/abs/2606.09287), and [causal abstraction](https://www.jmlr.org/beta/papers/v26/23-0058.html) cover much of the broad territory. The defensible distinction, if empirically supported, is **prospective prediction of selective cross-syntax intervention transfer across model families after strong answer and position controls**. Whether prior work already does this exact test still needs a full-method literature review before a paper claim.

## Recommendation

**Pursue one new prospective feasibility study, not a broader model sweep.** The first objective is to determine whether selective event-site role transport exists at all. If it does not, a paper could still emerge from a rigorously documented limitation of geometric trajectory similarity, but only after independent replication. If it does, the atlas becomes a predictive organizing device; its value must be demonstrated against simple baselines rather than assumed from an attractive diagram.
