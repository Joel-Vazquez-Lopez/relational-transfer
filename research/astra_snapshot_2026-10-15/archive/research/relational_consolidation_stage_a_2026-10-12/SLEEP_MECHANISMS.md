# Artificial consolidation: operational candidates

“Sleep” is a provisional analogy. Every method below changes trainable parameters unless noted. None has been tested in this new project. The fair question is whether its *information, loss, or timing* adds a measurable benefit beyond established replay, augmentation, consistency and distillation.

## S1 — replay of observed events

- **Input:** only previously seen wake-phase training events and their original target answers. No test event, new pair, new label or teacher output.
- **Generation/receipt:** a deterministic sampler replays stored examples. No synthetic facts are created.
- **Loss:** the exact original supervised answer loss, e.g. cross-entropy on actor and patient outputs. Update the same model parameters; use no external teacher/oracle.
- **Comparison:** equal additional updates on ordinary examples, and the identical replay examples interleaved rather than placed after wake training. Match tokens and optimizer schedule. If replay is merely extra training on the same data, its only difference from the matched extra-update arm is sample order; do **not** market it as a new mechanism.
- **Failure mode:** overfits narrow surface forms and worsens transfer. A local replay-only Qwen continuation already improved Bank candidate contrast by only 1/130 while patient clefts fell 618→295/750 ([saved report](../../llm_role_natural_repair/RESULTS.md)). This does not rule out diverse replay or other schedules.

## S2 — transformation-consistency consolidation

- **Input:** observed wake training events with trusted actor/patient labels; deterministic, grammar-checked equivalent realizations (active/passive/cleft) and a role-reversed *event* only when its labels are derivable from the training schema, not from held-out test answers.
- **Generation/receipt:** the renderer produces paired sentences and a permutation map for output roles. It must not render the reserved withheld participant pair. No teacher is needed for symbolic labels; the verified rule schema is an **oracle** and must be disclosed as extra supervision.
- **Loss:** `L = L_answer + λ_inv KL(p_roles(x) || p_roles(T_same(x))) + λ_eq KL(P_swap p_roles(x) || p_roles(T_swap(x)))`, with role distributions aligned by participant identity. For language generation, use a consistent role-classification interface or score candidate outputs; raw token-level KL between differently worded answers is invalid. The first term teaches correct outputs; the second preserves role assignments under meaning-preserving form changes; the third changes assignments under genuine role swap.
- **Parameters:** same model/adapter and optional task head update. Freeze the renderer and role schema. Record whether any separate head is trained; compare against equal-head controls.
- **Distinction from augmentation:** paired KL may constrain *relations among outputs* beyond ordinary supervised examples. If the same paired examples with ordinary answer labels and the same schedule perform equally well, the KL adds no demonstrated value. If offline and interleaved KL perform equally, there is no demonstrated sleep-phase advantage.
- **Failure mode:** agreement can be achieved by always naming one person; the equivariance term and joint both-role score block that trivial solution. Other failures include ungrammatical transformations, wrong role maps, oversmoothing, and loss of familiar performance.

## S3 — leakage-safe relational recombination

- **Input:** only training event schemas, training predicate/role lexicon, and explicitly declared inference rules. No evaluation event tuples or target labels.
- **Generation/receipt:** two separable variants: **S3a grammatical recombination** uses already observed participant pairs with new allowed wordings or verb–template combinations whose actor/patient answer is guaranteed by the event schema; **S3b logical recombination** derives new factual triples only where a formal rule and training premises entail them, with a stored proof trace. A graph path alone does not entail that two people interacted or who acted on whom.
- **Loss:** supervised answer cross-entropy on validated recombinations, optionally plus S2 consistency. Parameters update in the same model; the event schema/rule checker is an **external oracle**. Count every generated target as extra supervised information. If the original task has no inference rules, S3b is **not defined** and must be skipped.
- **Distinction from augmentation:** S3a is ordinary structured data augmentation unless the offline schedule itself beats an otherwise identical interleaved schedule. S3b is rule-based synthetic supervision, a potentially useful but different intervention; it cannot be credited to sleep without a same-proof interleaved control.
- **Failure mode:** covertly teaching held-out pair answers, generating false “plausible” facts, coverage rather than abstraction, or memorizing recombined patterns. The proof and partition audits in `BENCHMARK_SPEC.md` are mandatory.

## Optional S4 — offline role-state alignment

Only consider after the behavioral gate. Given paired, **training-only** realizations of one event, align a designated participant-relative state under active/passive while applying the role permutation under true reversals. Trainable parameters update via a contrastive loss with counterexamples matched for entity, position and construction. The model's own representations are not an external teacher; a human-designed pairing map is supervision. A same-pairs interleaved contrastive control and ordinary output-consistency control are required. A probe decoding roles afterward is insufficient; causal state intervention and hard behavioral transfer are needed. This resembles existing representation-alignment methods and is not presently a novelty claim.

## Attribution rule

An **offline advantage** means offline beats interleaving when examples, labels, objective, target count, tokens, updates and learning-rate trajectory are matched. An **objective advantage** means S2 KL beats ordinary supervised augmentation on the same pairs and schedule. An **information advantage** means S3 beats an arm lacking its extra derived examples; that alone does not show a new learning principle. These are different claims and should be reported separately.
