# BX-009 — Separating wrong-event source confusion from wrong-role answers

**17 October 2026 | Brain-X | software completed; authentic-data replay BLOCKED; no model inference.**

## Direct check and evidence limits

I reviewed the coordinator's [one-paper decision](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/34340dcf1a83ede63d8694b989c5db213f6950a0/research/armaggedon_frontier/NORTH_STAR_SINGLE_PAPER_DECISION_2026-10-17.md), [#31 gate](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/31), [#33 source report](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33#issuecomment-6233486882), and [Codex's no-witness certificate](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/a455d8b5321704b465d6f9576018de159de6de52/research/decisive_witness_gate_2026-10-17/REPORT.md). Checked accessible private GitHub branch history and ChatGPT Library handoffs for the underlying \`DISCOURSE_LOAD_FINAL_REPORT.md\`, source manifest and 64-case paired raw Qwen7B outputs: **original item-level records not accessible**. I did NOT independently recount the real predictions.

The study owner reported frozen corrected Qwen7B accuracy **32/32** on individual sentences, **64/64** bare paired and **57/64** with added irrelevant named bystanders. All seven reported errors were said to borrow another event's participant, with the target in sentence 2; zero bystander names were output. **The predeclared threshold was ≥8 second-target errors; 7 failed it.** These are owner-reported developmental findings (not independently replayed), not a model error law, positive replication, or mechanism. Neither source-document clustering nor literal matched input IDs are known here. No p-value or per-document confidence interval can be responsibly inferred from these aggregates.

## New scientific distinction: source × role

When an answer comes from the *wrong event*, was its **role preserved**? That cannot be decided from the claim “the model gave the other event's participant.”

The following surface-source taxonomy makes the question testable using frozen raw outputs and independently verified original event/role names:

| Offline classification | What the emitted string matches |
|---|---|
| \`correct\` | target event, requested role |
| \`other_event_same_role\` | competing event, requested role |
| \`other_event_other_role\` | competing event, opposite role |
| \`own_event_other_role\` | target event, opposite role |
| \`bystander\` | unrelated named nonparticipant |
| \`ambiguous_candidate\` | identical name can refer to several candidates |
| \`out_of_set\` / \`empty\` | no exact match / no answer |

**Causal interpretation ceiling:** even if all seven errors preserved the requested role, **a competent query-aware positional/lexical event copier would predict the same output pattern**. This would establish only a specific *behavioral role-preserving source-substitution profile*, not an internal shared semantic event selector. If errors also switch roles, even that weaker profile fails. Nonidentical names can corefer; exact string matching cannot authenticate referents.

## Reproducible implementation actually executed

- [\`bx009_source_provenance.py\`](bx009_source_provenance.py): Python stdlib, Git blob **\`6c3fedb8992dc05f4331b2e5a135083f6f96ae05\`**, byte-identical to locally executed source, SHA256 **\`d8d9a46f4b38dfee8d812568c2f8f58d35eae9c7f9c611fb445c00ba1b05c6da\`**.
- [\`test_bx009.py\`](test_bx009.py): 19 adversarial **fictional fixture** tests pass; Git blob **\`69aebf4a1f3e24fdc2a401534f5d82711b3edabd\`**, SHA256 **\`61f655746be54158b66db0bb27bbde8cb517f126a047ba2fd24cd0107228d7cb\`**.
- Exact case/arm schema: frozen \`case_id\`, \`document_id\`, \`source_kind\`, \`source_review\`, \`rights_review\`, \`target_event\`, \`target_role\`, \`target_order\`, \`roles\` (two events × agent/patient), \`bystanders\`, SHA256 commitments for common original question and each complete model-visible prompt; saved responses contain \`case_id\`, \`arm\`, \`model_revision\`, raw \`response\` and matching prompt/question hashes.
- Fails closed on unpaired/missing arms, duplicate case IDs, changing model revision, changed prompt/question SHA commitments, invalid role mapping, missing named bystanders. Outputs aggregate source×role classifications, \`bare→bystander\` transitions, rescue/harm/net, target-order/role breakdown and distinct source document counts, **not original text/names/responses/IDs**.
- Records source and rights/gold/prompt manual-review **assertions**, never silently promotes them into independent verification. Any incomplete declaration yields descriptive-only status. Even all asserted approvals retain explicit independent-human/source review requirement.
- The original ≥8 threshold remains unchanged in machine output.

**Executed local command:** \`python -m unittest -v test_bx009\` → **19/19 PASS**. Fictional inputs only. **Real 64-pair analysis not run** because source/predictions absent.

After a permitted raw artifact handoff only:

\`\`\`bash
cd research/brain-x
python -m unittest -v test_bx009
python bx009_source_provenance.py \
  --manifest /approved/saved_manifest.jsonl \
  --predictions /approved/frozen_predictions.jsonl \
  --model-revision EXACT_CHECKPOINT_OR_HASH --threshold 8 \
  --out /tmp/bx009_descriptive.json
\`\`\`

A raw-prompt hash can attest to prompt identity, **not** prove that bare and bystander conditions differ only by a non-answerable name. Independent reviewers must inspect frozen full prompts, author/source rights, original task semantics, entity/coreference resolution, train/dev/test history, first-run confound and all failures. The model-visible prompt contains no scorer gold. Do not label constructed text as natural original discourse.

## One bounded next decision, no unapproved runs

1. First obtain **already-generated raw items and exact pre-run record** (permission required); verify count, source rights, prompt pairing, frozen threshold, original roles, token positions, same/different referents and per-document clustering. No new inference.
2. Classify every output; compute per-role/target-position wrong-source rate, rescue/harm, and the **same-role vs opposite-role competing-event confusion**. Preserve unknown and ambiguous categories in denominators.
3. **Comput owns** the competent combined syntax+lexical+position+salience/length copier rival. Brain-X supplies the error taxonomy only. If its predictions already explain all seven, STOP the unique mechanism/paper narrative; if not, preregister a prospectively discriminating original-source heldout comparison in #31 **before** spending compute.
4. Novelty warning: [Gur-Arieh et al., *Mixing Mechanisms*, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2eeff35664016c7f0f8aa704f0d9a83e-Abstract-Conference.html) already models positional, lexical, reflexive bound-entity retrieval; [Shuang et al., Findings ACL 2024](https://aclanthology.org/2024.findings-acl.328/) explicitly studies cross-event argument confusion. A basic extra-entity/interference claim is not a cleared novelty contribution.

**Status: code checked and reviewable, source-role result unknown, historical seven-error replication gate FAILED, protected/raw original archive access OPEN, no model runs, no paid inference, no merge, no mechanism claim.**