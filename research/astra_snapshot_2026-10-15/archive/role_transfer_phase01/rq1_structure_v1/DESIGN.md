# RQ1: observed-relation network contrast

Protocol and graph manifest were frozen on 8 October 2026 before any model in
this study was trained. The experiment asks whether a different arrangement of
the same amount of relational training evidence changes transfer to unfamiliar
entity pairs. The ordered-role auxiliary objective is not used here.

Each graph has ten entities, twenty observed unordered actor/patient pairs, and
four partners for every entity. Each observed pair occurs in both role
directions and with both events. Eight new graphs with at most four triangles
are compared with eight new graphs with at least eight triangles. All sixteen
are different from each other and from earlier training graph classes up to
graph isomorphism. The two graphs in each comparison use the same permutation
of entity names and the same three training seeds. Both use the corrected
balanced-distractor generator, architecture, optimizer, and training budget.

The primary evaluation uses only entity pairs withheld in **both** graphs of
each comparison. This holds the tested identities and rendered test items fixed
within a graph pair. The primary score is mean binary actor/patient accuracy,
first across those shared withheld pairs, then across three starts, and then
across the eight graph pairs. The frozen direction predicts that fewer triangles
could help by spreading observed partners more broadly. The comparison tests
that prediction; it does not establish that triangles are the mechanism.

The graph pair is the independent unit for the prespecified two-sided sign-flip
test. Per-pair results, all starts, all 25 unseen-pair scores, familiar-pair
accuracy, event accuracy, and the second rendering suite will be retained and
reported. No graph or seed is replaced because its result is inconvenient.

The exact choices, thresholds, source hashes, and limits are in
`PROTOCOL.json` and `GRAPH_MANIFEST.json`. Source code is
`../rq1_structure_v1.py` and `../run_rq1_structure_v1.py`.
