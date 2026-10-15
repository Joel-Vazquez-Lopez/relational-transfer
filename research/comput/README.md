# Comput: falsification workstream (15 Oct 2026)

Status: deterministic, synthetic-only diagnostic. This is **not** an independent replication of historical model outputs. The archive with raw predictions is not in GitHub; see `evidence/HISTORICAL_DATA_ACCESS.md` on `main`.

Run: `python3 -m unittest discover -s research/comput -p 'test_*.py' -v`

Run: `python3 research/comput/identifiability.py research/comput/controlled_design.jsonl`

Objective: test whether first/last participant, grammatical subject and question-conditioned position mechanisms are identifiable on a given stimulus design. Never use gold role labels as predictor features.

Result: two-question subject-first design makes three pairs of rules indistinguishable. Counterbalancing grammatical subject (first/last) and question (actor/patient) separates all seven deterministic rules. A greedy three-row subset separates the seven signatures. This is a design-level calculation and does not explain LLM behavior without archived event-level predictions.

Limitation: the greedy algorithm is not a proof of global minimum for arbitrary designs. Reference implementation uses *two named participants* and provided subject annotations. No sealed data accessed.
