# Valid-swap route identifiability falsification

Read with Brain-O Experiments 021–029, Compass Investigations 10–11, Comput Report 011 and Astra's interface/shortcut audits.

This deterministic **negative identification control** enumerates 24 two-event worlds and four event/role questions per world. Each intervention swaps the entire queried event's actor/patient assignment while retaining a valid world. Four known policies answer every clean question correctly: cache-only, raw reconstruction, an intentionally equivalent source shortcut, and a hybrid.

Results on 96 world-question cases per policy:
- Cache: 96 cache-sensitive, 0 source-sensitive.
- Reconstruction: 0 cache-sensitive, 96 source-sensitive.
- Source shortcut: 0 cache-sensitive, 96 source-sensitive (identical signatures to reconstruction).
- Hybrid: 48 cache-sensitive, 48 source-sensitive.

The shortcut intentionally implements the same *observable* source lookup as the reconstruction policy. This is an identifiability counterexample, not proof that a particular LLM uses a shortcut or a demonstration of a distinct algorithm. A port-level intervention identifies which source is used, **not the internal algorithm implemented inside a source pathway**.

Run `python3 -m unittest discover -s research/coordinator_identifiability -p 'test_*.py' -v` and `python3 research/coordinator_identifiability/route_falsification.py`. Three analogous local tests were executed and passed in a standalone directory; the separately authored GitHub copy must be independently tested by CI/review. Python standard library only.

**Next gate:** competence-matched learned models with identical source access, blinded route schedules, a known intermediate to intervene on, varied non-position-solvable language and unmodified authentic validation. This does not constitute a new pretrained mechanism or ACL novelty. No source archive, frozen data, or other branches modified.