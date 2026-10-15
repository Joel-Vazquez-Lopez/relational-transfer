"""Post hoc, read-only reanalysis of existing development predictions.

No training, model inference, or sealed-set access. Never use these estimates as
independent confirmation. Printed role fields are outputs, not hidden states.
"""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def norm(x):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (x or "").casefold())).strip()


def audit(path):
    data = json.loads(path.read_text())
    rows = data["records"]
    counts = defaultdict(int)
    pairs = defaultdict(dict)
    seen = set()
    for r in rows:
        key = (r["id"], r.get("order", 0), r["role"])
        assert key not in seen
        seen.add(key)
        field_correct = bool(r["actor_correct"] if r["role"] == "actor" else r["patient_correct"])
        answer_correct = bool(r["correct"])
        counts[f"field_{int(field_correct)}_answer_{int(answer_correct)}"] += 1
        counts["both_fields_correct"] += bool(r["actor_correct"] and r["patient_correct"])
        counts["answer_correct"] += answer_correct
        counts["parsed"] += bool(r["parsed"])
        counts["rows"] += 1
        pairs[key[:2]][r["role"]] = r
    for pair in pairs.values():
        assert set(pair) == {"actor", "patient"}
        a, p = pair["actor"], pair["patient"]
        counts["question_pairs"] += 1
        both_parse = all(r["actor"] is not None and r["patient"] is not None for r in (a, p))
        counts["question_pairs_with_both_role_outputs"] += both_parse
        changed = both_parse and (norm(a["actor"]), norm(a["patient"])) != (norm(p["actor"]), norm(p["patient"]))
        counts["role_fields_change_with_question"] += bool(changed)
        counts["both_answers_correct"] += bool(a["correct"] and p["correct"])
    return {"source": str(path.relative_to(PROJECT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "counts": dict(counts)}


def main():
    patterns = {
        "candidate_free_role_binding_2026-10-14": ["EVAL_*.json"],
        "typed_goal_token_control_2026-10-13": ["EVAL_*.json"],
        "natural_role_verifier_development_2026-10-13": [
            "EVAL_*_role_first_development_normal.json",
            "EVAL_*_verifier_noisy_development_swapped.json"],
    }
    records = []
    for directory, globs in patterns.items():
        for glob in globs:
            records.extend(audit(p) for p in sorted((PROJECT / "research" / directory).glob(glob)))
    assert len(records) == 12, len(records)
    result = {"status": "post_hoc_reanalysis_of_existing_development_outputs",
              "limitations": "Question comparisons change the requested role and wording, and in typed arms the inferred cue. This is an output-level association, not isolated causal query interference. Units share sentence IDs, source distribution and pretrained family. No hidden state or new confirmation data is used.",
              "records": records}
    (ROOT / "CACHED_OUTPUT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    for x in records:
        c = x["counts"]
        print(Path(x["source"]).name, "correct-field/wrong-answer", c.get("field_1_answer_0", 0),
              "/", c.get("field_1_answer_0", 0)+c.get("field_1_answer_1", 0),
              "field-change", c.get("role_fields_change_with_question", 0), "/",
              c["question_pairs_with_both_role_outputs"])


if __name__ == "__main__":
    main()
