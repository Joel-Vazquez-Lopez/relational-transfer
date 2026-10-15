"""Post-result diagnostic of existing EWT checkpoint answers.

This analysis does not change the frozen EWT measures or inspect the new
active/passive challenge. It is descriptive and exploratory.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "llm_role_independent_eval"
NAMES = ("fresh1_source", "fresh1_natural", "fresh3_source", "fresh3_natural")


def norm(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())


def flags(row: dict) -> dict:
    actor, other = row["direct"]["actor"], row["direct"]["other"]
    a, b, ga, gb = map(norm, (actor["raw"], other["raw"], actor["gold"], other["gold"]))
    return {
        "actor_exact_frozen": actor["exact"], "other_exact_frozen": other["exact"],
        "both_exact_frozen": actor["exact"] and other["exact"],
        "same_response_to_both_questions": bool(a) and a == b,
        "other_question_repeats_actor_gold": b == ga,
        "actor_question_returns_other_gold": a == gb,
        "other_raw_contains_gold_as_subphrase": bool(gb) and gb in b and gb != b,
        "other_gold_contains_raw_as_subphrase": bool(b) and b in gb and gb != b,
        "candidate_correct": row["primary"] > 0,
        "omitted_candidate_correct": row["omitted"] > 0,
        "paraphrase_candidate_correct": row["paraphrase"] > 0,
    }


def main() -> None:
    raw = {name: json.loads((ROOT / f"{name}.json").read_text())["records"] for name in NAMES}
    ids = [r["id"] for r in raw[NAMES[0]]]
    assert all([r["id"] for r in raw[name]] == ids for name in NAMES)
    by_name = {name: {r["id"]: flags(r) for r in rows} for name, rows in raw.items()}
    summary = {name: dict(Counter(k for row in by_name[name].values() for k, v in row.items() if v))
               for name in NAMES}
    paired = {}
    for before, after in (("fresh1_source", "fresh1_natural"),
                          ("fresh3_source", "fresh3_natural")):
        transitions = {}
        for measure in ("candidate_correct", "actor_exact_frozen", "other_exact_frozen",
                        "both_exact_frozen", "same_response_to_both_questions",
                        "other_question_repeats_actor_gold"):
            old = by_name[before]
            new = by_name[after]
            transitions[measure] = {
                "false_to_true": sum(not old[i][measure] and new[i][measure] for i in ids),
                "true_to_false": sum(old[i][measure] and not new[i][measure] for i in ids),
                "true_in_both": sum(old[i][measure] and new[i][measure] for i in ids),
            }
        paired[f"{before}_to_{after}"] = transitions
    output = {"status": "exploratory_post_result_diagnostic", "n_items": len(ids),
              "summary": summary, "matched_transitions": paired,
              "limits": "All items are EWT grammatical subject/object examples, not human-verified semantic agent/patient examples. Some strict exact errors contain a correct span plus extra words. This analysis was selected after seeing EWT outcomes, and rows/starts share data."}
    target = ROOT / "POSTHOC_ERROR_DIAGNOSTIC.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
