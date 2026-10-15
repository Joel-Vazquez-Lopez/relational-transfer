"""Audit typed goal input versus constant tag on fresh2."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARTS = ("fresh1", "fresh3")
ARMS = ("typed_goal", "constant_tag")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows):
    assert len(rows) == 160
    ids = {r["id"] for r in rows}
    assert len(ids) == 40
    by_key = {(r["id"], r["role"], r["order"]): r for r in rows}
    assert len(by_key) == 160
    for item in ids:
        assert {(role, order) for (id_, role, order) in by_key if id_ == item} == {
            ("actor", 0), ("actor", 1), ("patient", 0), ("patient", 1)}
    return {
        "items": 40, "prompts": 160,
        "all_four": sum(all(by_key[(item, role, order)]["correct"]
                            for role in ("actor", "patient") for order in (0, 1)) for item in ids),
        "exact": sum(r["correct"] for r in rows),
        "parsed": sum(r["parsed"] for r in rows),
        "tag_correct_prompts": sum(r["tag_correct"] for r in rows),
        "both_fields_correct": sum(r["actor_correct"] and r["patient_correct"] for r in rows),
        "routed_exact": sum(r["routed_correct"] for r in rows),
        "answer_matches_field": sum(r["answer_matches_field"] for r in rows),
        "by_role_order": {f"{role}_{order}": sum(r["correct"] for r in rows
                                                    if r["role"] == role and r["order"] == order)
                          for role in ("actor", "patient") for order in (0, 1)},
    }


def paired_bootstrap(a_rows, b_rows):
    """Descriptive interval resampling the 40 source sentences, not 160 prompts."""
    paired = {}
    for a, b in zip(a_rows, b_rows):
        assert (a["id"], a["role"], a["order"]) == (b["id"], b["role"], b["order"])
        paired.setdefault(a["id"], []).append((int(a["correct"]), int(b["correct"])))
    assert len(paired) == 40 and all(len(v) == 4 for v in paired.values())
    events = list(paired.values())
    rng = random.Random(20261013)
    exact_differences, joint_differences = [], []
    for _ in range(10000):
        sample = [events[rng.randrange(40)] for _ in range(40)]
        exact_differences.append(sum(sum(a-b for a, b in e) for e in sample) / 160)
        joint_differences.append(sum(int(all(a for a, _ in e)) - int(all(b for _, b in e))
                                     for e in sample) / 40)
    exact_differences.sort()
    joint_differences.sort()
    return {"exact_difference_95pct": [exact_differences[250], exact_differences[9750]],
            "all_four_difference_95pct": [joint_differences[250], joint_differences[9750]],
            "method": "10000 paired bootstrap resamples of 40 sentence-level items, seed 20261013; descriptive only"}


def main():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    phash = sha(ROOT / "PROTOCOL.json")
    assert tuple(cfg["starts"]) == STARTS and tuple(cfg["arms"]) == ARMS
    out = {"protocol_sha256": phash, "paired": {}, "screens": {}}
    row_ids = None
    for start in STARTS:
        out["paired"][start] = {}
        saved_rows = {}
        for arm in ARMS:
            cp = ROOT / f"{start}_{arm}" / "adapter.pt"
            train = json.loads((cp.parent / "TRAINING.json").read_text())
            ev = json.loads((ROOT / f"EVAL_{start}_{arm}_fresh2.json").read_text())
            assert train["protocol_sha256"] == ev["protocol_sha256"] == phash
            assert train["checkpoint_sha256"] == ev["checkpoint_sha256"] == sha(cp)
            assert train["updates"] == cfg["updates_per_checkpoint"] == 160
            assert train["start"] == ev["start"] == start
            assert train["arm"] == ev["arm"] == arm
            these = [(r["id"], r["role"], r["order"], r["gold"]) for r in ev["records"]]
            if row_ids is None:
                row_ids = these
            else:
                assert row_ids == these
            out["paired"][start][arm] = {"training": {"updates": train["updates"],
                                                         "target_tokens": train["target_tokens"]},
                                          "test": counts(ev["records"])}
            saved_rows[arm] = ev["records"]
        a = out["paired"][start]["typed_goal"]
        b = out["paired"][start]["constant_tag"]
        assert a["training"]["target_tokens"] == b["training"]["target_tokens"]
        at, bt = a["test"], b["test"]
        out["screens"][start] = {
            "all_four_gain": at["all_four"] - bt["all_four"],
            "exact_gain": at["exact"] - bt["exact"],
            "both_fields_difference": at["both_fields_correct"] - bt["both_fields_correct"],
            "pass": (at["all_four"] - bt["all_four"] >= 8
                     and at["exact"] - bt["exact"] >= 20
                     and at["both_fields_correct"] >= bt["both_fields_correct"] - 8
                     and at["tag_correct_prompts"] >= 152),
            "paired_bootstrap": paired_bootstrap(saved_rows["typed_goal"], saved_rows["constant_tag"]),
        }
    out["screen_pass"] = all(x["pass"] for x in out["screens"].values())
    (ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"paired": out["paired"], "screens": out["screens"],
                      "screen_pass": out["screen_pass"]}, indent=2))


if __name__ == "__main__":
    main()
