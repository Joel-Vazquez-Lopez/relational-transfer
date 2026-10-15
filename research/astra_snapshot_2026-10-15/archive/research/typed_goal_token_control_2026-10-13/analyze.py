"""Audit the frozen 20-sentence token-matched control comparison."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TYPED = ROOT.parent / "typed_goal_conditioning_2026-10-13"
STARTS = ("fresh1", "fresh3")
ARMS = ("typed_goal", "null_two_token")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def counts(rows):
    assert len(rows) == 80
    ids = {r["id"] for r in rows}
    assert len(ids) == 20
    by_key = {(r["id"], r["role"], r["order"]): r for r in rows}
    assert len(by_key) == 80
    assert all({(role, order) for (id_, role, order) in by_key if id_ == item} ==
               {("actor", 0), ("actor", 1), ("patient", 0), ("patient", 1)} for item in ids)
    return {
        "items": 20, "prompts": 80,
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
    paired = {}
    for a, b in zip(a_rows, b_rows):
        assert (a["id"], a["role"], a["order"]) == (b["id"], b["role"], b["order"])
        paired.setdefault(a["id"], []).append((int(a["correct"]), int(b["correct"])))
    assert len(paired) == 20 and all(len(v) == 4 for v in paired.values())
    events = list(paired.values())
    rng = random.Random(20261013)
    exact_differences, joint_differences = [], []
    for _ in range(10000):
        sample = [events[rng.randrange(20)] for _ in range(20)]
        exact_differences.append(sum(sum(a-b for a, b in e) for e in sample) / 80)
        joint_differences.append(sum(int(all(a for a, _ in e)) - int(all(b for _, b in e))
                                     for e in sample) / 20)
    exact_differences.sort()
    joint_differences.sort()
    return {"exact_difference_95pct": [exact_differences[250], exact_differences[9750]],
            "all_four_difference_95pct": [joint_differences[250], joint_differences[9750]],
            "method": "10000 paired bootstrap resamples of 20 sentence-level items; descriptive only"}


def main():
    cfg = load(ROOT / "PROTOCOL.json")
    phash = sha(ROOT / "PROTOCOL.json")
    assert tuple(cfg["starts"]) == STARTS and tuple(cfg["arms"]) == ARMS
    assert cfg["token_counts"]["typed_goal"] == cfg["token_counts"]["null_two_token"]
    out = {"protocol_sha256": phash, "paired": {}, "screens": {}}
    row_ids = None
    for start in STARTS:
        out["paired"][start] = {}
        saved_rows = {}
        for arm in ARMS:
            cp = (TYPED / f"{start}_typed_goal" / "adapter.pt" if arm == "typed_goal"
                  else ROOT / f"{start}_null_two_token" / "adapter.pt")
            train_path = cp.parent / "TRAINING.json"
            train = load(train_path)
            ev = load(ROOT / f"EVAL_{start}_{arm}_fresh3.json")
            assert ev["protocol_sha256"] == phash
            assert ev["checkpoint_sha256"] == sha(cp)
            assert train["checkpoint_sha256"] == sha(cp)
            assert train["updates"] == 160
            assert train["target_tokens"] == cfg["token_counts"][arm]["supervised_target_tokens"]
            assert ev["start"] == start and ev["arm"] == arm
            these = [(r["id"], r["role"], r["order"], r["gold"]) for r in ev["records"]]
            if row_ids is None:
                row_ids = these
            else:
                assert row_ids == these
            out["paired"][start][arm] = {"training": {"updates": train["updates"],
                "target_tokens": train["target_tokens"]}, "test": counts(ev["records"])}
            saved_rows[arm] = ev["records"]
        at = out["paired"][start]["typed_goal"]["test"]
        bt = out["paired"][start]["null_two_token"]["test"]
        out["screens"][start] = {
            "all_four_gain": at["all_four"] - bt["all_four"],
            "exact_gain": at["exact"] - bt["exact"],
            "both_fields_difference": at["both_fields_correct"] - bt["both_fields_correct"],
            "pass": (at["all_four"] - bt["all_four"] >= 4
                     and at["exact"] - bt["exact"] >= 10
                     and at["both_fields_correct"] >= bt["both_fields_correct"] - 4
                     and at["tag_correct_prompts"] >= 38),
            "paired_bootstrap": paired_bootstrap(saved_rows["typed_goal"],
                                                   saved_rows["null_two_token"]),
        }
    out["screen_pass"] = all(x["pass"] for x in out["screens"].values())
    (ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
