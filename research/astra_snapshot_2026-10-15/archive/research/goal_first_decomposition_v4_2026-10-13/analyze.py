"""Audit the corrected, frozen v4 paired experiment without rerunning models."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (11, 23, 37, 41, 53)
ARMS = ("joint_replay", "clause_decomposition")
SPLITS = ("development", "in_domain_test", "mixed_test")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows):
    return {
        "n": len(rows),
        "full_both": sum(row["both_correct"] for row in rows),
        "normalized_both": sum(row["masked_both_correct"] for row in rows),
        "full_step1": sum(row["pred"][0] == row["gold"][0] for row in rows),
        "full_step2": sum(row["pred"][1] == row["gold"][1] for row in rows),
        "normalized_step1": sum(row["masked_pred"][0] == row["gold"][0] for row in rows),
        "normalized_step2": sum(row["masked_pred"][1] == row["gold"][1] for row in rows),
        "full_same_direction": sum(row["pred"][0] == row["pred"][1] for row in rows),
        "normalized_same_direction": sum(row["masked_pred"][0] == row["masked_pred"][1] for row in rows),
    }


def main():
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    protocol_hash = sha256(ROOT / "PROTOCOL.json")
    assert protocol["padding_token_id"] == 2
    assert tuple(protocol["seeds"]) == SEEDS
    out = {"protocol_sha256": protocol_hash, "paired": {}}
    for seed in SEEDS:
        pair = {}
        ids_by_split = {}
        gold_by_split = {}
        for arm in ARMS:
            train = json.loads((ROOT / f"TRAIN_{seed}_{arm}.json").read_text())
            evaluated = json.loads((ROOT / f"EVAL_{seed}_{arm}.json").read_text())
            model_hash = sha256(ROOT / f"MODEL_{seed}_{arm}.pt")
            assert train["protocol_sha256"] == evaluated["protocol_sha256"] == protocol_hash
            assert train["checkpoint_sha256"] == evaluated["checkpoint_sha256"] == model_hash
            assert train["seed"] == evaluated["seed"] == seed
            assert train["arm"] == evaluated["arm"] == arm
            assert set(evaluated["records"]) == set(SPLITS)
            pair[arm] = {"updates": train["updates"], "labels_seen": train["labels_seen"], "splits": {}}
            for split in SPLITS:
                rows = evaluated["records"][split]
                assert len(rows) == 400
                assert all(row["both_correct"] == (row["pred"] == row["gold"]) for row in rows)
                assert all(row["masked_both_correct"] == (row["masked_pred"] == row["gold"]) for row in rows)
                ids = [row["id"] for row in rows]
                gold = [row["gold"] for row in rows]
                assert len(set(ids)) == len(ids)
                if split not in ids_by_split:
                    ids_by_split[split], gold_by_split[split] = ids, gold
                else:
                    assert ids_by_split[split] == ids and gold_by_split[split] == gold
                pair[arm]["splits"][split] = counts(rows)
        assert pair[ARMS[0]]["updates"] == pair[ARMS[1]]["updates"]
        assert pair[ARMS[0]]["labels_seen"] == pair[ARMS[1]]["labels_seen"]
        out["paired"][str(seed)] = pair
    in_domain_ok = all(
        out["paired"][str(seed)][arm]["splits"]["in_domain_test"]["full_both"] >= 360
        for seed in SEEDS for arm in ARMS
    )
    full_gains = []
    normalized_gains = []
    full_positive_starts = 0
    normalized_positive_starts = 0
    for seed in SEEDS:
        pair = out["paired"][str(seed)]
        control, treatment = (pair[arm]["splits"] for arm in ARMS)
        full_gain = treatment["mixed_test"]["full_both"] - control["mixed_test"]["full_both"]
        normalized_gain = treatment["mixed_test"]["normalized_both"] - control["mixed_test"]["normalized_both"]
        full_gains.append(full_gain)
        normalized_gains.append(normalized_gain)
        if full_gain >= 40 and treatment["in_domain_test"]["full_both"] >= control["in_domain_test"]["full_both"] - 10:
            full_positive_starts += 1
        if treatment["mixed_test"]["normalized_both"] >= 380:
            normalized_positive_starts += 1
    out["screens"] = {
        "all_starts_in_domain_full_pass": in_domain_ok,
        "full_mixed_gains_per_400": full_gains,
        "full_positive_starts": full_positive_starts,
        "end_to_end_gate_pass": in_domain_ok and full_positive_starts >= 4,
        "normalized_mixed_gains_per_400": normalized_gains,
        "normalized_mean_gain_per_400": sum(normalized_gains) / len(normalized_gains),
        "normalized_positive_starts": normalized_positive_starts,
        "supplied_normalization_gate_pass": in_domain_ok and normalized_positive_starts >= 4 and sum(normalized_gains) / len(normalized_gains) >= 40,
    }
    path = ROOT / "ANALYSIS.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["screens"], indent=2))


if __name__ == "__main__":
    main()
