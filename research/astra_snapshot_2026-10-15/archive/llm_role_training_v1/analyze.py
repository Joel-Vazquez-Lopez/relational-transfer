"""Read-only analysis of the fixed local language-model pilot."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ("pretrained_no_finetuning", "unordered_identities", "ordered_roles")


def load(arm):
    return json.loads((ROOT / arm / "result.json").read_text())


def key(row):
    return (tuple(row["pair"]), row["actor"], row["patient"], row["verb"],
            row["voice"], row["role"], row["wording"])


def main():
    results = {arm: load(arm) for arm in ARMS}
    maps = {arm: {key(row): row for row in result["evaluation"]["rows"]}
            for arm, result in results.items()}
    assert all(len(rows) == 2160 for rows in maps.values())
    assert len(set(map(tuple, (maps[arm].keys() for arm in ARMS)))) == 1
    ordered = maps["ordered_roles"]
    unordered = maps["unordered_identities"]
    pair_delta = defaultdict(list)
    for k, row in ordered.items():
        if not row["seen"]:
            pair_delta[k[0]].append(int(row["correct"]) - int(unordered[k]["correct"]))
    pair_summary = {str(pair): sum(values) / len(values)
                    for pair, values in sorted(pair_delta.items())}
    gains = sum(value > 0 for value in pair_summary.values())
    ties = sum(value == 0 for value in pair_summary.values())
    losses = sum(value < 0 for value in pair_summary.values())

    def consistency(rows, field, values):
        groups = defaultdict(dict)
        for row in rows.values():
            if row["seen"]:
                continue
            group = (tuple(row["pair"]), row["actor"], row["patient"], row["verb"],
                     row["role"], row["wording"] if field == "voice" else row["voice"])
            groups[group][row[field]] = row
        assert len(groups) == 600
        assert all(set(v) == set(values) for v in groups.values())
        first, second = values
        same = sum(v[first]["answer"] == v[second]["answer"]
                   for v in groups.values())
        both_correct = sum(v[first]["correct"] and v[second]["correct"]
                           for v in groups.values())
        return {"same_answer": same, "both_correct": both_correct, "total": 600}

    pair_values = np.array(list(pair_summary.values()))
    rng = np.random.default_rng(820107)
    draws = rng.choice(pair_values, size=(100000, len(pair_values)), replace=True).mean(axis=1)

    report = {
        "protocol_sha256": {arm: results[arm]["protocol_sha256"] for arm in ARMS},
        "paired_initialization": (results["ordered_roles"]["training"]["initial_trainable_sha256"]
                                  == results["unordered_identities"]["training"]["initial_trainable_sha256"]),
        "accuracy": {arm: {
            "seen": results[arm]["evaluation"]["summary"]["(True,)"]["accuracy"],
            "unseen": results[arm]["evaluation"]["summary"]["(False,)"]["accuracy"],
        } for arm in ARMS},
        "unseen_pair_ordered_minus_unordered": pair_summary,
        "pair_directions": {"gains": gains, "ties": ties, "losses": losses},
        "ordered_minus_unordered_unseen_difference": float(pair_values.mean()),
        "exploratory_pair_bootstrap_95_interval": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "unseen_active_passive_consistency": {
            arm: consistency(maps[arm], "voice", ("active", "passive")) for arm in ARMS},
        "unseen_wording_consistency": {
            arm: consistency(maps[arm], "wording", ("generic", "verb_specific")) for arm in ARMS},
        "unseen_by_voice_role_wording": {},
    }
    for arm in ARMS:
        groups = defaultdict(list)
        for row in maps[arm].values():
            if not row["seen"]:
                groups[f'{row["voice"]}/{row["role"]}/{row["wording"]}'].append(row)
        report["unseen_by_voice_role_wording"][arm] = {
            label: sum(r["correct"] for r in rows) / len(rows)
            for label, rows in sorted(groups.items())}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
