"""Summarize frozen-checkpoint role mechanism diagnostics across graphs."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "role_mechanism_v4" / "SUMMARY.json"


def run() -> dict:
    graph_rows = []
    for index in range(8):
        world = f"fresh{index}"
        v1 = json.loads((ROOT / "role_mechanism_v1" / f"{world}.json").read_text())
        v3 = json.loads((ROOT / "role_mechanism_v3" / f"{world}.json").read_text())
        v4 = json.loads((ROOT / "role_mechanism_v4" / f"{world}.json").read_text())
        first = {r["arm"]: r for r in v1["records"]}
        row = {"world": world,
               "first_seed_probe_final_joint": {
                   arm: r["probe"]["final"]["joint_accuracy"]
                   for arm, r in first.items()}}
        row["first_seed_probe_gain"] = (
            row["first_seed_probe_final_joint"]["ordered_roles"]
            - row["first_seed_probe_final_joint"]["unordered_identities"])
        replicate = []
        for seed in (271002, 271003):
            arms = {r["arm"]: r for r in v3["records"] if r["seed"] == seed}
            drops = {arm: r["results"]["ordinary"]["mean_margin"]
                     - r["results"]["role_order"]["mean_margin"]
                     for arm, r in arms.items()}
            replicate.append(drops["ordered_roles"] - drops["unordered_identities"])
        row["held_aside_seed_margin_drop_contrast"] = mean(replicate)
        arms = {}
        for arm in ("unordered_identities", "ordered_roles"):
            records = [r for r in v4["records"] if r["arm"] == arm]
            assert len(records) == 3
            arms[arm] = {
                "ordinary_margin": mean(r["ordinary"]["mean_margin"] for r in records),
                "role_swap_margin": mean(r["role_order"]["mean_margin"] for r in records),
                "matched_random_margin": mean(mean(c["mean_margin"] for c in r["matched_energy_random"])
                                              for r in records),
                "ordinary_binary_accuracy": mean(r["ordinary"]["binary_role_accuracy"] for r in records),
                "role_swap_binary_accuracy": mean(r["role_order"]["binary_role_accuracy"] for r in records),
                "matched_random_binary_accuracy": mean(mean(c["binary_role_accuracy"]
                                                            for c in r["matched_energy_random"])
                                                       for r in records),
            }
            arms[arm]["matched_random_minus_role_margin"] = (
                arms[arm]["matched_random_margin"] - arms[arm]["role_swap_margin"])
            arms[arm]["matched_random_minus_role_binary_accuracy"] = (
                arms[arm]["matched_random_binary_accuracy"] - arms[arm]["role_swap_binary_accuracy"])
        row["matched_energy_arms"] = arms
        row["matched_energy_ordered_minus_unordered_margin_selectivity"] = (
            arms["ordered_roles"]["matched_random_minus_role_margin"]
            - arms["unordered_identities"]["matched_random_minus_role_margin"])
        graph_rows.append(row)
    output = {
        "status": "exploratory_mechanism_diagnostic_complete",
        "graphs": 8,
        "probe_checkpoints": 16,
        "held_aside_seed_checkpoints": 32,
        "matched_energy_checkpoints": 48,
        "graph_rows": graph_rows,
        "means": {
            "first_seed_probe_gain": mean(r["first_seed_probe_gain"] for r in graph_rows),
            "held_aside_seed_margin_drop_contrast": mean(
                r["held_aside_seed_margin_drop_contrast"] for r in graph_rows),
            "matched_energy_ordered_minus_unordered_margin_selectivity": mean(
                r["matched_energy_ordered_minus_unordered_margin_selectivity"] for r in graph_rows),
            "matched_energy_arms": {
                arm: {metric: mean(r["matched_energy_arms"][arm][metric] for r in graph_rows)
                      for metric in graph_rows[0]["matched_energy_arms"][arm]}
                for arm in ("unordered_identities", "ordered_roles")},
        },
        "interpretation_rule_passed": (
            sum(r["held_aside_seed_margin_drop_contrast"] > 0 for r in graph_rows) >= 7
            and mean(r["held_aside_seed_margin_drop_contrast"] for r in graph_rows) > 1
            and sum(r["matched_energy_arms"]["ordered_roles"]
                    ["matched_random_minus_role_margin"] > 0 for r in graph_rows) >= 7),
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    result = run()
    print(json.dumps({"means": result["means"],
                      "interpretation_rule_passed": result["interpretation_rule_passed"]}, indent=2))
