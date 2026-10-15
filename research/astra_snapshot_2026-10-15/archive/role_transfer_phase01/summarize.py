"""Summarize all-pair and saved-trajectory development results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import networkx as nx

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "SUMMARY.json"


def summarize_all_pairs() -> dict:
    data = json.loads((ROOT / "all_pair_results.json").read_text())
    audit = json.loads((ROOT / "phase0_audit.json").read_text())
    graphs = {(x["cohort"], x["world"], x["condition"]): x for x in audit["records"]}
    models = []
    pair_records = []
    for item in data["models"]:
        key = (item["cohort"], item["world"], item["condition"])
        graph_info = graphs[key]
        graph = nx.Graph()
        graph.add_nodes_from(range(10))
        graph.add_edges_from(graph_info["labeled_edges"])
        unseen = [p for p in item["pairs"] if not p["observed"]]
        seen = [p for p in item["pairs"] if p["observed"]]
        assert len(unseen) == 25 and len(seen) == 20
        def avg(rows: list[dict], metric: str) -> float:
            return mean(p[metric] for p in rows)
        weak = [p for p in unseen if p["binary_role_accuracy"] < .8]
        focal = next(p for p in unseen if p["pair"] == [0, 1])
        models.append({k: item[k] for k in ("cohort", "world", "condition", "seed")} |
                      {"unseen_role_mean": avg(unseen, "binary_role_accuracy"),
                       "unseen_four_choice_mean": avg(unseen, "four_choice_accuracy"),
                       "unseen_event_mean": avg(unseen, "event_accuracy"),
                       "seen_role_mean": avg(seen, "binary_role_accuracy"),
                       "seen_four_choice_mean": avg(seen, "four_choice_accuracy"),
                       "unseen_weak_pair_count": len(weak),
                       "unseen_role_min": min(p["binary_role_accuracy"] for p in unseen),
                       "focal_role": focal["binary_role_accuracy"],
                       "focal_four_choice": focal["four_choice_accuracy"]})
        for p in unseen:
            a, b = p["pair"]
            pair_records.append({"cohort": item["cohort"], "world": item["world"],
                                 "condition": item["condition"], "seed": item["seed"],
                                 "pair": p["pair"], "role": p["binary_role_accuracy"],
                                 "four_choice": p["four_choice_accuracy"],
                                 "event": p["event_accuracy"],
                                 "weak": p["binary_role_accuracy"] < .8,
                                 "contains_identity_0_or_1": 0 in p["pair"] or 1 in p["pair"],
                                 "common_neighbors": len(set(graph[a]) & set(graph[b])),
                                 "shortest_path": nx.shortest_path_length(graph, a, b)})
    cohorts = {}
    for name in ("screen", "confirmation_history"):
        mm = [m for m in models if m["cohort"] == name]
        pp = [p for p in pair_records if p["cohort"] == name]
        cohorts[name] = {
            "models": len(mm), "unseen_model_pair_cases": len(pp),
            "mean_of_model_unseen_role_accuracy": mean(m["unseen_role_mean"] for m in mm),
            "mean_of_model_unseen_four_choice_accuracy": mean(m["unseen_four_choice_mean"] for m in mm),
            "mean_of_model_unseen_event_accuracy": mean(m["unseen_event_mean"] for m in mm),
            "mean_of_model_seen_four_choice_accuracy": mean(m["seen_four_choice_mean"] for m in mm),
            "weak_pair_cases": sum(p["weak"] for p in pp),
            "models_with_any_weak_pair": sum(m["unseen_weak_pair_count"] > 0 for m in mm),
            "models_with_weak_focal_pair": sum(m["focal_role"] < .8 for m in mm),
            "weak_pair_identity_0_or_1": sum(p["weak"] and p["contains_identity_0_or_1"] for p in pp),
            "weak_pairs_by_common_neighbors": dict(Counter(p["common_neighbors"] for p in pp if p["weak"])),
            "all_pairs_by_common_neighbors": dict(Counter(p["common_neighbors"] for p in pp)),
            "weak_pairs_by_shortest_path": dict(Counter(p["shortest_path"] for p in pp if p["weak"])),
            "all_pairs_by_shortest_path": dict(Counter(p["shortest_path"] for p in pp)),
        }
    worlds = {}
    for cohort, world in sorted({(m["cohort"], m["world"]) for m in models}):
        mm = [m for m in models if m["cohort"] == cohort and m["world"] == world]
        worlds[f"{cohort}/{world}"] = {
            "models": len(mm), "mean_unseen_role": mean(m["unseen_role_mean"] for m in mm),
            "weak_pair_cases": sum(m["unseen_weak_pair_count"] for m in mm),
            "models_with_any_weak_pair": sum(m["unseen_weak_pair_count"] > 0 for m in mm),
            "mean_focal_role": mean(m["focal_role"] for m in mm),
        }
    weakest = sorted(pair_records, key=lambda r: r["role"])[:30]
    return {"cohorts": cohorts, "worlds": worlds, "models": models,
            "weakest_30_model_pairs": weakest,
            "global": {"model_count": len(models), "unseen_model_pair_cases": len(pair_records),
                       "weak_pair_cases": sum(p["weak"] for p in pair_records),
                       "median_unseen_pair_role": median(p["role"] for p in pair_records)}}


def summarize_trajectories() -> dict | None:
    path = ROOT / "trajectory_results.json"
    if not path.exists():
        return None
    models = json.loads(path.read_text())["models"]
    output = []
    for m in models:
        rows = []
        for stage in m["stages"]:
            seen = [p for p in stage["pairs"] if p["observed"]]
            unseen = [p for p in stage["pairs"] if not p["observed"]]
            focal = next(p for p in unseen if p["pair"] == [0, 1])
            rows.append({"step": stage["step"],
                         "seen_four_choice": mean(p["four_choice_accuracy"] for p in seen),
                         "unseen_four_choice": mean(p["four_choice_accuracy"] for p in unseen),
                         "unseen_role": mean(p["binary_role_accuracy"] for p in unseen),
                         "unseen_weak_pair_count": sum(p["binary_role_accuracy"] < .8 for p in unseen),
                         "focal_role": focal["binary_role_accuracy"],
                         "focal_four_choice": focal["four_choice_accuracy"]})
        output.append({"condition": m["condition"], "seed": m["seed"], "stages": rows})
    return {"scope": "new2 only, selected component replay, developmental", "models": output}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    result = {"all_pairs": summarize_all_pairs(), "trajectories": summarize_trajectories()}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cohorts": result["all_pairs"]["cohorts"],
                      "worlds": result["all_pairs"]["worlds"],
                      "trajectories": result["trajectories"]}, indent=2))


if __name__ == "__main__":
    main()
