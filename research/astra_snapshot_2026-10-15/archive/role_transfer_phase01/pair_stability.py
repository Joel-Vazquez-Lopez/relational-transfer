"""Describe how pair difficulty varies across seeds and graph relabelings."""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pair_stability.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    data = json.loads((ROOT / "all_pair_results.json").read_text())
    audit = json.loads((ROOT / "phase0_audit.json").read_text())
    by_graph = defaultdict(list)
    for model in data["models"]:
        by_graph[(model["cohort"], model["world"], model["condition"])].append(model)
    same_condition = []
    for key, models in by_graph.items():
        for first, second in itertools.combinations(models, 2):
            p = {tuple(x["pair"]): x for x in first["pairs"] if not x["observed"]}
            q = {tuple(x["pair"]): x for x in second["pairs"] if not x["observed"]}
            assert p.keys() == q.keys()
            a = np.array([p[k]["binary_role_accuracy"] for k in sorted(p)])
            b = np.array([q[k]["binary_role_accuracy"] for k in sorted(p)])
            aw = a < .8
            bw = b < .8
            same_condition.append({"cohort": key[0], "world": key[1], "condition": key[2],
                                   "seed_pair": [first["seed"], second["seed"]],
                                   "pair_correlation": float(np.corrcoef(a, b)[0, 1]),
                                   "weak_jaccard": float((aw & bw).sum() / (aw | bw).sum())
                                   if (aw | bw).any() else 1.0,
                                   "weak_both": int((aw & bw).sum())})
    # This comparison uses each world's canonical node identities, not the
    # differing zero/one relabelings seen by the model.
    info = {(r["cohort"], r["world"], r["condition"]): r for r in audit["records"]}
    cross_condition = []
    for cohort, world in sorted({(m["cohort"], m["world"]) for m in data["models"]}):
        for seed in sorted({m["seed"] for m in data["models"]
                            if m["cohort"] == cohort and m["world"] == world}):
            rows = {}
            for condition in ("shared1", "shared2"):
                model = next(m for m in data["models"] if m["cohort"] == cohort
                             and m["world"] == world and m["condition"] == condition
                             and m["seed"] == seed)
                original = info[(cohort, world, condition)]
                labels = {}
                # Recover old->new map from canonical and labeled edge lists
                # through the condition's recorded focus pair in source data.
                if cohort == "screen":
                    import all_pair_audit as phase
                    a, b = phase.SCREEN_WORLDS[world][condition]
                else:
                    import all_pair_audit as phase
                    a, b = phase.CONFIRM_WORLDS[world][condition]
                rest = [i for i in range(10) if i not in (a, b)]
                labels = {a: 0, b: 1, **{old: new for new, old in enumerate(rest, 2)}}
                inverse = {new: old for old, new in labels.items()}
                rows[condition] = {tuple(sorted((inverse[x["pair"][0]], inverse[x["pair"][1]]))):
                                   x["binary_role_accuracy"] for x in model["pairs"] if not x["observed"]}
            shared_keys = sorted(rows["shared1"].keys() & rows["shared2"].keys())
            a = np.array([rows["shared1"][k] for k in shared_keys])
            b = np.array([rows["shared2"][k] for k in shared_keys])
            cross_condition.append({"cohort": cohort, "world": world, "seed": seed,
                                    "shared_canonical_unseen_pairs": len(shared_keys),
                                    "canonical_pair_correlation": float(np.corrcoef(a, b)[0, 1]),
                                    "weak_jaccard": float(((a < .8) & (b < .8)).sum() /
                                                          ((a < .8) | (b < .8)).sum())
                                    if ((a < .8) | (b < .8)).any() else 1.0})
    result = {"status": "descriptive_development_analysis",
              "same_condition_seed_pairs": same_condition,
              "mean_same_condition_correlation": mean(r["pair_correlation"] for r in same_condition),
              "mean_same_condition_weak_jaccard": mean(r["weak_jaccard"] for r in same_condition),
              "cross_condition_same_seed": cross_condition,
              "mean_cross_condition_correlation": mean(r["canonical_pair_correlation"] for r in cross_condition),
              "mean_cross_condition_weak_jaccard": mean(r["weak_jaccard"] for r in cross_condition)}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
