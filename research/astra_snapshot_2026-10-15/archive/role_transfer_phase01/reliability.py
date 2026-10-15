"""Compare two independently rendered all-pair audits of the same models."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rendering_reliability.json"


def load(name: str) -> dict:
    data = json.loads((ROOT / name).read_text())
    result = {}
    for model in data["models"]:
        key = (model["cohort"], model["world"], model["condition"], model["seed"])
        for p in model["pairs"]:
            result[key + tuple(p["pair"])] = p
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    left = load("all_pair_results.json")
    right = load("all_pair_replication.json")
    assert left.keys() == right.keys()
    summaries = {}
    for observed in (False, True):
        keys = [k for k in left if left[k]["observed"] == observed]
        a = np.array([left[k]["binary_role_accuracy"] for k in keys])
        b = np.array([right[k]["binary_role_accuracy"] for k in keys])
        weak_a = a < .8
        weak_b = b < .8
        summaries["observed" if observed else "unseen"] = {
            "model_pair_cases": len(keys),
            "role_mean_seed_260107": float(a.mean()),
            "role_mean_seed_260108": float(b.mean()),
            "mean_absolute_pair_difference": float(np.abs(a - b).mean()),
            "pair_pearson_correlation": float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None,
            "weak_cases_seed_260107": int(weak_a.sum()),
            "weak_cases_seed_260108": int(weak_b.sum()),
            "weak_in_both": int((weak_a & weak_b).sum()),
            "weak_in_either": int((weak_a | weak_b).sum()),
            "threshold_disagreements": int((weak_a != weak_b).sum()),
        }
    output = {"status": "adaptive_rendering_reliability_check_not_fresh_hypothesis_confirmation",
              "results": summaries}
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
