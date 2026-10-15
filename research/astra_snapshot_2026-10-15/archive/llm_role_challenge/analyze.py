"""Paired descriptive analysis after all six frozen checkpoint evaluations."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ("untouched_qwen05", "fresh1_source", "fresh1_natural",
         "fresh3_source", "fresh3_natural", "fresh1_copy_control")
PAIRS = (("fresh1_source", "fresh1_natural"),
         ("fresh3_source", "fresh3_natural"),
         ("fresh1_copy_control", "fresh1_natural"))


def cell(record: dict, form: str, wording: int) -> dict:
    return record["forms"][form][str(wording)]


def bootstrap_interval(deltas: list[int], draws: int = 10000) -> list[float]:
    if not deltas:
        return [float("nan"), float("nan")]
    rng = random.Random(20261012)
    n = len(deltas)
    vals = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(draws))
    return [round(vals[int(.025 * draws)], 4), round(vals[int(.975 * draws)], 4)]


def main() -> None:
    data = {}
    for name in NAMES:
        path = ROOT / f"EVAL_{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen checkpoint evaluation: {path}")
        data[name] = json.loads(path.read_text())
    hashes = {(x["frozen_items_sha256"], x["protocol_sha256"]) for x in data.values()}
    assert len(hashes) == 1
    indexed = {name: {r["id"]: r for r in obj["records"]} for name, obj in data.items()}
    ids = set(next(iter(indexed.values())))
    assert all(set(x) == ids for x in indexed.values())
    results = {}
    for before, after in PAIRS:
        result = {}
        for form in ("active", "passive"):
            for wording in (1, 2):
                key = f"{form}_wording{wording}"
                entry = {}
                for metric in ("candidate_correct", "both_roles_correct"):
                    old = [bool(cell(indexed[before][i], form, wording)[metric]) for i in sorted(ids)]
                    new = [bool(cell(indexed[after][i], form, wording)[metric]) for i in sorted(ids)]
                    deltas = [int(b) - int(a) for a, b in zip(old, new)]
                    entry[metric] = {"before_correct": sum(old), "after_correct": sum(new),
                                     "improved_items": sum(d == 1 for d in deltas),
                                     "worsened_items": sum(d == -1 for d in deltas),
                                     "gain_proportion": round(sum(deltas) / len(deltas), 4),
                                     "event_bootstrap_95pct": bootstrap_interval(deltas)}
                result[key] = entry
        results[f"{after}_vs_{before}"] = result
    target = ROOT / "PAIRED_ANALYSIS.json"
    if target.exists():
        raise FileExistsError(target)
    output = {"n_event_pairs": len(ids), "limitations":
              "Bootstrap resamples source event pairs only. It does not reflect model-family, pretraining, annotation, or checkpoint-selection uncertainty. The sample is too small for a broad generalization claim.",
              "comparisons": results}
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
