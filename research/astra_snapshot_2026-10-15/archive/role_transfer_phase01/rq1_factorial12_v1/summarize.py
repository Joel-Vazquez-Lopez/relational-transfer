"""Prespecified family-level analysis; refuses to run until all 96 models exist."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
SEEDS = (312001, 312002, 312003)
CELLS = ("cn1_p3", "cn1_p4", "cn2_p3", "cn2_p4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signflip(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    return sum(abs(float(np.mean(np.asarray(values) * np.asarray(signs)))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / 2 ** len(values)


def interval(values: list[float]) -> list[float]:
    half = float(t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    mean = float(np.mean(values))
    return [mean - half, mean + half]


def main() -> None:
    manifest = json.loads((HERE / "GRAPH_MANIFEST.json").read_text())
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    assert sha256(HERE / "GRAPH_MANIFEST.json") == protocol["graph_manifest_sha256"]
    rows = []
    missing = []
    for family in range(8):
        scores = {}
        familiar = {}
        by_seed = {}
        for cell, seed in itertools.product(CELLS, SEEDS):
            directory = HERE / "full" / f"family{family}" / cell / f"seed_{seed}"
            path = directory / "metrics.json"
            if not path.exists():
                missing.append((family, cell, seed))
                continue
            data = json.loads(path.read_text())
            assert data["model_sha256"] == sha256(directory / "model.pt")
            assert data["manifest_sha256"] == protocol["graph_manifest_sha256"]
            assert data["steps"] == 6000 and data["seed"] == seed
            by_seed.setdefault(seed, {})[cell] = data["focal"]["260107"]["binary_role_accuracy"]
            scores.setdefault(cell, []).append(data["focal"]["260107"]["binary_role_accuracy"])
            familiar.setdefault(cell, []).append(data["familiar_four_choice"])
        if missing:
            continue
        means = {cell: float(np.mean(scores[cell])) for cell in CELLS}
        cn = ((means["cn2_p3"] - means["cn1_p3"]) +
              (means["cn2_p4"] - means["cn1_p4"])) / 2
        p3 = ((means["cn1_p4"] - means["cn1_p3"]) +
              (means["cn2_p4"] - means["cn2_p3"])) / 2
        interaction = (means["cn2_p4"] - means["cn2_p3"] -
                       means["cn1_p4"] + means["cn1_p3"])
        rows.append({"family": family, "generator_seed": manifest["families"][family]["generator_seed"],
                     "cell_means": means, "by_seed": by_seed, "cn_effect": cn,
                     "p3_effect": p3, "interaction": interaction,
                     "familiar_cell_min": {cell: min(familiar[cell]) for cell in CELLS}})
    if missing:
        raise RuntimeError(f"Analysis sealed until complete: {len(missing)} of 96 missing")
    effects = {name: [row[name] for row in rows] for name in ("cn_effect", "p3_effect", "interaction")}
    p_values = {name: signflip(values) for name, values in effects.items()}
    primary_sorted = sorted((p_values[name], name) for name in ("cn_effect", "p3_effect"))
    holm = {primary_sorted[0][1]: min(1, primary_sorted[0][0] * 2),
            primary_sorted[1][1]: max(min(1, primary_sorted[0][0] * 2), primary_sorted[1][0])}
    overall = {name: {"mean": float(np.mean(values)), "interval_95pct": interval(values),
                      "p_exact_two_sided": p_values[name],
                      "p_holm": holm.get(name), "family_effects": values}
               for name, values in effects.items()}
    min_familiar = min(value for row in rows for value in row["familiar_cell_min"].values())
    success = any(holm[name] <= .05 and abs(overall[name]["mean"]) >= .10
                  for name in ("cn_effect", "p3_effect")) and min_familiar >= .95
    result = {"complete_models": 96, "manifest_sha256": sha256(HERE / "GRAPH_MANIFEST.json"),
              "protocol_sha256": sha256(HERE / "PROTOCOL.json"),
              "family_rows": rows, "effects": overall, "min_familiar": min_familiar,
              "prespecified_practical_success": success,
              "interpretation": "Counts are experimentally varied within tightly matched one-swap graph families, but other local graph motifs can also change."}
    destination = HERE / "ANALYSIS.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete_models": 96, "effects": overall,
                      "min_familiar": min_familiar,
                      "prespecified_practical_success": success}, indent=2))


if __name__ == "__main__":
    main()
