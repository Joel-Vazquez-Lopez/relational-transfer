"""Supplementary all-45-pair diagnostic for final Phase 2 checkpoints."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import experiment as exp

HERE = Path(__file__).resolve().parent
SCENE = 260107


def run_one(index: int, condition: str, seed: int, arm: str):
    dest = exp.output_dir(index, condition, seed, arm, False, exp.SWAP_WEIGHT)
    if not (dest / "metrics.json").exists():
        raise FileNotFoundError(dest / "metrics.json")
    exp.configure(index, condition)
    model = exp.auxiliary.RoleListener()
    checkpoint = dest / "model_step_6000.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    row, edges = exp.graph(index, condition)
    seen = {tuple(e) for e in edges}
    focal = tuple(row["focal_pair"])
    results = []
    for pair_index, pair in enumerate(itertools.combinations(range(10), 2)):
        score = exp.eval_pair(model, *pair, pair_index, SCENE, repeats=1)
        results.append({"pair": pair, "observed": pair in seen,
                        "focal": pair == focal, **score})
    groups = {"seen": [r for r in results if r["observed"]],
              "unseen_nonfocal": [r for r in results if not r["observed"] and not r["focal"]]}
    summary = {group: {metric: float(np.mean([r[metric] for r in xs]))
                       for metric in ("four_choice_accuracy", "binary_role_accuracy", "event_accuracy")}
               for group, xs in groups.items()}
    output = {"family": index, "condition": condition, "seed": seed, "arm": arm,
              "scene_seed": SCENE, "renderings_per_pair": 128,
              "model_sha256": exp.digest(checkpoint), "summary": summary, "pairs": results}
    (dest / "all_pair_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def main():
    complete = 0
    for i, condition, seed, arm in itertools.product(
            range(8), exp.CONDITIONS, exp.SEEDS, exp.ARMS):
        dest = exp.output_dir(i, condition, seed, arm, False, exp.SWAP_WEIGHT)
        if (dest / "all_pair_audit.json").exists():
            complete += 1
            continue
        run_one(i, condition, seed, arm)
        complete += 1
        print(f"all_pair_audit {complete}/160", flush=True)
    print("ALL_PAIR_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
