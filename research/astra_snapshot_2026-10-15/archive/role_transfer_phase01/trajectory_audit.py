"""Evaluate already saved original-component trajectories on the all-pair suite."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import torch

import all_pair_audit as phase

ROOT = Path(__file__).resolve().parent
SOURCE = phase.STUDY / "results" / "component_20261006" / "original"
OUT = ROOT / "trajectory_results.json"
STEPS = ((0, "step0.pt"), (1000, "step1000.pt"),
         (3000, "step3000.pt"), (6000, "model.pt"))


def run() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    phase.base.N_ID = 10
    phase.base.VOCAB = 12
    torch.set_num_threads(4)
    world_info = phase.SCREEN_WORLDS["new2"]
    suites = {pair: phase.rows_for_pair(*pair, index)
              for index, pair in enumerate(itertools.combinations(range(10), 2))}
    results = []
    for condition in ("shared1", "shared2"):
        edges = phase.mapped_edges(world_info, condition)
        for seed in (10402, 10403):
            directory = SOURCE / condition / f"seed_{seed}"
            stages = []
            for step, filename in STEPS:
                path = directory / filename
                model = phase.study.prior.Listener10()
                model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
                pair_scores = []
                for pair, rows in suites.items():
                    pair_scores.append({"pair": list(pair), "observed": pair in edges,
                                        **phase.score_pair(model, rows)})
                stages.append({"step": step, "checkpoint_sha256": phase.sha256(path),
                               "pairs": pair_scores})
                print(condition, seed, step, flush=True)
            results.append({"world": "new2", "condition": condition,
                            "seed": seed, "stages": stages})
    OUT.write_text(json.dumps({"protocol": "PROTOCOL.json", "models": results}, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    run()
