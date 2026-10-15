"""Ability-only calibration of ordered-role loss strength on examined graphs.

This module never evaluates the withheld focal pair during calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path.insert(0, str(PHASE / "phase2_development_v1"))
import experiment as prior  # noqa: E402

WEIGHTS = (0.001, 0.005, 0.02, 0.08)
SEEDS = prior.SEEDS
CONDITIONS = prior.CONDITIONS
STEPS = prior.TRAIN_STEPS
MARKS = prior.CHECKPOINTS
SCENE = 260107


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(weight: float) -> str:
    assert weight in WEIGHTS
    return format(weight, ".3g").replace(".", "p")


def dest(family: int, condition: str, seed: int, weight: float) -> Path:
    return HERE / "models" / f"family{family}" / condition / f"seed_{seed}" / f"weight_{label(weight)}"


def protocol() -> dict:
    return json.loads((HERE / "PROTOCOL.json").read_text())


def audit() -> None:
    p = protocol()
    assert p["source_sha256"] == digest(Path(__file__))
    assert p["phase2_source_sha256"] == digest(Path(prior.__file__))
    assert p["graph_manifest_sha256"] == digest(prior.SOURCE_MANIFEST)
    assert tuple(p["weights"]) == WEIGHTS
    assert tuple(p["seeds"]) == SEEDS
    assert p["steps"] == STEPS
    assert p["scene_seed"] == SCENE


def ability_evaluation(model: torch.nn.Module, family: int, condition: str) -> dict:
    prior.configure(family, condition)
    row, edges = prior.graph(family, condition)
    focal = tuple(row["focal_pair"])
    seen = {tuple(edge) for edge in edges}
    pairs = list(itertools.combinations(range(10), 2))
    indexed = {pair: i for i, pair in enumerate(pairs)}
    nonfocal = [pair for pair in pairs if pair not in seen and pair != focal][:8]
    familiar = [pair for pair in pairs if pair in seen][:4]
    assert len(nonfocal) == 8 and len(familiar) == 4
    nonfocal_rows = [prior.eval_pair(model, *pair, indexed[pair], SCENE) for pair in nonfocal]
    familiar_rows = [prior.eval_pair(model, *pair, indexed[pair], SCENE) for pair in familiar]
    return {
        "nonfocal_binary_role": float(np.mean([r["binary_role_accuracy"] for r in nonfocal_rows])),
        "familiar_four_choice": float(np.mean([r["four_choice_accuracy"] for r in familiar_rows])),
        "nonfocal_pairs": [list(pair) for pair in nonfocal],
        "familiar_pairs": [list(pair) for pair in familiar],
        "renderings_per_pair": 128,
    }


def train(family: int, condition: str, seed: int, weight: float) -> None:
    audit()
    assert family in range(8) and condition in CONDITIONS and seed in SEEDS and weight in WEIGHTS
    output = dest(family, condition, seed, weight)
    final = output / "metrics.json"
    if final.exists():
        raise FileExistsError(final)
    output.mkdir(parents=True, exist_ok=True)
    prior.configure(family, condition)
    prior.base.set_seed(seed)
    torch.set_num_threads(2)
    model = prior.auxiliary.RoleListener()
    optimizer = torch.optim.AdamW(model.parameters(), lr=prior.base.LR)
    prior.auxiliary.WEIGHT = weight
    state_path = output / "training_state.pt"
    start = 0
    trace = []
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assert (state["family"], state["condition"], state["seed"], state["weight"]) == (family, condition, seed, weight)
        assert state["protocol_sha256"] == digest(HERE / "PROTOCOL.json")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        start = state["step"]
        trace = state["trace"]
    t0 = time.monotonic()
    for step, rows in enumerate(prior.batches(family, condition, seed), 1):
        if step <= start:
            continue
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, detail = prior.objective(model, rows, "ordered_roles")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at {step}")
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            trace.append({"step": step, **detail})
        if step in MARKS:
            payload = {"step": step, "family": family, "condition": condition,
                       "seed": seed, "weight": weight, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                       "trace": trace, "protocol_sha256": digest(HERE / "PROTOCOL.json")}
            torch.save(payload, state_path)
            print(json.dumps({"checkpoint": str(output), "step": step}), flush=True)
        if step == STEPS:
            break
    assert step == STEPS
    model.eval()
    ability = ability_evaluation(model, family, condition)
    snapshot = output / "model_step_6000.pt"
    torch.save(model.state_dict(), snapshot)
    record = {"family": family, "condition": condition, "seed": seed,
              "weight": weight, "steps": STEPS, "scene_seed": SCENE,
              "ability_only": ability, "model_sha256": digest(snapshot),
              "protocol_sha256": digest(HERE / "PROTOCOL.json"),
              "script_sha256": digest(Path(__file__)), "loss_trace": trace,
              "elapsed_seconds_since_resume": time.monotonic() - t0}
    final.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"completed": str(output), "ability": ability["nonfocal_binary_role"],
                      "familiar": ability["familiar_four_choice"]}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("audit", "train"))
    ap.add_argument("--family", type=int, default=0)
    ap.add_argument("--condition", choices=CONDITIONS, default="sham")
    ap.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    ap.add_argument("--weight", type=float, choices=WEIGHTS, default=WEIGHTS[0])
    args = ap.parse_args()
    if args.action == "audit":
        audit()
        prior.audit()
        print("ABILITY_MATCH_AUDIT_OK")
    else:
        train(args.family, args.condition, args.seed, args.weight)


if __name__ == "__main__":
    main()
