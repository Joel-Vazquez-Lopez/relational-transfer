"""Compare causal role-order and unordered-identity subspaces on frozen models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import confirmation_runner_v1 as confirmation
import role_mechanism_v1 as v1

ROOT = Path(__file__).resolve().parent / "role_mechanism_v2"
ARMS = ("unordered_identities", "ordered_roles")


def project(weight: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(weight, full_matrices=False)
    assert s[8] > 1e-8
    basis = u[:, :9]
    return basis @ basis.T


def one_model(world: str, arm: str, train: tuple, episodes: dict) -> dict:
    confirmation.configure_modules()
    confirmation.gate.configure(world)
    torch.set_num_threads(2)
    path = confirmation.model_path(world, v1.SEED, arm) / "model.pt"
    model = confirmation.auxiliary.RoleListener()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    scene, labels, pairs = train
    observed = {tuple(x) for x in confirmation.MANIFEST["worlds"][world]["labeled_edges"]}
    selected = np.asarray([tuple(pair) in observed for pair in pairs])
    assert selected.sum() == 640
    x = v1.scene_states(model, scene[selected])["final"]
    _, weight = v1.ridge_roles(x, labels[selected], x, labels[selected])
    source, target = weight[:, :10], weight[:, 10:]
    subspaces = {"role_order": project(source - target),
                 "unordered_identity": project(source + target)}
    for index in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([740107, int(world[-1]), index]))
        basis, _ = np.linalg.qr(rng.normal(size=(48, 9)), mode="reduced")
        subspaces[f"random_{index}"] = basis @ basis.T
    vectors = [v1.embeddings(model, episodes[pair]) for pair in episodes if pair not in observed]
    q, correct, reverse = (np.concatenate([v[i] for v in vectors]) for i in range(3))
    delta = reverse - correct
    total_energy = float(np.sum(delta * delta))
    results = {"ordinary": v1.assess_swap(q, correct, reverse, None),
               "full_swap": v1.assess_swap(q, reverse, correct, None)}
    for name, p in subspaces.items():
        results[name] = {**v1.assess_swap(q, correct, reverse, p),
                         "delta_energy_fraction": float(np.sum((delta @ p) ** 2) / total_energy)}
    return {"world": world, "seed": v1.SEED, "arm": arm,
            "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "episodes": len(q), "projection_rank": 9, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=confirmation.WORLDS, required=True)
    args = parser.parse_args()
    destination = ROOT / f"{args.world}.json"
    if destination.exists():
        raise FileExistsError(destination)
    confirmation.audit()
    confirmation.configure_modules()
    train = v1.scene_suite(620107)
    episodes = v1.eval_rows()
    records = [one_model(args.world, arm, train, episodes) for arm in ARMS]
    output = {"protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
              "records": records}
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"world": args.world, "summaries": [{
        "arm": r["arm"],
        "ordinary": r["results"]["ordinary"]["binary_role_accuracy"],
        "role_order": r["results"]["role_order"]["binary_role_accuracy"],
        "unordered_identity": r["results"]["unordered_identity"]["binary_role_accuracy"],
        "random_mean": float(np.mean([r["results"][f"random_{i}"]["binary_role_accuracy"]
                                      for i in range(5)]))}
        for r in records]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
