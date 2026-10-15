"""Match random intervention magnitude to role-order intervention magnitude."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import confirmation_runner_v1 as confirmation
import role_mechanism_v1 as v1
import role_mechanism_v2 as v2

ROOT = Path(__file__).resolve().parent / "role_mechanism_v4"
SEEDS = (271001, 271002, 271003)


def evaluate_shift(query: np.ndarray, correct: np.ndarray,
                   reverse: np.ndarray, shift: np.ndarray) -> dict:
    a = correct + shift
    b = reverse - shift
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    margin = 10 * np.sum(query * (a - b), axis=1)
    return {"mean_margin": float(np.mean(margin)),
            "binary_role_accuracy": float(np.mean(margin > 0) + .5 * np.mean(margin == 0))}


def one_model(world: str, seed: int, arm: str, train: tuple, episodes: dict) -> dict:
    confirmation.configure_modules()
    confirmation.gate.configure(world)
    torch.set_num_threads(2)
    path = confirmation.model_path(world, seed, arm) / "model.pt"
    model = confirmation.auxiliary.RoleListener()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    scene, labels, pairs = train
    observed = {tuple(x) for x in confirmation.MANIFEST["worlds"][world]["labeled_edges"]}
    selected = np.asarray([tuple(pair) in observed for pair in pairs])
    assert selected.sum() == 640
    x = v1.scene_states(model, scene[selected])["final"]
    _, weight = v1.ridge_roles(x, labels[selected], x, labels[selected])
    role_project = v2.project(weight[:, :10] - weight[:, 10:])
    vectors = [v1.embeddings(model, episodes[pair]) for pair in episodes if pair not in observed]
    q, correct, reverse = (np.concatenate([v[i] for v in vectors]) for i in range(3))
    delta = reverse - correct
    role_shift = delta @ role_project
    target_norm = np.linalg.norm(role_shift, axis=1, keepdims=True)
    assert np.all(target_norm > 1e-8)
    ordinary = v1.assess_swap(q, correct, reverse, None)
    role = evaluate_shift(q, correct, reverse, role_shift)
    random_controls = []
    for index in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([750107, int(world[-1]), index]))
        basis, _ = np.linalg.qr(rng.normal(size=(48, 9)), mode="reduced")
        direction = delta @ (basis @ basis.T)
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        assert np.all(direction_norm > 1e-8)
        matched_shift = direction / direction_norm * target_norm
        random_controls.append(evaluate_shift(q, correct, reverse, matched_shift))
    return {"world":world,"seed":seed,"arm":arm,
            "model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "episodes":len(q),"ordinary":ordinary,"role_order":role,
            "matched_energy_random":random_controls,
            "mean_role_shift_norm":float(target_norm.mean())}


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
    records = [one_model(args.world, seed, arm, train, episodes)
               for seed in SEEDS for arm in v2.ARMS]
    output = {"protocol_sha256":hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
              "records":records}
    destination.write_text(json.dumps(output,indent=2)+"\n")
    brief=[]
    for arm in v2.ARMS:
        subset=[r for r in records if r['arm']==arm]
        contrasts=[np.mean([c['mean_margin'] for c in r['matched_energy_random']])
                   -r['role_order']['mean_margin'] for r in subset]
        brief.append({"arm":arm,"mean_matched_random_minus_role_margin":float(np.mean(contrasts)),
                      "starts":len(subset)})
    print(json.dumps({"world":args.world,"summary":brief},indent=2),flush=True)


if __name__ == "__main__":
    main()
