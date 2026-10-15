"""Capacity-checked role decoders fitted only on observed semantic relations."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import all_pair_audit as phase

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "role_decoder_results.json"
WEAK = (("new0", "shared2", 10401), ("new0", "shared2", 10402),
        ("new2", "shared2", 10402), ("new2", "shared2", 10403),
        ("new3", "shared1", 10403))
CASES = WEAK + tuple((world, "shared1" if condition == "shared2" else "shared2", seed)
                     for world, condition, seed in WEAK)


def all_meanings(edges: set[tuple[int, int]], observed: bool) -> list[tuple[int, int, int]]:
    pairs = [p for p in itertools.combinations(range(10), 2) if (p in edges) == observed]
    return [(s, t, event) for a, b in pairs for s, t in ((a, b), (b, a)) for event in (0, 1)]


@torch.no_grad()
def representations(model: torch.nn.Module, meanings: list[tuple[int, int, int]],
                    scene_seed: int) -> dict:
    query_vectors = []
    scene_vectors = []
    scene_labels = []
    model.eval()
    for index, (src, tgt, event) in enumerate(meanings):
        meaning = phase.base.Meaning(src, tgt, event)
        message = phase.base.oracle([meaning], torch.device("cpu"))
        query_vectors.append(model.query(message)[0].numpy())
        others = [i for i in range(10) if i not in (src, tgt)]
        scene_batch = np.stack([
            phase.base.scene(meaning, distractor,
                np.random.default_rng(np.random.SeedSequence([scene_seed, index, distractor])))
            for distractor in others
        ])
        representation = F.normalize(model.candidate(torch.from_numpy(scene_batch)), dim=-1)
        scene_vectors.extend(representation.numpy())
        scene_labels.extend([(src, tgt, event)] * len(others))
    return {"query_x": np.stack(query_vectors), "query_y": np.array(meanings),
            "scene_x": np.stack(scene_vectors), "scene_y": np.array(scene_labels)}


def network(family: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if family == "linear":
        return nn.Linear(48, 22)
    return nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 22))


def loss_and_accuracy(net: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    z = net(x)
    loss = F.cross_entropy(z[:, :10], y[:, 0]) + F.cross_entropy(z[:, 10:20], y[:, 1]) + F.cross_entropy(z[:, 20:], y[:, 2])
    source = z[:, :10].argmax(-1) == y[:, 0]
    target = z[:, 10:20].argmax(-1) == y[:, 1]
    event = z[:, 20:].argmax(-1) == y[:, 2]
    return loss, {"source": float(source.float().mean()),
                  "target": float(target.float().mean()),
                  "roles": float((source & target).float().mean()),
                  "event": float(event.float().mean())}


def train_probe(train_x, train_y, val_x, val_y, family: str, seed: int) -> tuple[int, list]:
    tx = torch.from_numpy(train_x).float()
    ty = torch.from_numpy(train_y).long()
    vx = torch.from_numpy(val_x).float()
    vy = torch.from_numpy(val_y).long()
    net = network(family, seed)
    opt = torch.optim.Adam(net.parameters(), lr=.01)
    best_step, best_loss = 0, float("inf")
    path = []
    for step in range(1, 1001):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_and_accuracy(net, tx, ty)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                vloss, vmetrics = loss_and_accuracy(net, vx, vy)
            value = float(vloss)
            path.append({"step": step, "observed_meaning_val_loss": value,
                         "observed_meaning_val_roles": vmetrics["roles"]})
            if value < best_loss:
                best_step, best_loss = step, value
    return best_step, path


def refit_and_score(x, y, held_x, held_y, family: str, seed: int, steps: int) -> dict:
    tx, ty = torch.from_numpy(x).float(), torch.from_numpy(y).long()
    hx, hy = torch.from_numpy(held_x).float(), torch.from_numpy(held_y).long()
    net = network(family, seed)
    opt = torch.optim.Adam(net.parameters(), lr=.01)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_and_accuracy(net, tx, ty)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, familiar = loss_and_accuracy(net, tx, ty)
        _, unseen = loss_and_accuracy(net, hx, hy)
    return {"familiar_training": familiar, "unseen": unseen}


def one_case(world: str, condition: str, seed: int) -> dict:
    phase.study.WORLDS = phase.SCREEN_WORLDS
    phase.study.configure(world)
    model_path = phase.SCREEN_ROOT / world / condition / f"seed_{seed}" / "model.pt"
    model = phase.study.prior.Listener10()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    edges = phase.mapped_edges(phase.SCREEN_WORLDS[world], condition)
    familiar = representations(model, all_meanings(edges, True), 560107)
    unseen = representations(model, all_meanings(edges, False), 560108)
    assert len(familiar["query_x"]) == 80 and len(unseen["query_x"]) == 100
    rng = np.random.default_rng(260109)
    val_meanings = set(rng.choice(80, size=16, replace=False).tolist())
    measures = {}
    for tower in ("query", "scene"):
        x = familiar[f"{tower}_x"]
        y = familiar[f"{tower}_y"]
        hx = unseen[f"{tower}_x"]
        hy = unseen[f"{tower}_y"]
        per_meaning = 1 if tower == "query" else 8
        train_ix = [i for i in range(len(x)) if (i // per_meaning) not in val_meanings]
        val_ix = [i for i in range(len(x)) if (i // per_meaning) in val_meanings]
        assert len(train_ix) == 64 * per_meaning and len(val_ix) == 16 * per_meaning
        measures[tower] = {}
        for family in ("linear", "mlp64"):
            probe_seed = 660107 + seed + (0 if tower == "query" else 100) + (0 if family == "linear" else 1000)
            steps, validation_path = train_probe(x[train_ix], y[train_ix], x[val_ix], y[val_ix], family, probe_seed)
            result = refit_and_score(x, y, hx, hy, family, probe_seed, steps)
            measures[tower][family] = {"selected_steps": steps, "observed_validation_path": validation_path, **result}
    return {"world": world, "condition": condition, "seed": seed,
            "model_sha256": phase.sha256(model_path), "historical_focal_weak": (world, condition, seed) in WEAK,
            "measures": measures}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(4)
    records = []
    for case in CASES:
        result = one_case(*case)
        records.append(result)
        print(case, {tower: {family: round(value["unseen"]["roles"], 3)
                              for family, value in result["measures"][tower].items()}
                     for tower in ("query", "scene")}, flush=True)
    OUT.write_text(json.dumps({"protocol": "PROTOCOL.json", "status": "development_diagnostic_only",
                               "models": records}, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
