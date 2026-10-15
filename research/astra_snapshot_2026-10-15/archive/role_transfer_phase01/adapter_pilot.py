"""Restricted frozen-score adapter on five historical focal-pair failures."""
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
OUT = ROOT / "adapter_pilot_results.json"
CASES = (
    ("new0", "shared2", 10401),
    ("new0", "shared2", 10402),
    ("new2", "shared2", 10402),
    ("new2", "shared2", 10403),
    ("new3", "shared1", 10403),
)


@torch.no_grad()
def extract(model: torch.nn.Module, rows: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, c, labels = [], [], []
    model.eval()
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        messages = phase.base.oracle([r[0] for r in batch], torch.device("cpu"))
        scenes = torch.from_numpy(np.stack([r[1] for r in batch]))
        q.append(model.query(messages))
        c.append(F.normalize(model.candidate(scenes.reshape(-1, 3, 18)).reshape(-1, 4, 48), dim=-1))
        labels.append(torch.tensor([r[2] for r in batch]))
    return torch.cat(q), torch.cat(c), torch.cat(labels)


def familiar_rows(world: str, condition: str, scene_seed: int) -> list[tuple]:
    phase.study.WORLDS = phase.SCREEN_WORLDS
    phase.study.configure(world)
    pool = phase.base.make_pool("canonical")
    remap = phase.study.mapping(world, condition)
    return [phase.study.mapped_example(spec, scene_seed, i, remap)
            for i, spec in enumerate(pool)]


@torch.no_grad()
def evaluate_pair(model: torch.nn.Module, pair_rows: tuple[np.ndarray, ...], w: torch.Tensor) -> dict:
    messages, scenes, labels, categories = pair_rows
    selected = []
    margins = []
    binary = []
    model.eval()
    for start in range(0, len(labels), 128):
        stop = start + 128
        msg = F.one_hot(torch.tensor(messages[start:stop]), 12).float()
        cand = torch.from_numpy(scenes[start:stop])
        q = model.query(msg)
        c = F.normalize(model.candidate(cand.reshape(-1, 3, 18)).reshape(-1, 4, 48), dim=-1)
        scores = (10 * torch.einsum("bd,df,bkf->bk", q, w, c)).numpy()
        ix = np.arange(len(scores))
        category = categories[start:stop]
        label = labels[start:stop]
        reverse = np.argmax(category == 1, axis=1)
        margin = scores[ix, label] - scores[ix, reverse]
        selected.extend(category[ix, scores.argmax(axis=1)].tolist())
        margins.extend(margin.tolist())
        binary.extend(((margin > 0).astype(float) + .5 * (margin == 0)).tolist())
    fraction = np.bincount(selected, minlength=4) / len(selected)
    return {"four_choice_accuracy": float(fraction[0]),
            "binary_role_accuracy": float(np.mean(binary)),
            "event_accuracy": float(fraction[0] + fraction[1]),
            "mean_correct_minus_reverse_margin": float(np.mean(margins))}


def train_one(world: str, condition: str, seed: int) -> dict:
    phase.study.WORLDS = phase.SCREEN_WORLDS
    phase.study.configure(world)
    model_path = phase.SCREEN_ROOT / world / condition / f"seed_{seed}" / "model.pt"
    assert phase.sha256(model_path) == next(c["model_sha256"] for r in
        json.loads((ROOT / "phase0_audit.json").read_text())["records"]
        if r["cohort"] == "screen" and r["world"] == world and r["condition"] == condition
        for c in r["checkpoints"] if c["seed"] == seed)
    model = phase.study.prior.Listener10()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    train = extract(model, familiar_rows(world, condition, scene_seed=360107))
    valid = extract(model, familiar_rows(world, condition, scene_seed=360108))
    generator = torch.Generator().manual_seed(460107 + seed)
    u = nn.Parameter(.01 * torch.randn(48, 4, generator=generator))
    v = nn.Parameter(torch.zeros(48, 4))
    optimizer = torch.optim.Adam((u, v), lr=.01)
    identity = torch.eye(48)
    best = None
    path = []
    for step in range(801):
        if step % 100 == 0:
            with torch.no_grad():
                w = identity + u @ v.T
                score = 10 * torch.einsum("bd,df,bkf->bk", valid[0], w, valid[1])
                val_loss = float(F.cross_entropy(score, valid[2]))
                val_acc = float((score.argmax(-1) == valid[2]).float().mean())
                path.append({"step": step, "familiar_validation_loss": val_loss,
                             "familiar_validation_accuracy": val_acc})
                if best is None or val_loss < best["familiar_validation_loss"]:
                    best = {"step": step, "familiar_validation_loss": val_loss,
                            "weight": w.detach().clone()}
        if step == 800:
            break
        optimizer.zero_grad(set_to_none=True)
        w = identity + u @ v.T
        score = 10 * torch.einsum("bd,df,bkf->bk", train[0], w, train[1])
        loss = F.cross_entropy(score, train[2]) + .01 * (w - identity).square().sum()
        loss.backward()
        optimizer.step()
    assert best is not None
    edges = phase.mapped_edges(phase.SCREEN_WORLDS[world], condition)
    all_pair_scores = []
    for pair_index, pair in enumerate(itertools.combinations(range(10), 2)):
        rows = phase.rows_for_pair(*pair, pair_index)
        all_pair_scores.append({"pair": list(pair), "observed": pair in edges,
                                **evaluate_pair(model, rows, best["weight"])})
    return {"world": world, "condition": condition, "seed": seed,
            "model_sha256": phase.sha256(model_path),
            "selected_adapter_step": best["step"],
            "familiar_validation_loss": best["familiar_validation_loss"],
            "familiar_validation_path": path,
            "weight_deviation_frobenius": float(torch.norm(best["weight"] - identity)),
            "pairs": all_pair_scores}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(4)
    results = []
    for case in CASES:
        result = train_one(*case)
        results.append(result)
        focal = next(x for x in result["pairs"] if x["pair"] == [0, 1])
        print(case, result["selected_adapter_step"], focal["binary_role_accuracy"], flush=True)
    OUT.write_text(json.dumps({"protocol": "PROTOCOL.json", "status": "development_diagnostic_only",
                               "models": results}, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
