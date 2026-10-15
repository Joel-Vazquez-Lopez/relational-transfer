"""Observed-only role probes and projected scene-state swaps on frozen fresh-graph models."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import confirmation_runner_v1 as confirmation
import all_pair_audit as evaluator

ROOT = Path(__file__).resolve().parent / "role_mechanism_v1"
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
SEED = 271001
ARMS = ("unordered_identities", "ordered_roles")
RIDGE = 0.1


def scene_suite(scene_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenes, labels, pairs = [], [], []
    for pair_index, (a, b) in enumerate(itertools.combinations(range(10), 2)):
        for direction, event in itertools.product(range(2), range(2)):
            src, tgt = (a, b) if direction == 0 else (b, a)
            meaning = confirmation.gate.base.Meaning(src, tgt, event)
            for distractor in (i for i in range(10) if i not in (a, b)):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [scene_seed, pair_index, direction, event, distractor]))
                scenes.append(confirmation.gate.base.scene(meaning, distractor, rng))
                labels.append((src, tgt))
                pairs.append((a, b))
    return np.stack(scenes), np.asarray(labels), np.asarray(pairs)


@torch.no_grad()
def scene_states(model: torch.nn.Module, scenes: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    batches = {"block1": [], "block2": [], "final": []}
    encoder = model.candidate
    for start in range(0, len(scenes), 128):
        scene = torch.from_numpy(scenes[start:start + 128])
        x = torch.cat((encoder.cls.expand(len(scene), -1, -1), encoder.inp(scene)), dim=1) + encoder.pos
        for index, block in enumerate(encoder.blocks, 1):
            x = block(x)
            batches[f"block{index}"].append(x[:, 0].cpu().numpy())
        final = F.normalize(encoder.norm(x[:, 0]), dim=-1)
        reference = F.normalize(encoder(scene), dim=-1)
        assert torch.allclose(final, reference, atol=1e-6)
        batches["final"].append(final.cpu().numpy())
    return {stage: np.concatenate(values) for stage, values in batches.items()}


def ridge_roles(train_x: np.ndarray, train_y: np.ndarray,
                test_x: np.ndarray, test_y: np.ndarray) -> tuple[dict, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    x = (train_x - mean) / scale
    test = (test_x - mean) / scale
    targets = np.concatenate((np.eye(10)[train_y[:, 0]],
                              np.eye(10)[train_y[:, 1]]), axis=1)
    weight = np.linalg.solve(x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ targets)
    intercept = targets.mean(axis=0)
    logits = test @ weight + intercept
    source = logits[:, :10].argmax(axis=1)
    target = logits[:, 10:].argmax(axis=1)
    scores = {"source_accuracy": float(np.mean(source == test_y[:, 0])),
              "target_accuracy": float(np.mean(target == test_y[:, 1])),
              "joint_accuracy": float(np.mean((source == test_y[:, 0])
                                              & (target == test_y[:, 1]))),
              "train_n": len(train_x), "test_n": len(test_x)}
    return scores, weight / scale[:, None]


def projection(weight: np.ndarray) -> np.ndarray:
    basis, singular, _ = np.linalg.svd(weight, full_matrices=False)
    assert basis.shape == (48, 20) and singular[17] > 1e-8
    basis = basis[:, :18]
    return basis @ basis.T


def eval_rows() -> dict[tuple[int, int], tuple[np.ndarray, ...]]:
    evaluator.TEST_SEED = 260108
    result = {}
    for pair_index, pair in enumerate(itertools.combinations(range(10), 2)):
        data = evaluator.rows_for_pair(*pair, pair_index)
        sample = tuple(item[::4].copy() for item in data)
        assert len(sample[2]) == 128 and np.array_equal(
            np.bincount(sample[2], minlength=4), np.full(4, 32))
        sample[0][:, 3] = confirmation.gate.PAD_ID
        result[pair] = sample
    return result


@torch.no_grad()
def embeddings(model: torch.nn.Module, data: tuple[np.ndarray, ...]
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    messages, scenes, labels, categories = data
    msg = F.one_hot(torch.from_numpy(messages), confirmation.gate.VOCAB).float()
    query = model.query(msg)
    candidate = F.normalize(model.candidate(torch.from_numpy(
        scenes.reshape(-1, 3, 18))).reshape(-1, 4, 48), dim=-1)
    index = torch.arange(len(labels))
    reverse = torch.from_numpy(np.argmax(categories == 1, axis=1))
    correct = torch.from_numpy(labels)
    return (query.cpu().numpy(), candidate[index, correct].cpu().numpy(),
            candidate[index, reverse].cpu().numpy())


def assess_swap(query: np.ndarray, correct: np.ndarray, reverse: np.ndarray,
                project: np.ndarray | None) -> dict:
    if project is None:
        a, b = correct, reverse
    else:
        shift = (reverse - correct) @ project
        a = correct + shift
        b = reverse - shift
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
    margin = 10 * np.sum(query * (a - b), axis=1)
    return {"mean_margin": float(np.mean(margin)),
            "binary_role_accuracy": float(np.mean(margin > 0) + .5 * np.mean(margin == 0))}


def one_model(world: str, arm: str, train_suite: tuple,
              test_suite: tuple, episodes: dict) -> dict:
    confirmation.configure_modules()
    confirmation.gate.configure(world)
    torch.set_num_threads(2)
    path = confirmation.model_path(world, SEED, arm) / "model.pt"
    model = confirmation.auxiliary.RoleListener()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_edges = {tuple(edge) for edge in confirmation.MANIFEST["worlds"][world]["labeled_edges"]}
    train_scene, train_y, train_pair = train_suite
    test_scene, test_y, test_pair = test_suite
    seen = np.array([tuple(p) in manifest_edges for p in train_pair])
    unseen = np.array([tuple(p) not in manifest_edges for p in test_pair])
    assert int(seen.sum()) == 640 and int(unseen.sum()) == 800
    train_state = scene_states(model, train_scene[seen])
    test_state = scene_states(model, test_scene[unseen])
    probes = {}
    role_weight = None
    for stage in ("block1", "block2", "final"):
        scores, weight = ridge_roles(train_state[stage], train_y[seen],
                                     test_state[stage], test_y[unseen])
        probes[stage] = scores
        if stage == "final":
            role_weight = weight
    assert role_weight is not None
    role_project = projection(role_weight)
    random_projects = []
    for index in range(5):
        rng = np.random.default_rng(np.random.SeedSequence([730107, int(world[-1]), index]))
        basis, _ = np.linalg.qr(rng.normal(size=(48, 18)), mode="reduced")
        random_projects.append(basis @ basis.T)
    vectors = [embeddings(model, episodes[pair]) for pair in episodes if pair not in manifest_edges]
    q, correct, reverse = (np.concatenate([v[i] for v in vectors]) for i in range(3))
    assert len(q) == 25 * 128
    ordinary = assess_swap(q, correct, reverse, None)
    role = assess_swap(q, correct, reverse, role_project)
    controls = [assess_swap(q, correct, reverse, p) for p in random_projects]
    full = assess_swap(q, reverse, correct, None)
    return {"world": world, "seed": SEED, "arm": arm,
            "model_sha256": checksum, "probe": probes,
            "intervention": {"ordinary": ordinary, "role_subspace_swap": role,
                             "random_subspace_swaps": controls,
                             "full_swap": full,
                             "episodes": len(q), "subspace_rank": 18}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=confirmation.WORLDS, required=True)
    args = parser.parse_args()
    destination = ROOT / f"{args.world}.json"
    if destination.exists():
        raise FileExistsError(destination)
    confirmation.audit()
    confirmation.configure_modules()
    train = scene_suite(620107)
    test = scene_suite(620108)
    episodes = eval_rows()
    records = [one_model(args.world, arm, train, test, episodes) for arm in ARMS]
    output = {"protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(),
              "records": records}
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"world": args.world, "summary": [{
        "arm": r["arm"], "probe_final_joint": r["probe"]["final"]["joint_accuracy"],
        "ordinary": r["intervention"]["ordinary"],
        "role_swap": r["intervention"]["role_subspace_swap"],
        "random_mean_accuracy": float(np.mean([
            c["binary_role_accuracy"] for c in r["intervention"]["random_subspace_swaps"]]))
    } for r in records]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
