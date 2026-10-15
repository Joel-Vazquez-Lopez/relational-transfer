"""Frozen Phase-1 12-entity factorial; graph selection is outcome blind."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "d5_indirect_support"))
sys.path.insert(0, str(PROJECT / "role_transfer_phase01"))
import run as base  # noqa: E402
import generator_gate_v1 as gate  # noqa: E402

MANIFEST = HERE / "GRAPH_MANIFEST.json"
PROTOCOL = HERE / "PROTOCOL.json"
SEEDS = (312001, 312002, 312003)
CELLS = ("cn1_p3", "cn1_p4", "cn2_p3", "cn2_p4")
N = 12
VOCAB = 15
PAD = 14
TRAIN_STEPS = 6000
EVAL_SEEDS = (260107, 260108)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(edges) -> None:
    base.N_ID = N
    base.VOCAB = VOCAB
    base.BASE_EDGES = tuple(map(tuple, edges))
    base.MAPPINGS = {"canonical": tuple(range(N))}


class Listener12(base.Listener):
    def __init__(self):
        super().__init__()
        self.candidate = base.Encoder(N + 8, 3)

    def forward(self, messages: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        query = self.query(messages)
        scene = self.candidate(candidates.reshape(-1, 3, N + 8)).reshape(-1, 4, base.DIM)
        return torch.einsum("bd,bkd->bk", query, F.normalize(scene, dim=-1)) * 10


def oracle(meanings) -> torch.Tensor:
    ids = torch.tensor([[m.src, m.tgt, N + m.event, PAD] for m in meanings], dtype=torch.long)
    return F.one_hot(ids, VOCAB).float()


def schedules(pool: list, seed: int, first_epoch: int):
    """Five epochs; every semantic group receives each allowed foil equally."""
    orders = [np.random.default_rng(np.random.SeedSequence([seed, 700, epoch])).permutation(len(pool))
              for epoch in range(first_epoch, first_epoch + 5)]
    groups = defaultdict(list)
    for offset, order in enumerate(orders):
        for row, pool_index in enumerate(order):
            family, (src, tgt, third), event, _, _ = pool[int(pool_index)]
            for member, meaning in enumerate(base.qset(family, src, tgt, third, event)):
                groups[(meaning.src, meaning.tgt, meaning.event)].append((offset, row, member))
    assignments = {}
    for meaning, locations in sorted(groups.items()):
        allowed = [identity for identity in range(N) if identity not in meaning[:2]]
        assert len(locations) % len(allowed) == 0
        rng = np.random.default_rng(np.random.SeedSequence([seed, first_epoch, 801, *meaning]))
        values = np.tile(np.asarray(allowed), len(locations) // len(allowed))
        rng.shuffle(values)
        assignments.update(zip(locations, map(int, values)))
    assert len(assignments) == 5 * len(pool) * 4
    return orders, assignments


def eval_rows(a: int, b: int, scene_seed: int, repeats: int = 2):
    count = 2 * 2 * 4 * (N - 2) * repeats
    messages = np.empty((count, 4), np.int64)
    candidates = np.empty((count, 4, 3, N + 8), np.float32)
    answers = np.empty(count, np.int64)
    reverse = np.empty(count, np.int64)
    slot = 0
    for direction, event, answer, distractor, repeat in itertools.product(
        range(2), range(2), range(4), [i for i in range(N) if i not in (a, b)], range(repeats)
    ):
        src, tgt = ((a, b) if direction == 0 else (b, a))
        meanings = base.qset("R", src, tgt, distractor, event)
        rng = np.random.default_rng(np.random.SeedSequence([scene_seed, a, b, slot, 11]))
        others = [1, 2, 3]
        rng.shuffle(others)
        order = others[:answer] + [0] + others[answer:]
        for position, member in enumerate(order):
            scene_rng = np.random.default_rng(np.random.SeedSequence(
                [scene_seed, a, b, slot, 20 + position]))
            candidates[slot, position] = base.scene(meanings[member], distractor, scene_rng)
        messages[slot] = (src, tgt, N + event, PAD)
        answers[slot] = answer
        reverse[slot] = order.index(1)
        slot += 1
    assert slot == count and np.array_equal(np.bincount(answers, minlength=4), np.full(4, count // 4))
    return messages, candidates, answers, reverse


@torch.no_grad()
def score(model: Listener12, rows) -> dict:
    messages, candidates, answers, reverse = rows
    model.eval()
    correct = binary = event = 0
    for start in range(0, len(answers), 128):
        stop = min(start + 128, len(answers))
        input_ids = torch.tensor(messages[start:stop])
        logits = model(F.one_hot(input_ids, VOCAB).float(),
                       torch.from_numpy(candidates[start:stop])).cpu().numpy()
        batch = np.arange(stop - start)
        chosen = logits.argmax(axis=1)
        correct += int(np.sum(chosen == answers[start:stop]))
        binary += float(np.sum(logits[batch, answers[start:stop]] >
                               logits[batch, reverse[start:stop]]))
        event += int(np.sum((chosen == answers[start:stop]) | (chosen == reverse[start:stop])))
    return {"n": len(answers), "four_choice_accuracy": correct / len(answers),
            "binary_role_accuracy": binary / len(answers), "event_accuracy": event / len(answers)}


def protocol_audit() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    assert sha256(MANIFEST) == protocol["graph_manifest_sha256"]
    assert len(manifest["families"]) == 8
    assert tuple(protocol["design"]["model_seeds"]) == SEEDS
    assert protocol["design"]["training_updates"] == TRAIN_STEPS
    assert protocol["design"]["evaluation"].startswith("fixed 320-rendering")
    assert sorted(manifest["families"][0]["conditions"]) == sorted(CELLS)
    for family in manifest["families"]:
        for cell in CELLS:
            configure(family["conditions"][cell]["edges"])
            pool = base.make_pool("canonical")
            assert len(pool) == 1536
            seen = {tuple(sorted((spec[1][0], spec[1][1]))) for spec in pool}
            assert len(seen) == 24 and (0, 1) not in seen
            assert all(sum((meaning.src, meaning.tgt) == (src, tgt)
                           for meaning in base.qset("R", src, tgt, 2, event)) == 2
                       for src, tgt, event in ((0, 1, 0), (1, 0, 1)))
    configure(manifest["families"][0]["conditions"][CELLS[0]]["edges"])
    pool = base.make_pool("canonical")
    orders, assignment = schedules(pool, SEEDS[0], 0)
    assert len(assignment) == 5 * len(pool) * 4
    base.set_seed(SEEDS[0])
    model = Listener12()
    row = gate.episode(pool[int(orders[0][0])], SEEDS[0], 0,
                       tuple(assignment[(0, 0, member)] for member in range(4)))
    logits = model(oracle([row[0]]), torch.from_numpy(row[1][None]))
    assert logits.shape == (1, 4) and torch.isfinite(logits).all()
    focal = eval_rows(0, 1, EVAL_SEEDS[0])
    assert len(focal[2]) == 320 and set(focal[0][:, -1]) == {PAD}
    return {"valid": True, "manifest_sha256": sha256(MANIFEST),
            "protocol_sha256": sha256(PROTOCOL), "pool": len(pool),
            "block_assignments": len(assignment), "focal_renderings": len(focal[2])}


def train(family_index: int | None, cell: str | None, seed: int,
          preflight: bool = False) -> None:
    protocol_audit()
    if preflight:
        import networkx as nx
        graph = nx.random_regular_graph(4, N, seed=89999)
        if graph.has_edge(0, 1):
            a, b = next(pair for pair in itertools.combinations(range(N), 2)
                        if not graph.has_edge(*pair))
            labels = [a, b] + [node for node in range(N) if node not in (a, b)]
            graph = nx.relabel_nodes(graph, {old: new for new, old in enumerate(labels)})
        assert not graph.has_edge(0, 1)
        edges = list(map(list, sorted(tuple(sorted(edge)) for edge in graph.edges())))
        steps = 960
        output = HERE / "preflight" / "seed_311999"
        assert seed == 311999 and family_index is None and cell is None
    else:
        assert family_index is not None and 0 <= family_index < 8 and cell in CELLS and seed in SEEDS
        manifest = json.loads(MANIFEST.read_text())
        edges = manifest["families"][family_index]["conditions"][cell]["edges"]
        steps = TRAIN_STEPS
        output = HERE / "full" / f"family{family_index}" / cell / f"seed_{seed}"
    if output.exists():
        raise FileExistsError(output)
    configure(edges)
    base.set_seed(seed)
    torch.set_num_threads(4)
    model = Listener12()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR)
    pool = base.make_pool("canonical")
    assert steps * base.BATCH % len(pool) == 0
    epochs = steps * base.BATCH // len(pool)
    assert epochs % 5 == 0
    losses = []
    for first_epoch in range(0, epochs, 5):
        orders, assignments = schedules(pool, seed, first_epoch)
        for block_epoch, order in enumerate(orders):
            epoch = first_epoch + block_epoch
            for offset in range(0, len(pool), base.BATCH):
                batch = []
                for row, pool_index in enumerate(order[offset:offset + base.BATCH], start=offset):
                    serial = epoch * len(pool) + row
                    distractors = tuple(assignments[(block_epoch, row, member)] for member in range(4))
                    batch.append(gate.episode(pool[int(pool_index)], seed, serial, distractors))
                msg = oracle([item[0] for item in batch])
                scenes = torch.from_numpy(np.stack([item[1] for item in batch]))
                labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(msg, scenes), labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                if len(losses) % 500 == 0:
                    print(f"family={family_index} cell={cell} seed={seed} step={len(losses)} loss={np.mean(losses[-100:]):.4f}", flush=True)
    result = {"family": family_index, "cell": cell, "seed": seed,
              "preflight": preflight, "steps": steps, "last100_loss": float(np.mean(losses[-100:])),
              "manifest_sha256": sha256(MANIFEST), "protocol_sha256": sha256(PROTOCOL)}
    # Preflight does not inspect the held-out pair.
    if preflight:
        familiar_pairs = [tuple(edge) for edge in edges[:4]]
        result["familiar_four_choice"] = float(np.mean([
            score(model, eval_rows(a, b, EVAL_SEEDS[0], repeats=1))["four_choice_accuracy"]
            for a, b in familiar_pairs]))
    else:
        result["focal"] = {str(e): score(model, eval_rows(0, 1, e)) for e in EVAL_SEEDS}
        familiar_pairs = [tuple(edge) for edge in edges]
        result["familiar_four_choice"] = float(np.mean([
            score(model, eval_rows(a, b, EVAL_SEEDS[0], repeats=1))["four_choice_accuracy"]
            for a, b in familiar_pairs]))
    output.mkdir(parents=True)
    torch.save(model.state_dict(), output / "model.pt")
    result["model_sha256"] = sha256(output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed": str(output), "loss": result["last100_loss"],
                      "familiar": result["familiar_four_choice"],
                      "focal": result.get("focal")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "train", "preflight"))
    parser.add_argument("--family", type=int)
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "audit":
        print(json.dumps(protocol_audit(), indent=2))
    elif args.action == "preflight":
        train(None, None, 311999, True)
    else:
        train(args.family, args.cell, args.seed)
